"""Backfill the last hour of ADS-B state vectors from OpenSky.

Authenticated access to `/states/all` reaches one hour into the past at
5-second resolution, so a single run can reconstruct a continuous hour of
Southern California traffic without waiting for it to happen. That is enough
to build real trajectories, observed durations and encounter histories -- none
of which a single snapshot can support.

Direction matters. The one-hour window slides forward as the run proceeds, so
frames are requested **oldest first**: the ones closest to expiring are
collected while they are still available, and every later request is younger
than the one before it. Walking newest-first would guarantee that the oldest
frames expired before the run reached them.

Timestamps older than one hour are refused with HTTP 403 (the API docs say
400). That is a window expiry, not an authentication problem, so it is handled
separately from a genuine auth failure.

Usage:
    python -m src.data.backfill_opensky
    python -m src.data.backfill_opensky --window-minutes 30 --step-seconds 5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.data.fetch_opensky import (
    SOCAL_BBOX,
    fetch_states,
    get_access_token,
    to_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"

# The API refuses anything older than one hour. Start inside that boundary so
# the first request is not racing the cutoff.
MAX_HISTORY_SECONDS = 3_600
SAFETY_MARGIN_SECONDS = 60
# Never request anything older than this. The margin keeps the first request
# clear of the cutoff instead of racing it.
OLDEST_SAFE_AGE_SECONDS = MAX_HISTORY_SECONDS - SAFETY_MARGIN_SECONDS

# Stay a minute inside the safe age so the run starts with real headroom
# rather than sitting exactly on the boundary.
DEFAULT_WINDOW_MINUTES = 58.0
DEFAULT_STEP_SECONDS = 10.0

# A short pause keeps a few hundred sequential requests from arriving as one
# burst. It also keeps each iteration well under the step size, which is what
# stops the sliding window from overtaking the run.
REQUEST_SPACING_SECONDS = 0.3

TOKEN_REFRESH_SECONDS = 20 * 60
MAX_CONSECUTIVE_FAILURES = 5
RETRY_BACKOFF_SECONDS = 5.0


# Step 1: walk a past window oldest-first and write every distinct frame.
def backfill_window(
    bbox: dict[str, float],
    output_path: Path,
    window_seconds: float,
    step_seconds: float,
    anonymous: bool = False,
    progress_every: int = 30,
) -> dict[str, int]:
    """Collect one past window; return a summary of what was retrieved."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero.")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be greater than zero.")
    if window_seconds > OLDEST_SAFE_AGE_SECONDS:
        raise ValueError(
            "window_seconds must stay inside the one-hour history limit "
            f"(max {OLDEST_SAFE_AGE_SECONDS} with margin)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    token, token_issued_at = _acquire_token(anonymous)
    requested_time = int(time.time() - window_seconds)
    end_time = int(time.time())

    seen_frame_times: set[int] = set()
    frames_written = 0
    rows_written = 0
    duplicate_frames = 0
    expired_frames = 0
    failures = 0
    consecutive_failures = 0
    attempts = 0

    while requested_time <= end_time:
        # The window slides while the run proceeds. A frame that has aged out
        # is clamped forward to the oldest timestamp still retrievable, then
        # requested -- never looped back for another age check, which would
        # spin forever whenever the clamp lands on the boundary it tests.
        oldest_safe = int(time.time() - OLDEST_SAFE_AGE_SECONDS)
        if requested_time < oldest_safe:
            expired_frames += max(1, (oldest_safe - requested_time) // int(step_seconds))
            requested_time = oldest_safe

        if token is not None and time.monotonic() - token_issued_at > TOKEN_REFRESH_SECONDS:
            token, token_issued_at = _acquire_token(anonymous)

        attempts += 1
        try:
            payload = fetch_states(bbox, token, at_time=requested_time)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status == 403:
                # Window expiry, not an auth problem -- skip this frame.
                expired_frames += 1
                requested_time += int(step_seconds)
                continue
            failures += 1
            consecutive_failures += 1
            print(f"  HTTP {status} at t={requested_time} ({consecutive_failures})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("  too many consecutive failures -- stopping early.")
                break
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        except requests.RequestException as error:
            failures += 1
            consecutive_failures += 1
            print(f"  request failed at t={requested_time} ({consecutive_failures}): {error}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("  too many consecutive failures -- stopping early.")
                break
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        consecutive_failures = 0
        requested_time += int(step_seconds)

        frame_time = payload.get("time")
        if frame_time is None or not payload.get("states"):
            continue

        # The API snaps each request to a 5-second boundary, so neighbouring
        # requests can resolve to the same frame.
        if frame_time in seen_frame_times:
            duplicate_frames += 1
            continue

        frame = to_dataframe(payload)
        if frame.empty:
            continue

        seen_frame_times.add(frame_time)
        write_header = frames_written == 0
        frame.to_csv(
            output_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        frames_written += 1
        rows_written += len(frame)

        if progress_every and frames_written % progress_every == 0:
            print(f"  {frames_written} frames, {rows_written} rows")

        time.sleep(REQUEST_SPACING_SECONDS)

    span = max(seen_frame_times) - min(seen_frame_times) if seen_frame_times else 0
    return {
        "frames": frames_written,
        "rows": rows_written,
        "span_seconds": int(span),
        "duplicate_frames": duplicate_frames,
        "expired_frames": expired_frames,
        "failures": failures,
        "credits_spent": attempts,
    }


# Step 1: fetch a token unless the run was asked to go anonymous.
def _acquire_token(anonymous: bool) -> tuple[str | None, float]:
    if anonymous:
        return None, time.monotonic()

    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    if not (client_id and client_secret):
        raise RuntimeError(
            "Backfill needs credentials: anonymous access cannot request past "
            "timestamps. Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET in .env."
        )

    return get_access_token(client_id, client_secret), time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamin", type=float, default=SOCAL_BBOX["lamin"])
    parser.add_argument("--lomin", type=float, default=SOCAL_BBOX["lomin"])
    parser.add_argument("--lamax", type=float, default=SOCAL_BBOX["lamax"])
    parser.add_argument("--lomax", type=float, default=SOCAL_BBOX["lomax"])
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=DEFAULT_WINDOW_MINUTES,
        help="How far back to start, in minutes (default: 59).",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=DEFAULT_STEP_SECONDS,
        help="Spacing between requested frames (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the backfill CSV (default: data/raw/sessions/).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    bbox = {
        "lamin": args.lamin,
        "lomin": args.lomin,
        "lamax": args.lamax,
        "lomax": args.lomax,
    }
    run_start = int(time.time())
    output_path = args.output_dir / f"states_socal_backfill_{run_start}.csv"
    planned = int(args.window_minutes * 60 / args.step_seconds)

    print(f"Run start       : {run_start}")
    print(f"Window          : {args.window_minutes:g} min back, {args.step_seconds:g}s steps")
    print(f"Planned frames  : {planned}  (~{planned} credits)")
    print(f"Writing to      : {output_path.relative_to(PROJECT_ROOT)}")

    summary = backfill_window(
        bbox=bbox,
        output_path=output_path,
        window_seconds=args.window_minutes * 60,
        step_seconds=args.step_seconds,
        )

    print(f"Frames captured : {summary['frames']}")
    print(f"Rows written    : {summary['rows']}")
    print(f"Span covered    : {summary['span_seconds']}s")
    print(f"Duplicate frames: {summary['duplicate_frames']}")
    print(f"Expired frames  : {summary['expired_frames']}")
    print(f"Failed requests : {summary['failures']}")
    print(f"Credits spent   : ~{summary['credits_spent']}")


if __name__ == "__main__":
    main()
