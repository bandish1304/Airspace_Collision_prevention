"""Poll OpenSky for a timed session of ADS-B snapshots.

Week 1 collected a single snapshot, which is enough to build pair features but
not to observe how an encounter develops. This script runs one *session*: it
repeatedly calls the same endpoint as `fetch_opensky.py` on a fixed interval
and writes every distinct frame to one CSV, producing the multi-snapshot data
the trajectory, duration and time-of-day analysis needs.

One invocation = one session, then exit. That makes it schedulable: run it at
several fixed times a day and each run drops a separate session file into
`data/raw/sessions/`.

Usage:
    python -m src.data.poll_opensky
    python -m src.data.poll_opensky --session-minutes 60 --interval-seconds 5
    python -m src.data.poll_opensky --session-minutes 1 --interval-seconds 10
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
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

DEFAULT_SESSION_MINUTES = 30.0
DEFAULT_INTERVAL_SECONDS = 10.0

# OpenSky access tokens last roughly 30 minutes. Refresh well before that so a
# long session never fails mid-flight on an expired token.
TOKEN_REFRESH_SECONDS = 20 * 60

# Give up only after several failures in a row -- a single timeout or 503 is
# routine and should not end a session that is otherwise collecting fine.
MAX_CONSECUTIVE_FAILURES = 5
RETRY_BACKOFF_SECONDS = 5.0


# Step 0: run one polling session and append every new frame to a single CSV.
def run_session(
    bbox: dict[str, float],
    output_path: Path,
    session_seconds: float,
    interval_seconds: float,
    anonymous: bool = False,
) -> dict[str, int]:
    """Poll until the session elapses; return a summary of what was collected."""
    if session_seconds <= 0:
        raise ValueError("session_seconds must be greater than zero.")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    token, token_issued_at = _acquire_token(anonymous)
    seen_snapshot_times: set[int] = set()
    frames_written = 0
    rows_written = 0
    duplicate_frames = 0
    failures = 0
    consecutive_failures = 0

    session_deadline = time.monotonic() + session_seconds
    next_request_at = time.monotonic()

    while time.monotonic() < session_deadline:
        _sleep_until(next_request_at)
        next_request_at = time.monotonic() + interval_seconds

        if token is not None and time.monotonic() - token_issued_at > TOKEN_REFRESH_SECONDS:
            token, token_issued_at = _acquire_token(anonymous)

        try:
            payload = fetch_states(bbox, token)
        except requests.RequestException as error:
            failures += 1
            consecutive_failures += 1
            print(f"  request failed ({consecutive_failures}): {error}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("  too many consecutive failures -- ending session early.")
                break
            next_request_at = time.monotonic() + RETRY_BACKOFF_SECONDS
            continue

        consecutive_failures = 0
        snapshot_time = payload.get("time")
        if snapshot_time is None or not payload.get("states"):
            continue

        # The API updates on its own cadence, so two polls can return the same
        # frame. Keeping both would fabricate duplicate observations.
        if snapshot_time in seen_snapshot_times:
            duplicate_frames += 1
            continue

        frame = to_dataframe(payload)
        if frame.empty:
            continue

        seen_snapshot_times.add(snapshot_time)
        write_header = frames_written == 0
        frame.to_csv(
            output_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        frames_written += 1
        rows_written += len(frame)

    return {
        "frames": frames_written,
        "rows": rows_written,
        "duplicate_frames": duplicate_frames,
        "failures": failures,
        "credits_spent": frames_written + duplicate_frames + failures,
    }


# Step 0: fetch a token unless the session was asked to run anonymously.
def _acquire_token(anonymous: bool) -> tuple[str | None, float]:
    if anonymous:
        return None, time.monotonic()

    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    if not (client_id and client_secret):
        print("No credentials in .env -- polling anonymously (10s resolution).")
        return None, time.monotonic()

    return get_access_token(client_id, client_secret), time.monotonic()


# Step 0: hold the requested interval without drifting as requests take time.
def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamin", type=float, default=SOCAL_BBOX["lamin"])
    parser.add_argument("--lomin", type=float, default=SOCAL_BBOX["lomin"])
    parser.add_argument("--lamax", type=float, default=SOCAL_BBOX["lamax"])
    parser.add_argument("--lomax", type=float, default=SOCAL_BBOX["lomax"])
    parser.add_argument(
        "--session-minutes",
        type=float,
        default=DEFAULT_SESSION_MINUTES,
        help="How long to keep polling (default: 30).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between polls (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the session CSV (default: data/raw/sessions/).",
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="Skip authentication and use the lower anonymous rate limit.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    bbox = {
        "lamin": args.lamin,
        "lomin": args.lomin,
        "lamax": args.lamax,
        "lomax": args.lomax,
    }
    session_start = int(time.time())
    output_path = args.output_dir / f"states_socal_session_{session_start}.csv"

    print(f"Session start   : {session_start}")
    print(f"Planned frames  : {int(args.session_minutes * 60 / args.interval_seconds)}")
    print(f"Writing to      : {output_path.relative_to(PROJECT_ROOT)}")

    summary = run_session(
        bbox=bbox,
        output_path=output_path,
        session_seconds=args.session_minutes * 60,
        interval_seconds=args.interval_seconds,
        anonymous=args.anonymous,
    )

    print(f"Frames captured : {summary['frames']}")
    print(f"Rows written    : {summary['rows']}")
    print(f"Duplicate frames: {summary['duplicate_frames']}")
    print(f"Failed requests : {summary['failures']}")
    print(f"Credits spent   : ~{summary['credits_spent']}")


if __name__ == "__main__":
    main()
