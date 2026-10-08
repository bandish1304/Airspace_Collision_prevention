"""Pull ADS-B state vectors from the OpenSky Network REST API.

Week 1, step 3: a basic sample pull for Southern California airspace.

Authentication uses the OAuth2 client credentials flow; OpenSky removed basic
auth (username/password) in March 2026. Credentials are read from `.env` as
OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET. Without them the request is made
anonymously, which works but carries a much lower rate limit.

This endpoint returns a *current* snapshot only. Bulk historical data
comes from OpenSky's Trino interface, which requires separate approved access.

Usage:
    python -m src.data.fetch_opensky
    python -m src.data.fetch_opensky --save-json
    python -m src.data.fetch_opensky --lamin 33.0 --lamax 34.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
STATES_URL = "https://opensky-network.org/api/states/all"

# Southern California airspace. Covers LAX, SAN, ONT, BUR, SNA and the
# surrounding en-route sectors -- dense traffic with a mix of commercial and
# general aviation, which is the point of choosing this region.
SOCAL_BBOX = {"lamin": 32.5, "lomin": -120.5, "lamax": 35.5, "lomax": -116.0}

# Field order of each state vector, per the OpenSky REST API documentation.
STATE_VECTOR_FIELDS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
    "category",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def get_access_token(client_id: str, client_secret: str) -> str:
    """Exchange client credentials for an OAuth2 access token.

    Tokens expire after roughly 30 minutes. This pulls a fresh one per run,
    which is fine for a single snapshot; the Phase 2 ingestion loop will need
    to cache and refresh instead.
    """
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_states(
    bbox: dict[str, float],
    token: str | None = None,
    at_time: int | None = None,
) -> dict:
    """Fetch a snapshot of state vectors within a bounding box.

    Passing `at_time` (Unix seconds) requests a past snapshot instead of the
    current one. Authenticated access reaches back one hour and snaps to the
    nearest 5-second boundary; older timestamps are refused with HTTP 403, not
    400 as the API documentation states.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = dict(bbox)
    if at_time is not None:
        params["time"] = int(at_time)
    response = requests.get(STATES_URL, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def to_dataframe(payload: dict) -> pd.DataFrame:
    """Turn the raw API payload into a tabular frame.

    `sensors` is dropped: it is a nested list of receiver IDs that does not fit
    a flat table and carries no signal for collision risk.

    The API omits trailing fields it has nothing to report -- `category` is
    only returned when the request sets `extended=1` -- so columns are matched
    to the width actually returned rather than assumed.
    """
    states = payload.get("states") or []
    width = len(states[0]) if states else len(STATE_VECTOR_FIELDS)
    frame = pd.DataFrame(states, columns=STATE_VECTOR_FIELDS[:width])
    frame = frame.drop(columns=["sensors"], errors="ignore")
    frame["snapshot_time"] = payload.get("time")
    if "callsign" in frame:
        frame["callsign"] = frame["callsign"].str.strip()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamin", type=float, default=SOCAL_BBOX["lamin"])
    parser.add_argument("--lomin", type=float, default=SOCAL_BBOX["lomin"])
    parser.add_argument("--lamax", type=float, default=SOCAL_BBOX["lamax"])
    parser.add_argument("--lomax", type=float, default=SOCAL_BBOX["lomax"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the snapshot (default: data/raw/).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also write the unmodified API response alongside the CSV.",
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="Skip authentication and use the lower anonymous rate limit.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    token = None
    if not args.anonymous:
        client_id = os.getenv("OPENSKY_CLIENT_ID")
        client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
        if client_id and client_secret:
            token = get_access_token(client_id, client_secret)
            print("Authenticated via OAuth2 client credentials.")
        else:
            print(
                "No credentials found in .env -- falling back to anonymous "
                "access (lower rate limit)."
            )

    bbox = {
        "lamin": args.lamin,
        "lomin": args.lomin,
        "lamax": args.lamax,
        "lomax": args.lomax,
    }
    payload = fetch_states(bbox, token)
    frame = to_dataframe(payload)

    snapshot_time = payload.get("time") or int(time.time())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"states_socal_{snapshot_time}.csv"
    frame.to_csv(csv_path, index=False)

    if args.save_json:
        json_path = args.output_dir / f"states_socal_{snapshot_time}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote raw response to {json_path.relative_to(PROJECT_ROOT)}")

    print(f"Snapshot time : {snapshot_time}")
    print(f"Aircraft      : {len(frame)}")
    print(f"Airborne      : {int((~frame['on_ground']).sum()) if len(frame) else 0}")
    print(f"Wrote         : {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
