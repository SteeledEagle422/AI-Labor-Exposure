"""
Fetch BLS Current Employment Statistics (CES) series via the official
public API v2.

*** THIS SCRIPT CANNOT BE RUN OR TESTED FROM WITHIN THE SANDBOX THIS PROJECT
*** WAS SCAFFOLDED IN -- api.bls.gov is not reachable from that environment.
*** Run it yourself locally; the request/response handling below follows the
*** documented BLS API v2 contract exactly (https://www.bls.gov/developers/).

Setup (one-time, free, ~instant approval):
  1. Register at https://data.bls.gov/registrationEngine/
  2. export BLS_API_KEY="your-key-here"
     (or put BLS_API_KEY=your-key-here in a .env file in the project root)

Why the API and not a scrape: v2 with a registered key allows up to 20 years
of data and 50 series per query, 500 queries/day -- plenty for this project's
handful of series. Without a key you're capped at 10 years / 25 series /
25 queries per day, which won't cover 2019-2026 in one call.

Series IDs are set in config/config.yaml under bls_ces.series and were
verified against FRED as of July 2026 (search "CES6054150001 FRED" etc. to
re-verify if BLS ever restructures CES industry codes).

Usage:
    python -m src.data.fetch_bls_ces
"""
from __future__ import annotations

import os
import sys
import json
import time

import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

BLS_DIR = RAW_DIR / "bls"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
MAX_SERIES_PER_CALL = 50  # BLS API v2 limit with a registered key


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_bls_ces(cfg: dict | None = None, api_key: str | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(BLS_DIR)

    api_key = api_key or os.environ.get("BLS_API_KEY")
    if not api_key:
        print(
            "[bls] WARNING: no BLS_API_KEY found in environment.\n"
            "       The request below will still run against the public "
            "(unregistered) endpoint, but is limited to ~10 years of history "
            "and 25 queries/day. Register a free key at "
            "https://data.bls.gov/registrationEngine/ and set BLS_API_KEY "
            "for the full 2019-2026 pull.",
            file=sys.stderr,
        )

    series_map = cfg["bls_ces"]["series"]              # name -> series_id
    series_ids = list(series_map.values())
    id_to_name = {v: k for k, v in series_map.items()}

    start_year = str(cfg["bls_ces"]["start_year"])
    end_year = str(cfg["bls_ces"]["end_year"])

    all_records = []
    for chunk in _chunk(series_ids, MAX_SERIES_PER_CALL):
        payload = {
            "seriesid": chunk,
            "startyear": start_year,
            "endyear": end_year,
        }
        if api_key:
            payload["registrationkey"] = api_key

        print(f"[bls] requesting {len(chunk)} series, {start_year}-{end_year} ...")
        resp = requests.post(
            BLS_API_URL,
            data=json.dumps(payload),
            headers={"Content-type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(
                f"BLS API request failed: {result.get('status')} -- "
                f"{result.get('message')}"
            )

        for series in result["Results"]["series"]:
            sid = series["seriesID"]
            name = id_to_name.get(sid, sid)
            for obs in series["data"]:
                if obs["period"] == "M13":  # annual average row, skip
                    continue
                all_records.append({
                    "series_name": name,
                    "series_id": sid,
                    "year": int(obs["year"]),
                    "period": obs["period"],           # "M01".."M12"
                    "month": int(obs["period"][1:]),
                    "value": float(obs["value"]),
                })
        time.sleep(0.5)  # be polite between chunked requests

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )
    df = df.sort_values(["series_name", "date"]).reset_index(drop=True)

    out_path = BLS_DIR / "bls_ces_monthly.csv"
    df.to_csv(out_path, index=False)
    print(f"[bls]   saved {out_path}  ({len(df):,} rows, "
          f"{df['series_name'].nunique()} series)")
    return df


if __name__ == "__main__":
    df = fetch_bls_ces()
    print(df.head())
    sys.exit(0)
