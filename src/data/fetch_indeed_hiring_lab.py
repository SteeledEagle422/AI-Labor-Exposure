"""
Fetch Indeed Hiring Lab job postings data.

Source: https://github.com/hiring-lab/job_postings_tracker (CC-BY-4.0)
No API key required. This is the PRIMARY outcome data (Sec 4 of the paper).

Pulls two files for the US:
  - aggregate_job_postings_US.csv  (national daily index, all sectors pooled)
  - job_postings_by_sector_US.csv  (daily index broken out by 41 occupational
    sectors -- this is what the tech/non-tech/exposure-weighted analysis uses)

Usage:
    python -m src.data.fetch_indeed_hiring_lab
"""
from __future__ import annotations

import sys
import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

INDEED_DIR = RAW_DIR / "indeed"


def _fetch_csv(url: str, out_path) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return pd.read_csv(out_path)


def fetch_indeed_data(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    ensure_dirs(INDEED_DIR)
    base = cfg["indeed"]["repo_raw_base"]
    files = cfg["indeed"]["files"]

    out = {}
    for key, rel_path in files.items():
        url = f"{base}/{rel_path}"
        out_path = INDEED_DIR / f"{key}_US.csv"
        print(f"[indeed] fetching {key} <- {url}")
        df = _fetch_csv(url, out_path)
        print(f"[indeed]   saved {out_path}  ({len(df):,} rows)")
        out[key] = df
    return out


if __name__ == "__main__":
    data = fetch_indeed_data()
    sector_df = data["sector"]
    print("\nSectors available:", sector_df["display_name"].nunique())
    print("Date range:", sector_df["date"].min(), "to", sector_df["date"].max())
    sys.exit(0)
