"""
Fetch Google Trends search interest for the mechanism terms in Sec 4 (Data)
of the paper -- e.g. does search interest in "learn to code" / "coding
bootcamp" move around the same event dates as the postings decline, which
would support a labor-supply-side mechanism story alongside labor demand.

*** trends.google.com is not reachable from the sandbox this project was
*** scaffolded in, so this script is written correctly against the pytrends
*** API but UNTESTED end-to-end. Run it locally.

No API key needed, but Google will throttle/CAPTCHA fast or repeated pulls.
Practical tips if you hit 429s when running this yourself:
  - Add delay between terms (already done below, `sleep_between_terms`)
  - Pull fewer terms per run
  - If it keeps failing, there are pytrends forks with proxy/backoff support

Usage:
    python -m src.data.fetch_google_trends
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

TRENDS_DIR = RAW_DIR / "trends"


def fetch_google_trends(cfg: dict | None = None, sleep_between_terms: float = 2.0) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(TRENDS_DIR)

    try:
        from pytrends.request import TrendReq
    except ImportError as e:
        raise ImportError(
            "pytrends is not installed. pip install pytrends --break-system-packages"
        ) from e

    terms = cfg["google_trends"]["terms"]
    geo = cfg["google_trends"]["geo"]
    timeframe = cfg["google_trends"]["timeframe"]

    pytrends = TrendReq(hl="en-US", tz=0)

    all_series = []
    for term in terms:
        print(f"[trends] fetching '{term}' ({geo}, {timeframe})")
        pytrends.build_payload([term], cat=0, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty:
            print(f"[trends]   WARNING: no data returned for '{term}'")
            continue
        df = df.rename(columns={term: "search_interest"})
        df["term"] = term
        df = df.reset_index()[["date", "term", "search_interest"]]
        all_series.append(df)
        time.sleep(sleep_between_terms)

    if not all_series:
        raise RuntimeError("No Google Trends series were successfully fetched.")

    out = pd.concat(all_series, ignore_index=True)
    out_path = TRENDS_DIR / "google_trends_terms.csv"
    out.to_csv(out_path, index=False)
    print(f"[trends]   saved {out_path}  ({len(out):,} rows, {out['term'].nunique()} terms)")
    return out


if __name__ == "__main__":
    df = fetch_google_trends()
    print(df.head())
    sys.exit(0)
