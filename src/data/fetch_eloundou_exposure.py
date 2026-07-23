"""
Fetch Eloundou, Manning, Mishkin & Rock (2023/2024) "GPTs are GPTs"
occupation-level LLM exposure scores.

Source: https://github.com/openai/GPTs-are-GPTs (data/occ_level.csv)
No API key required.

Columns of interest (all in [0, 1], one row per O*NET-SOC detailed occupation):
  dv_rating_alpha  -- E1: exposure using an LLM alone
  dv_rating_beta   -- E1+E2: exposure using an LLM + existing software     [DEFAULT]
  dv_rating_gamma  -- E1+E2+E3: + complementary/LLM-powered tooling
  human_rating_*   -- human-labeled counterparts to the above (dv_ = LLM-labeled)

Usage:
    python -m src.data.fetch_eloundou_exposure
"""
from __future__ import annotations

import sys
import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

EXPOSURE_DIR = RAW_DIR / "exposure"


def fetch_eloundou_scores(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(EXPOSURE_DIR)
    url = cfg["exposure"]["eloundou"]["url"]
    out_path = EXPOSURE_DIR / "eloundou_occ_level.csv"

    print(f"[eloundou] fetching <- {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

    df = pd.read_csv(out_path)
    df = df.rename(columns={"O*NET-SOC Code": "onet_soc_code", "Title": "occupation_title"})
    # SOC (6-digit, e.g. 15-1252) vs O*NET-SOC (8-digit, e.g. 15-1252.00) --
    # OEWS staffing patterns use 6-digit SOC, so keep both.
    df["soc_code"] = df["onet_soc_code"].str.slice(0, 7)
    df.to_csv(out_path, index=False)
    print(f"[eloundou]   saved {out_path}  ({len(df):,} occupations, "
          f"{df['soc_code'].nunique()} distinct 6-digit SOC codes)")
    return df


if __name__ == "__main__":
    df = fetch_eloundou_scores()
    print(df[["soc_code", "occupation_title", "dv_rating_alpha",
              "dv_rating_beta", "dv_rating_gamma"]].head())
    sys.exit(0)
