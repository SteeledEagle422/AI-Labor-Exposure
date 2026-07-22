"""
Fetch Felten, Raj & Seamans (2021) AI Occupational Exposure (AIOE) data.

Source: https://github.com/AIOE-Data/AIOE (AIOE_DataAppendix.xlsx)
No API key required.

This is the ROBUSTNESS-CHECK exposure measure (Sec 7, "alternate exposure
measure"). It is also useful as the industry-level confirmatory measure
because Appendix B is already at 4-digit NAICS -- no SOC->NAICS crosswalk
needed for that piece.

Sheets used:
  Appendix A -- SOC-level AIOE            (occupation, needs OEWS crosswalk)
  Appendix B -- NAICS(4-digit)-level AIIE (industry, usable directly)

Usage:
    python -m src.data.fetch_aioe_exposure
"""
import sys
import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

EXPOSURE_DIR = RAW_DIR / "exposure"


def fetch_aioe_scores(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    ensure_dirs(EXPOSURE_DIR)
    url = cfg["exposure"]["aioe"]["url"]
    xlsx_path = EXPOSURE_DIR / "AIOE_DataAppendix.xlsx"

    print(f"[aioe] fetching <- {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    xlsx_path.write_bytes(resp.content)

    occ_sheet = cfg["exposure"]["aioe"]["occupation_sheet"]
    ind_sheet = cfg["exposure"]["aioe"]["industry_sheet"]

    occ_df = pd.read_excel(xlsx_path, sheet_name=occ_sheet)
    occ_df.columns = ["soc_code", "occupation_title", "aioe"]
    occ_out = EXPOSURE_DIR / "aioe_occupation_level.csv"
    occ_df.to_csv(occ_out, index=False)
    print(f"[aioe]   saved {occ_out}  ({len(occ_df):,} occupations)")

    ind_df = pd.read_excel(xlsx_path, sheet_name=ind_sheet)
    ind_df.columns = ["naics", "industry_title", "aiie"]
    ind_out = EXPOSURE_DIR / "aioe_industry_level.csv"
    ind_df.to_csv(ind_out, index=False)
    print(f"[aioe]   saved {ind_out}  ({len(ind_df):,} 4-digit NAICS industries)")

    return {"occupation": occ_df, "industry": ind_df}


if __name__ == "__main__":
    data = fetch_aioe_scores()
    print(data["occupation"].head())
    print(data["industry"].head())
    sys.exit(0)
