"""
Fetch and parse BLS OEWS "National Industry-Specific Occupational Employment
and Wage Estimates" -- i.e. staffing patterns: how many workers of each SOC
occupation sit inside each 4-digit NAICS industry.

*** api/download access to bls.gov is not reachable from the sandbox this
*** project was scaffolded in. This script's download step is written and
*** documented but UNTESTED end-to-end -- run it locally. If BLS has moved
*** the file (they restructure URLs across vintages), the script prints the
*** manual fallback page and expects the file to be dropped in
*** data/raw/oews/ by hand. See MANUAL STEP below.

Why this file, specifically: this is the missing link that lets us take
Eloundou et al.'s SOC-level LLM exposure scores and produce a defensible
NAICS-level (and therefore Indeed-sector-level) exposure score, instead of
picking one "representative occupation" per sector by hand. We compute:

    sector_exposure = sum_over_soc( employment_share_within_naics * exposure_soc )

aggregated across all NAICS codes mapped to that Indeed sector in
config/sector_naics_map.csv.

MANUAL STEP (if the automatic download fails):
  1. Go to https://www.bls.gov/oes/tables.htm
  2. Find "May {YEAR} National Industry-Specific Occupational Employment
     and Wage Estimates" and download the XLS/XLSX bulk file (all industries,
     4-digit NAICS). Historically named oesm{YY}in4.xlsx or .zip.
  3. Save it as:
       data/raw/oews/oesm{YY}in4.xlsx
     (unzip first if BLS ships it as a .zip)
  4. Re-run this script -- it will find the local file and skip the download.

Usage:
    python -m src.data.fetch_oews_staffing_patterns
"""
import io
import sys
import zipfile

import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

OEWS_DIR = RAW_DIR / "oews"

# Historical BLS bulk-file URL pattern for the 4-digit NAICS industry file.
# BLS does periodically restructure this -- if it 404s, use the manual step.
CANDIDATE_URL_PATTERNS = [
    "https://www.bls.gov/oes/special.requests/oesm{yy}in4.zip",
    "https://www.bls.gov/oes/special.requests/oesm{yy}in4.xlsx",
]

# Column names BLS has used across vintages for the fields we need.
COL_ALIASES = {
    "naics": ["NAICS", "naics"],
    "naics_title": ["NAICS_TITLE", "naics_title"],
    "occ_code": ["OCC_CODE", "occ_code"],
    "occ_title": ["OCC_TITLE", "occ_title"],
    "tot_emp": ["TOT_EMP", "tot_emp"],
}


def _find_local_file(yy: str) -> "pd.io.common.FilePath | None":
    for ext in ("xlsx", "xls"):
        p = OEWS_DIR / f"oesm{yy}in4.{ext}"
        if p.exists():
            return p
    return None


def _download(yy: str) -> "pd.io.common.FilePath | None":
    for pattern in CANDIDATE_URL_PATTERNS:
        url = pattern.format(yy=yy)
        print(f"[oews] attempting download <- {url}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[oews]   failed ({e})")
            continue

        if url.endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            xlsx_names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
            if not xlsx_names:
                print("[oews]   zip contained no xlsx/xls file, skipping")
                continue
            out_path = OEWS_DIR / f"oesm{yy}in4.xlsx"
            out_path.write_bytes(zf.read(xlsx_names[0]))
        else:
            out_path = OEWS_DIR / f"oesm{yy}in4.xlsx"
            out_path.write_bytes(resp.content)

        print(f"[oews]   saved {out_path}")
        return out_path

    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for target, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename[a] = target
                break
        else:
            raise KeyError(
                f"Could not find any of {aliases} in OEWS file columns: "
                f"{list(df.columns)[:20]}..."
            )
    return df.rename(columns=rename)[list(COL_ALIASES.keys())]


def fetch_oews_staffing_patterns(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(OEWS_DIR)
    year = cfg["exposure"]["oews"]["reference_year"]
    yy = str(year)[-2:]

    local_path = _find_local_file(yy)
    if local_path is None:
        local_path = _download(yy)

    if local_path is None:
        fallback = cfg["exposure"]["oews"]["manual_fallback_url"]
        raise FileNotFoundError(
            f"Could not obtain the OEWS industry file automatically.\n"
            f"Please download the May {year} National Industry-Specific "
            f"Occupational Employment and Wage Estimates (4-digit NAICS) "
            f"file manually from {fallback}, unzip if needed, and save it "
            f"as {OEWS_DIR / f'oesm{yy}in4.xlsx'}, then re-run this script."
        )

    print(f"[oews] parsing {local_path} (this file is large, may take a minute)")
    raw = pd.read_excel(local_path)
    df = _standardize_columns(raw)

    # TOT_EMP has non-numeric placeholder codes ("*", "**") for suppressed cells.
    df["tot_emp"] = pd.to_numeric(df["tot_emp"], errors="coerce")
    df = df.dropna(subset=["tot_emp"])
    df["naics"] = df["naics"].astype(str).str.strip()
    df["occ_code"] = df["occ_code"].astype(str).str.strip()

    # Drop the "00-0000 All Occupations" total rows -- we want detailed
    # occupations only so employment shares within a NAICS sum to ~1.
    df = df[df["occ_code"] != "00-0000"]

    out_path = OEWS_DIR / "oews_staffing_patterns.csv"
    df.to_csv(out_path, index=False)
    print(f"[oews]   saved {out_path}  ({len(df):,} NAICS x SOC rows, "
          f"{df['naics'].nunique()} industries)")
    return df


if __name__ == "__main__":
    df = fetch_oews_staffing_patterns()
    print(df.head())
    sys.exit(0)
