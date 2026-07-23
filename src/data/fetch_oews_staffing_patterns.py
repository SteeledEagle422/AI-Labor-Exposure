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
  bls.gov rate-limits automated downloads and will start returning
  "403 Access Denied" after a few requests, even with a well-behaved
  User-Agent. When that happens, fetch the file once from a browser:

  1. Go to https://www.bls.gov/oes/tables.htm
  2. Find "May {YEAR} National Industry-Specific Occupational Employment
     and Wage Estimates" and download the bulk file for all industries at
     4-digit NAICS. As of the May 2024 vintage the direct link is
     https://www.bls.gov/oes/special-requests/oesm24in4.zip (~32 MB).
  3. Save it as EITHER:
       data/raw/oews/oesm{YY}in4.zip     (preferred -- kept for re-parsing)
       data/raw/oews/oesm{YY}in4.xlsx    (if you unzip it yourself)
     If you unzip by hand, take the large estimates workbook, NOT the small
     "field_descriptions" readme that ships beside it.
  4. Re-run this script -- it finds the local file and skips the download
     entirely, so no further requests hit bls.gov.

Usage:
    python -m src.data.fetch_oews_staffing_patterns
"""
from __future__ import annotations

import io
import re
import sys
import zipfile

import requests
import pandas as pd

from src.utils.config import load_config, RAW_DIR, ensure_dirs

OEWS_DIR = RAW_DIR / "oews"

# BLS bulk-file URL patterns for the 4-digit NAICS industry file.
# BLS renamed the directory "special.requests" -> "special-requests" (dot to
# hyphen); the old dotted path now silently 301s to the OEWS landing page for
# recent vintages instead of 404ing, so the hyphenated form is tried FIRST and
# the response is content-type checked below rather than trusted on status
# code alone. The dotted form is kept as a fallback for older vintages.
CANDIDATE_URL_PATTERNS = [
    "https://www.bls.gov/oes/special-requests/oesm{yy}in4.zip",
    "https://www.bls.gov/oes/special-requests/oesm{yy}in4.xlsx",
    "https://www.bls.gov/oes/special.requests/oesm{yy}in4.zip",
    "https://www.bls.gov/oes/special.requests/oesm{yy}in4.xlsx",
]

# bls.gov returns 403 to the default python-requests User-Agent. These are
# public-domain bulk files published for download; BLS just wants automated
# clients to identify themselves, so send a descriptive UA.
REQUEST_HEADERS = {
    "User-Agent": "ai-labor-exposure/1.0 (research data pipeline; python-requests)"
}

# Column names BLS has used across vintages for the fields we need.
# o_group is optional (older vintages lack it) -- see _drop_nested_occupations.
COL_ALIASES = {
    "naics": ["NAICS", "naics"],
    "naics_title": ["NAICS_TITLE", "naics_title"],
    "occ_code": ["OCC_CODE", "occ_code"],
    "occ_title": ["OCC_TITLE", "occ_title"],
    "tot_emp": ["TOT_EMP", "tot_emp"],
}
OPTIONAL_COL_ALIASES = {
    "o_group": ["O_GROUP", "o_group"],
}

# The estimates workbook we want is the national 4-digit NAICS file
# ("nat4d_M2024_dl.xlsx"). The archive also ships nat3d / nat5d_6d / natsector
# cuts at other NAICS granularities, and "_owner" variants that split the same
# employment across ownership classes -- summing one of those would double
# count. Match the 4-digit non-owner file explicitly rather than guessing.
_PREFERRED_NAME_RE = re.compile(r"nat4d_M\d{4}_dl\.xlsx?$", re.IGNORECASE)


def _find_local_file(yy: str) -> "pd.io.common.FilePath | None":
    for ext in ("xlsx", "xls"):
        p = OEWS_DIR / f"oesm{yy}in4.{ext}"
        if p.exists():
            return p
    # Also accept the estimates workbook under its original BLS name, so a
    # hand-extracted archive can just be dropped into data/raw/oews/.
    named = sorted(p for p in OEWS_DIR.glob("*.xls*") if _PREFERRED_NAME_RE.search(p.name))
    if named:
        print(f"[oews] using local {named[0].name}")
        return named[0]
    # Fall back to a previously downloaded zip (either cached by _download or
    # dropped in by hand per the MANUAL STEP) so a re-parse never needs another
    # request -- BLS rate-limits repeat downloads and starts returning 403s.
    zip_path = OEWS_DIR / f"oesm{yy}in4.zip"
    if zip_path.exists():
        print(f"[oews] found local {zip_path}, extracting instead of downloading")
        return _extract_data_xlsx(zip_path, yy)
    return None


# Metadata/readme workbooks that ship alongside the actual estimates and must
# not be mistaken for it.
_METADATA_NAME_HINTS = ("field_description", "field descriptions", "readme", "notes")


def _extract_data_xlsx(zip_path, yy: str) -> "pd.io.common.FilePath | None":
    """Pull the actual estimates workbook out of a BLS OEWS zip.

    The archive ships eight workbooks: the national 4-digit NAICS estimates we
    want, the same data cut at other NAICS granularities (nat3d, nat5d_6d,
    natsector), "_owner" variants that split employment across ownership
    classes, and a small "field_descriptions" readme. Taking namelist()[0]
    grabbed the readme, which then failed column standardization with a
    confusing KeyError.

    Match the 4-digit non-owner file by name first. Falling back to "largest
    workbook" would happen to work for the May 2024 vintage but is not safe in
    general -- picking nat3d or an _owner cut would silently feed the crosswalk
    the wrong NAICS granularity or double-counted employment.
    """
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [
            i for i in zf.infolist()
            if i.filename.lower().endswith((".xlsx", ".xls"))
            and not any(h in i.filename.lower() for h in _METADATA_NAME_HINTS)
        ]
        if not candidates:
            return None
        preferred = [i for i in candidates if _PREFERRED_NAME_RE.search(i.filename)]
        if preferred:
            best = preferred[0]
        else:
            best = max(candidates, key=lambda i: i.file_size)
            print(f"[oews]   WARNING: no nat4d file in archive, falling back to "
                  f"largest workbook ({best.filename}) -- verify this is the "
                  f"4-digit NAICS national estimates file")
        out_path = OEWS_DIR / f"oesm{yy}in4.xlsx"
        out_path.write_bytes(zf.read(best.filename))
        print(f"[oews]   extracted {best.filename} "
              f"({best.file_size / 1e6:.1f} MB) -> {out_path}")
        return out_path


def _download(yy: str) -> "pd.io.common.FilePath | None":
    for pattern in CANDIDATE_URL_PATTERNS:
        url = pattern.format(yy=yy)
        print(f"[oews] attempting download <- {url}")
        try:
            resp = requests.get(url, timeout=180, headers=REQUEST_HEADERS)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[oews]   failed ({e})")
            continue

        # A retired URL 301s to the OEWS landing page and still returns 200,
        # so status alone doesn't mean we got the data file -- check we were
        # actually handed a binary payload and not an HTML page.
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            print(f"[oews]   got an HTML page, not the data file "
                  f"(redirected to {resp.url}) -- trying next candidate")
            continue

        if url.endswith(".zip"):
            # Keep the raw zip so a re-parse never needs a second download --
            # BLS rate-limits repeated hits and will start returning 403s.
            zip_path = OEWS_DIR / f"oesm{yy}in4.zip"
            zip_path.write_bytes(resp.content)
            print(f"[oews]   saved {zip_path}")

            out_path = _extract_data_xlsx(zip_path, yy)
            if out_path is None:
                print("[oews]   zip contained no usable xlsx/xls file, skipping")
                continue
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
    keep = list(COL_ALIASES.keys())
    for target, aliases in OPTIONAL_COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename[a] = target
                keep.append(target)
                break
    return df.rename(columns=rename)[keep]


def _drop_nested_occupations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only detailed (6-digit SOC) occupation rows.

    OEWS reports each NAICS industry at five nested occupation levels --
    total ("00-0000 All Occupations"), major, minor, broad, and detailed --
    and the coarser rows are sums of the finer ones. Dropping only the
    "00-0000" total still leaves major/minor/broad in, so summing tot_emp
    counts every worker about four times over (613M vs. the true ~153M for
    the May 2024 vintage).

    That inflation happens to cancel out of the final exposure scores today,
    because np.average renormalizes the weights and no aggregate SOC code
    matches an Eloundou/AIOE score anyway -- but it makes the
    `employment_covered` diagnostic meaningless and would start silently
    biasing the weights the moment a coarse code did match. Filter properly.
    """
    if "o_group" in df.columns:
        detailed = df[df["o_group"].astype(str).str.strip().str.lower() == "detailed"]
        if len(detailed) > 0:
            return detailed.drop(columns=["o_group"])
        print("[oews]   WARNING: o_group column present but no 'detailed' rows "
              "found -- falling back to dropping the 00-0000 total only")
        df = df.drop(columns=["o_group"])
    else:
        print("[oews]   note: no O_GROUP column in this vintage -- falling back "
              "to dropping the 00-0000 total only (aggregate rows may remain)")
    return df[df["occ_code"] != "00-0000"]


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

    # Keep detailed occupations only, so employment shares within a NAICS sum
    # to ~1 instead of ~4 (see _drop_nested_occupations).
    n_before = len(df)
    df = _drop_nested_occupations(df)
    print(f"[oews]   kept {len(df):,} detailed-occupation rows of {n_before:,} "
          f"(dropped nested total/major/minor/broad aggregates)")

    out_path = OEWS_DIR / "oews_staffing_patterns.csv"
    df.to_csv(out_path, index=False)
    print(f"[oews]   saved {out_path}  ({len(df):,} NAICS x SOC rows, "
          f"{df['naics'].nunique()} industries, "
          f"{df['tot_emp'].sum():,.0f} total employment)")
    return df


if __name__ == "__main__":
    df = fetch_oews_staffing_patterns()
    print(df.head())
    sys.exit(0)
