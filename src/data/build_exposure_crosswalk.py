"""
Build sector-level (Indeed "display_name") LLM/AI exposure scores from
occupation-level (SOC) exposure scores, using BLS OEWS staffing patterns as
employment weights.

This is the crosswalk logic flagged explicitly in Sec 4 (Data) and Sec 8
(Limitations) of the paper: Indeed's sectors are occupation-title based,
Eloundou/AIOE scores are SOC-based, BLS CES is NAICS-based. Three different
classification systems. The bridge is:

    Indeed sector --(config/sector_naics_map.csv, hand-curated)--> NAICS
    NAICS --(OEWS staffing pattern: SOC employment within that NAICS)--> SOC mix
    SOC mix x SOC-level exposure score --(employment-weighted average)--> sector exposure

For each Indeed sector mapped to one or more NAICS codes:
    sector_exposure = sum_soc( emp_share_soc_within_mapped_naics * exposure_soc )

NAICS matching is done with a fallback hierarchy because OEWS does not
publish every industry at the same digit-level of NAICS detail (some only
go to 3-digit): exact code match -> 4-digit prefix match -> 3-digit prefix
match. Coverage diagnostics are printed so a mismatch is visible immediately
rather than silently producing NaN/garbage exposure scores.

Depends on having already run (in order):
    python -m src.data.fetch_eloundou_exposure
    python -m src.data.fetch_aioe_exposure
    python -m src.data.fetch_oews_staffing_patterns   <- needs local/manual OEWS file
Usage:
    python -m src.data.build_exposure_crosswalk
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from src.utils.config import load_config, RAW_DIR, PROCESSED_DIR, SECTOR_MAP_PATH, ensure_dirs

EXPOSURE_DIR = RAW_DIR / "exposure"
OEWS_DIR = RAW_DIR / "oews"


def _load_inputs():
    eloundou = pd.read_csv(EXPOSURE_DIR / "eloundou_occ_level.csv")
    aioe_occ = pd.read_csv(EXPOSURE_DIR / "aioe_occupation_level.csv")
    aioe_ind = pd.read_csv(EXPOSURE_DIR / "aioe_industry_level.csv")
    oews = pd.read_csv(OEWS_DIR / "oews_staffing_patterns.csv", dtype={"naics": str, "occ_code": str})
    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    return eloundou, aioe_occ, aioe_ind, oews, sector_map


def _match_naics_employment(oews: pd.DataFrame, target_naics: str) -> tuple[pd.DataFrame, str]:
    """Return (OEWS rows matching a target NAICS code, match_level) using
    exact -> 4-digit prefix -> 3-digit prefix fallback.

    match_level is surfaced so a low-confidence match is visible rather than
    silently blended in: "prefix3_COARSE" means we fell all the way back to
    a subsector-wide match (e.g. all of NAICS 541 "Professional, Scientific,
    and Technical Services"), which is much less defensible than an exact
    4-digit hit and should be reported as such in any write-up.
    """
    target_naics = str(target_naics).strip()
    exact = oews[oews["naics"] == target_naics]
    if len(exact) > 0:
        return exact, "exact"
    prefix4 = oews[oews["naics"].str.startswith(target_naics[:4])]
    if len(prefix4) > 0:
        return prefix4, "prefix4"
    prefix3 = oews[oews["naics"].str.startswith(target_naics[:3])]
    if len(prefix3) > 0:
        return prefix3, "prefix3_COARSE"
    return prefix3, "no_match"  # empty -- caller handles that


_MATCH_RANK = {"exact": 0, "prefix4": 1, "prefix3_COARSE": 2, "no_match": 3}


def _sector_employment_weights(oews: pd.DataFrame, naics_codes: list[str]) -> tuple[pd.DataFrame, str]:
    """Pool employment across all NAICS codes mapped to one Indeed sector,
    then collapse to SOC-level employment shares. Also returns the worst
    (coarsest) match level used across the sector's mapped NAICS codes."""
    results = [_match_naics_employment(oews, n) for n in naics_codes]
    frames = [f for f, _ in results if len(f) > 0]
    levels = [lvl for _, lvl in results]
    worst_level = max(levels, key=lambda l: _MATCH_RANK[l]) if levels else "no_match"

    if not frames:
        return pd.DataFrame(columns=["soc_code", "tot_emp", "emp_share"]), worst_level

    # drop_duplicates guards against double-counting: if two of the sector's
    # mapped NAICS codes both fall back to the same coarse prefix match (see
    # _match_naics_employment), they'd otherwise return the identical set of
    # OEWS rows and sum their employment twice.
    pooled = pd.concat(frames, ignore_index=True).drop_duplicates()
    # occ_code in OEWS is already 6-char SOC ("15-1252"); Eloundou/AIOE use the
    # same 6-char SOC after our own fetch scripts normalized it.
    by_soc = pooled.groupby("occ_code", as_index=False)["tot_emp"].sum()
    by_soc = by_soc.rename(columns={"occ_code": "soc_code"})
    total_emp = by_soc["tot_emp"].sum()
    by_soc["emp_share"] = by_soc["tot_emp"] / total_emp if total_emp > 0 else np.nan
    return by_soc, worst_level


def build_sector_exposure_scores(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(PROCESSED_DIR)
    eloundou, aioe_occ, aioe_ind, oews, sector_map = _load_inputs()

    eloundou_cols = ["soc_code", "dv_rating_alpha", "dv_rating_beta", "dv_rating_gamma"]
    eloundou_by_soc = eloundou.groupby("soc_code", as_index=False)[eloundou_cols[1:]].mean()

    aioe_by_soc = aioe_occ.groupby("soc_code", as_index=False)["aioe"].mean()

    rows = []
    for _, r in sector_map.iterrows():
        naics_codes = [c.strip() for c in str(r["naics_codes"]).split(";")]
        weights, worst_match_level = _sector_employment_weights(oews, naics_codes)

        merged = weights.merge(eloundou_by_soc, on="soc_code", how="left")
        merged = merged.merge(aioe_by_soc, on="soc_code", how="left")

        n_matched_soc = merged["dv_rating_beta"].notna().sum()
        emp_covered = weights["tot_emp"].sum() if len(weights) else 0.0

        def wavg(col):
            valid = merged.dropna(subset=[col])
            if valid.empty or valid["emp_share"].sum() == 0:
                return np.nan
            return np.average(valid[col], weights=valid["emp_share"])

        # Direct NAICS-level AIIE (Appendix B) -- employment-weighted across
        # the mapped NAICS codes themselves (no SOC crosswalk needed here).
        naics_emp = {}
        for n in naics_codes:
            sub, _ = _match_naics_employment(oews, n)
            naics_emp[n] = sub["tot_emp"].sum() if len(sub) else 0.0
        total_naics_emp = sum(naics_emp.values())
        aiie_direct = np.nan
        if total_naics_emp > 0:
            vals, wts = [], []
            for n in naics_codes:
                row = aioe_ind[aioe_ind["naics"].astype(str).str.strip() == n]
                if len(row) and naics_emp[n] > 0:
                    vals.append(row["aiie"].iloc[0])
                    wts.append(naics_emp[n])
            if vals:
                aiie_direct = np.average(vals, weights=wts)

        rows.append({
            "indeed_display_name": r["indeed_display_name"],
            "group": r["group"],
            "crosswalk_confidence": r["crosswalk_confidence"],
            "naics_codes": r["naics_codes"],
            "naics_match_level": worst_match_level,
            "n_soc_matched": n_matched_soc,
            "employment_covered": emp_covered,
            "exposure_eloundou_alpha": wavg("dv_rating_alpha"),
            "exposure_eloundou_beta": wavg("dv_rating_beta"),
            "exposure_eloundou_gamma": wavg("dv_rating_gamma"),
            "exposure_aioe_via_crosswalk": wavg("aioe"),
            "exposure_aioe_direct_naics": aiie_direct,
        })

    out = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "sector_exposure_scores.csv"
    out.to_csv(out_path, index=False)

    n_zero_coverage = (out["employment_covered"] == 0).sum()
    n_coarse = (out["naics_match_level"] == "prefix3_COARSE").sum()
    print(f"[crosswalk] saved {out_path}  ({len(out)} sectors)")
    print(f"[crosswalk] {n_zero_coverage} / {len(out)} sectors had ZERO OEWS "
          f"employment matched -- check config/sector_naics_map.csv NAICS "
          f"codes and OEWS file coverage for those.")
    print(f"[crosswalk] {n_coarse} / {len(out)} sectors only matched at the "
          f"coarse 3-digit NAICS subsector level -- treat their exposure "
          f"scores as lower-confidence and say so in the write-up.")
    return out


if __name__ == "__main__":
    df = build_sector_exposure_scores()
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["indeed_display_name", "group", "employment_covered",
                   "exposure_eloundou_beta", "exposure_aioe_direct_naics"]])
    sys.exit(0)
