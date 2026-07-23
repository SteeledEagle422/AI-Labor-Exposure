"""
Build the final sector x time analysis panel.

Merges:
  - Indeed Hiring Lab sector-level postings index (outcome)
  - config/sector_naics_map.csv (tech / stem_control / non_stem_control group)
  - data/processed/sector_exposure_scores.csv (continuous exposure scores,
    built by src.data.build_exposure_crosswalk -- run that first)

Produces one row per (sector, time period) with:
  - outcome:            indeed_job_postings_index
  - group flags:        is_tech, is_stem_control, is_non_stem_control
  - continuous exposure: exposure_eloundou_beta, exposure_aioe_direct_naics, ...
  - event-time vars for BOTH candidate event dates (Copilot GA primary,
    ChatGPT launch as the Sec 7 robustness alt), plus the Fed hiking start
    (used to define the "macro shock" window that Sec 5's triple-diff nets out)

Usage:
    python -m src.data.build_panel
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from src.utils.config import load_config, RAW_DIR, PROCESSED_DIR, SECTOR_MAP_PATH, ensure_dirs

INDEED_DIR = RAW_DIR / "indeed"


def _aggregate_to_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Collapse daily Indeed index to weekly/monthly by averaging within
    period. The published index is already a 7-day trailing average, so a
    within-week mean is a mild extra smoothing, not a re-definition."""
    if freq == "daily":
        df["period"] = df["date"]
        return df
    rule = {"weekly": "W-SUN", "monthly": "MS"}[freq]
    df = (
        df.set_index("date")
        .groupby(["display_name", "variable"])["indeed_job_postings_index"]
        .resample(rule)
        .mean()
        .reset_index()
        .rename(columns={"date": "period"})
    )
    # resample() inserts a row for every calendar period in each sector's
    # span, even ones with no underlying observations -- drop those rather
    # than let NaN flow into the regressions downstream.
    return df.dropna(subset=["indeed_job_postings_index"])


def build_panel(cfg: dict | None = None, postings_variable: str = "total postings") -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(PROCESSED_DIR)

    sector_path = INDEED_DIR / "sector_US.csv"
    if not sector_path.exists():
        raise FileNotFoundError(
            f"{sector_path} not found. Run `python -m src.data.fetch_indeed_hiring_lab` first."
        )
    raw = pd.read_csv(sector_path, parse_dates=["date"])
    raw = raw[raw["variable"] == postings_variable].copy()

    freq = cfg["project"]["time_aggregation"]
    panel = _aggregate_to_period(raw, freq)

    # ---- sector group / exposure merge -------------------------------------
    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    panel = panel.merge(
        sector_map[["indeed_display_name", "group", "crosswalk_confidence"]],
        left_on="display_name", right_on="indeed_display_name", how="left",
    ).drop(columns=["indeed_display_name"])

    exposure_path = PROCESSED_DIR / "sector_exposure_scores.csv"
    if exposure_path.exists():
        exposure = pd.read_csv(exposure_path)
        exposure_cols = [c for c in exposure.columns if c.startswith("exposure_")]
        panel = panel.merge(
            exposure[["indeed_display_name"] + exposure_cols + ["naics_match_level", "employment_covered"]],
            left_on="display_name", right_on="indeed_display_name", how="left",
        ).drop(columns=["indeed_display_name"])
    else:
        warnings.warn(
            f"{exposure_path} not found -- continuous-exposure specs will have "
            f"all-NaN exposure columns. Run "
            f"`python -m src.data.build_exposure_crosswalk` first (which itself "
            f"needs OEWS staffing patterns -- see that script's docstring)."
        )
        for c in ["exposure_eloundou_alpha", "exposure_eloundou_beta",
                  "exposure_eloundou_gamma", "exposure_aioe_via_crosswalk",
                  "exposure_aioe_direct_naics"]:
            panel[c] = np.nan

    # ---- group dummies -------------------------------------------------------
    panel["is_tech"] = (panel["group"] == "tech").astype(int)
    panel["is_stem_control"] = (panel["group"] == "stem_control").astype(int)
    panel["is_non_stem_control"] = (panel["group"] == "non_stem_control").astype(int)

    # ---- event-time variables --------------------------------------------
    events = cfg["events"]
    for name in ["copilot_ga", "chatgpt_launch", "fed_hiking_start"]:
        event_date = pd.Timestamp(events[name])
        panel[f"post_{name}"] = (panel["period"] >= event_date).astype(int)
        if freq == "weekly":
            panel[f"weeks_since_{name}"] = (
                (panel["period"] - event_date).dt.days / 7
            ).round().astype(int)
        elif freq == "monthly":
            panel[f"months_since_{name}"] = (
                (panel["period"].dt.to_period("M") - event_date.to_period("M")).apply(lambda x: x.n)
            )

    primary = events["primary_event_date"]
    panel["post"] = panel[f"post_{primary}"]
    panel["event_time"] = panel.get(
        f"weeks_since_{primary}", panel.get(f"months_since_{primary}")
    )

    # "Macro shock window": the 2022-23 tech-wide correction the paper's
    # triple-diff is designed to net out.
    panel["in_layoff_window"] = (
        (panel["period"] >= pd.Timestamp(events["major_layoff_wave_start"]))
        & (panel["period"] <= pd.Timestamp(events["major_layoff_wave_end"]))
    ).astype(int)

    # ---- sample window trim -------------------------------------------------
    sample = cfg["sample"]
    panel = panel[
        (panel["period"] >= pd.Timestamp(sample["start_date"]))
        & (panel["period"] <= pd.Timestamp(sample["end_date"]))
    ].reset_index(drop=True)

    out_path = PROCESSED_DIR / "analysis_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"[panel] saved {out_path}  ({len(panel):,} rows, "
          f"{panel['display_name'].nunique()} sectors, "
          f"{panel['period'].min().date()} to {panel['period'].max().date()})")
    print(f"[panel] group counts (sectors):")
    print(panel.groupby("group")["display_name"].nunique())
    return panel


if __name__ == "__main__":
    df = build_panel()
    sys.exit(0)
