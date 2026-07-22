"""
Triple-diff: this is the paper's main identification claim (Sec 5).

Adds a group x period fixed effect on top of the continuous-exposure DiD.
Group here means {tech, stem_control, non_stem_control} -- so a group x
period effect absorbs whatever is common to ALL tech sectors in a given
week, which is exactly where the 2022-23 rate-hike-driven tech correction
lives (it hit tech broadly, not just the most LLM-exposed corners of tech).

What's left to identify beta: only the within-group-period variation in
exposure. Concretely, in any given week, "Software Development" (high
exposure) is being compared to "Data & Analytics" (lower exposure) -- both
tech, both hit by the same macro shock that week, which cancels out. If
Software Development's postings fall MORE than Data & Analytics' in the same
week, that's attributed to the exposure gap between them, not to "tech had a
bad week."

Spec:  log(postings_index)_st = beta * (exposure_s * post_t)
                                 + sector FE_s + (group x period) FE_gt + e_st

Usage:
    python -m src.analysis.triple_diff
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, ensure_dirs
from src.analysis.utils import absorb_fixed_effects, cluster_robust_se, build_group_period


def run_triple_diff(
    panel: pd.DataFrame,
    event_col: str = "post_copilot_ga",
    exposure_col: str = "exposure_eloundou_beta",
    groups: tuple[str, ...] = ("tech", "stem_control", "non_stem_control"),
) -> dict:
    df = panel[panel["group"].isin(groups)].copy()
    df = df.dropna(subset=[exposure_col])
    if df.empty:
        raise ValueError(
            f"No rows with non-null {exposure_col}. Run "
            f"`python -m src.data.build_exposure_crosswalk` (needs OEWS staffing "
            f"patterns first) then rebuild the panel."
        )

    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
    df["exposure_z"] = (df[exposure_col] - df[exposure_col].mean()) / df[exposure_col].std()
    df["exposure_x_post"] = df["exposure_z"] * df[event_col]
    df["group_period"] = build_group_period(df, "group", "period")

    resid = absorb_fixed_effects(df, cols=["log_postings", "exposure_x_post"],
                                  fe_groups=["display_name", "group_period"])
    y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
    X = (resid["exposure_x_post"] - resid["exposure_x_post"].mean()).to_numpy().reshape(-1, 1)

    model = sm.OLS(y, X).fit()
    vcov_cl = cluster_robust_se(model.resid, X, df["display_name"])
    se_cl = float(np.sqrt(vcov_cl[0, 0]))
    beta = float(model.params[0])

    return {
        "spec": "triple_diff",
        "event_col": event_col,
        "exposure_col": exposure_col,
        "groups": groups,
        "n_obs": len(df),
        "n_sectors": df["display_name"].nunique(),
        "n_group_period_cells": df["group_period"].nunique(),
        "beta": beta,
        "se_cluster": se_cl,
        "t_stat": beta / se_cl,
    }


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(OUTPUT_TABLES_DIR)
    panel_path = PROCESSED_DIR / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `python -m src.data.build_panel` first.")
    panel = pd.read_csv(panel_path, parse_dates=["period"])

    result = run_triple_diff(panel)
    print("=== Triple-diff (sector FE + group x period FE) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    pd.DataFrame([result]).to_csv(OUTPUT_TABLES_DIR / "triple_diff.csv", index=False)
    sys.exit(0)
