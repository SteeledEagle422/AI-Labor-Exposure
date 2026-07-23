"""
Continuous-exposure DiD: replaces the binary tech/non-tech dummy with each
sector's actual LLM exposure score, so a "Software Development" sector
(high exposure) and a "Data & Analytics" sector (lower exposure) aren't
forced into the same treatment bucket.

Still uses plain sector FE + period FE only -- i.e. still vulnerable to the
2022-23 tech-wide correction, since that shock isn't restricted to hit
exposure-weighted sectors differently within the tech group. See
triple_diff.py for the version that nets that out. Comparing the two is the
main "skill display" table in Sec 6 of the paper.

Spec:  log(postings_index)_st = beta * (exposure_s * post_t) + sector FE_s + period FE_t + e_st

Usage:
    python -m src.analysis.continuous_exposure_did
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, ensure_dirs, exposure_primary_column
from src.analysis.utils import absorb_fixed_effects, cluster_robust_se


def run_continuous_exposure_did(
    panel: pd.DataFrame,
    event_col: str = "post_copilot_ga",
    exposure_col: str | None = None,
    groups: tuple[str, ...] = ("tech", "stem_control", "non_stem_control"),
) -> dict:
    exposure_col = exposure_col or exposure_primary_column()
    df = panel[panel["group"].isin(groups)].copy()
    df = df.dropna(subset=[exposure_col])
    if df.empty:
        raise ValueError(
            f"No rows with non-null {exposure_col}. Run "
            f"`python -m src.data.build_exposure_crosswalk` (needs OEWS staffing "
            f"patterns first) then rebuild the panel with "
            f"`python -m src.data.build_panel`."
        )

    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
    # standardize exposure to mean 0 / sd 1 so beta reads as "effect of a
    # 1-SD increase in exposure", not tied to the arbitrary [0,1] scale
    df["exposure_z"] = (df[exposure_col] - df[exposure_col].mean()) / df[exposure_col].std()
    df["exposure_x_post"] = df["exposure_z"] * df[event_col]

    resid = absorb_fixed_effects(df, cols=["log_postings", "exposure_x_post"],
                                  fe_groups=["display_name", "period"])
    y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
    X = (resid["exposure_x_post"] - resid["exposure_x_post"].mean()).to_numpy().reshape(-1, 1)

    model = sm.OLS(y, X).fit()
    vcov_cl = cluster_robust_se(model.resid, X, df["display_name"])
    se_cl = float(np.sqrt(vcov_cl[0, 0]))
    beta = float(model.params[0])

    return {
        "spec": "continuous_exposure_did",
        "event_col": event_col,
        "exposure_col": exposure_col,
        "groups": groups,
        "n_obs": len(df),
        "n_sectors": df["display_name"].nunique(),
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

    result = run_continuous_exposure_did(panel)
    print("=== Continuous-exposure DiD (sector FE + period FE only) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    pd.DataFrame([result]).to_csv(OUTPUT_TABLES_DIR / "continuous_exposure_did.csv", index=False)
    sys.exit(0)
