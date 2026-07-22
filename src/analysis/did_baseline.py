"""
Baseline binary DiD: tech vs. non-tech postings around a single event date.

This is the "naive hypothesis a casual observer would test" from Sec 1 --
deliberately simple, and deliberately confounded (Sec 5/6 discuss why you
shouldn't trust this number alone: it can't distinguish an LLM effect from
the 2022-23 rate-hike-driven tech correction, since ALL tech sectors get
pooled into one treatment dummy regardless of how exposed they actually are).

Spec:  log(postings_index)_st = beta * (is_tech_s * post_t) + sector FE_s + period FE_t + e_st

Usage:
    python -m src.analysis.did_baseline
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, ensure_dirs
from src.analysis.utils import absorb_fixed_effects, cluster_robust_se


def run_baseline_did(
    panel: pd.DataFrame,
    event_col: str = "post_chatgpt_launch",
    control_groups: tuple[str, ...] = ("stem_control", "non_stem_control"),
) -> dict:
    """control_groups=("stem_control","non_stem_control") pools everything
    non-tech into the control group, matching the naive "tech vs non-tech"
    comparison described in Sec 1. Pass a narrower tuple to tighten the
    control group definition (see robustness.py for the Sec 7 variants)."""
    df = panel[panel["group"].isin(("tech",) + control_groups)].copy()
    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
    df["treat_post"] = df["is_tech"] * df[event_col]

    resid = absorb_fixed_effects(df, cols=["log_postings", "treat_post"],
                                  fe_groups=["display_name", "period"])
    y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
    X = (resid["treat_post"] - resid["treat_post"].mean()).to_numpy().reshape(-1, 1)

    model = sm.OLS(y, X).fit()
    vcov_cl = cluster_robust_se(model.resid, X, df["display_name"])
    se_cl = float(np.sqrt(vcov_cl[0, 0]))
    beta = float(model.params[0])

    return {
        "spec": "baseline_binary_did",
        "event_col": event_col,
        "control_groups": control_groups,
        "n_obs": len(df),
        "n_sectors": df["display_name"].nunique(),
        "beta": beta,
        "se_cluster": se_cl,
        "t_stat": beta / se_cl,
        "beta_pct_naive": np.exp(beta) - 1,  # naive %-change interpretation of a log-linear coef
    }


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(OUTPUT_TABLES_DIR)
    panel_path = PROCESSED_DIR / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `python -m src.data.build_panel` first.")
    panel = pd.read_csv(panel_path, parse_dates=["period"])

    result = run_baseline_did(panel)
    print("=== Baseline binary DiD (tech vs. non-tech, naive spec) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    pd.DataFrame([result]).to_csv(OUTPUT_TABLES_DIR / "baseline_did.csv", index=False)
    sys.exit(0)
