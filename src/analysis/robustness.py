"""
Robustness checks (Sec 7).

Each function returns a small dict/DataFrame so results can be assembled
into the "coefficients move as you add rigor" table style used throughout
this project (see did_baseline.py / continuous_exposure_did.py / triple_diff.py).

Covers:
  - placebo_event_date:       fake pre-period cutoff, expect a null result
  - alt_event_date_comparison: Copilot GA vs. ChatGPT launch as the treatment date
  - alt_control_group_comparison: all-STEM / non-STEM-only / all-occupations controls
  - alt_exposure_measure_comparison: Eloundou et al. vs. AIOE
  - synthetic_control: weighted combination of control sectors matching tech's pre-trend

Usage:
    python -m src.analysis.robustness
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR, ensure_dirs
from src.analysis.did_baseline import run_baseline_did
from src.analysis.continuous_exposure_did import run_continuous_exposure_did
from src.analysis.triple_diff import run_triple_diff


# ---------------------------------------------------------------------------
# Placebo event date
# ---------------------------------------------------------------------------
def placebo_event_date(
    panel: pd.DataFrame,
    placebo_date: str,
    regression_fn=run_baseline_did,
    **regression_kwargs,
) -> dict:
    """Re-run a spec with a fake treatment date drawn from the pre-period.
    If the design is valid, this should come back statistically
    indistinguishable from zero -- a significant placebo result means
    something other than the AI-tool event is driving the main estimate
    (e.g. a pre-existing trend the FE structure isn't fully absorbing)."""
    df = panel.copy()
    placebo_ts = pd.Timestamp(placebo_date)
    if placebo_ts >= panel["period"].min() + pd.Timedelta(weeks=8):
        pass  # fine, plenty of pre-period on both sides
    df["post_placebo"] = (df["period"] >= placebo_ts).astype(int)

    result = regression_fn(df, event_col="post_placebo", **regression_kwargs)
    result["placebo_date"] = placebo_date
    result["check"] = "placebo_event_date"
    return result


# ---------------------------------------------------------------------------
# Alternate event date: Copilot GA (primary) vs. ChatGPT launch
# ---------------------------------------------------------------------------
def alt_event_date_comparison(panel: pd.DataFrame, regression_fn=run_continuous_exposure_did, **kwargs) -> pd.DataFrame:
    rows = []
    for name, col in [("copilot_ga", "post_copilot_ga"), ("chatgpt_launch", "post_chatgpt_launch")]:
        r = regression_fn(panel, event_col=col, **kwargs)
        r["event_date_label"] = name
        r["check"] = "alt_event_date"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alternate control group definitions
# ---------------------------------------------------------------------------
def alt_control_group_comparison(panel: pd.DataFrame, regression_fn=run_baseline_did, **kwargs) -> pd.DataFrame:
    variants = {
        "all_stem_control": ("stem_control",),
        "non_stem_control_only": ("non_stem_control",),
        "all_occupations_pooled": ("stem_control", "non_stem_control"),
    }
    rows = []
    for label, groups in variants.items():
        r = regression_fn(panel, control_groups=groups, **kwargs) if regression_fn is run_baseline_did \
            else regression_fn(panel, groups=("tech",) + groups, **kwargs)
        r["control_group_label"] = label
        r["check"] = "alt_control_group"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alternate exposure measure: Eloundou (via OEWS crosswalk) vs. AIOE (direct NAICS)
# ---------------------------------------------------------------------------
def alt_exposure_measure_comparison(panel: pd.DataFrame, regression_fn=run_triple_diff, **kwargs) -> pd.DataFrame:
    rows = []
    for col in ["exposure_eloundou_beta", "exposure_aioe_via_crosswalk", "exposure_aioe_direct_naics"]:
        try:
            r = regression_fn(panel, exposure_col=col, **kwargs)
        except ValueError as e:
            r = {"spec": regression_fn.__name__, "exposure_col": col, "error": str(e)}
        r["check"] = "alt_exposure_measure"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic control (as an alternative to the binary DiD given a small
# comparison-group N)
# ---------------------------------------------------------------------------
def synthetic_control(
    panel: pd.DataFrame,
    treated_sector: str,
    donor_pool_group: tuple[str, ...] = ("stem_control", "non_stem_control"),
    event_col: str = "post_copilot_ga",
) -> dict:
    """Constrained least squares synthetic control: find nonnegative donor
    weights summing to 1 that best match the treated sector's PRE-PERIOD log
    postings trajectory, then compare post-period gap (treated - synthetic)
    to the pre-period gap (which should be ~0 by construction)."""
    df = panel.copy()
    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))

    treated = df[df["display_name"] == treated_sector].set_index("period")["log_postings"]
    donors_df = df[df["group"].isin(donor_pool_group)]
    donor_names = sorted(donors_df["display_name"].unique())
    donor_wide = donors_df.pivot(index="period", columns="display_name", values="log_postings")

    common_idx = treated.index.intersection(donor_wide.index)
    treated = treated.loc[common_idx]
    donor_wide = donor_wide.loc[common_idx, donor_names].dropna(axis=1)
    donor_names = list(donor_wide.columns)

    pre_mask = df.set_index("period").loc[common_idx, event_col] == 0
    pre_mask = pre_mask[~pre_mask.index.duplicated()]

    y_pre = treated[pre_mask].to_numpy()
    X_pre = donor_wide.loc[pre_mask].to_numpy()

    n_donors = X_pre.shape[1]

    def loss(w):
        return np.sum((y_pre - X_pre @ w) ** 2)

    w0 = np.repeat(1 / n_donors, n_donors)
    bounds = [(0, 1)] * n_donors
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=[constraint])
    weights = res.x

    synthetic_series = donor_wide.to_numpy() @ weights
    gap = treated.to_numpy() - synthetic_series

    post_mask = ~pre_mask
    pre_rmse = float(np.sqrt(np.mean(gap[pre_mask.to_numpy()] ** 2)))
    post_gap_mean = float(np.mean(gap[post_mask.to_numpy()]))

    weight_table = pd.Series(weights, index=donor_names).sort_values(ascending=False)

    return {
        "treated_sector": treated_sector,
        "n_donors": n_donors,
        "pre_period_rmse": pre_rmse,
        "post_period_avg_gap": post_gap_mean,  # analogue of the DiD beta
        "top_donor_weights": weight_table[weight_table > 0.01].to_dict(),
        "gap_series": pd.Series(gap, index=common_idx),
    }


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR)
    panel_path = PROCESSED_DIR / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `python -m src.data.build_panel` first.")
    panel = pd.read_csv(panel_path, parse_dates=["period"])

    print("=== Placebo event date (2021-09-01, well before any real treatment) ===")
    placebo = placebo_event_date(panel, "2021-09-01")
    print(placebo)

    print("\n=== Alternate event date: Copilot GA vs. ChatGPT launch ===")
    try:
        alt_dates = alt_event_date_comparison(panel)
        print(alt_dates[["event_date_label", "beta", "se_cluster", "n_sectors"]])
    except ValueError as e:
        print(f"(skipped -- {e})")

    print("\n=== Alternate control group definitions ===")
    alt_ctrl = alt_control_group_comparison(panel)
    print(alt_ctrl[["control_group_label", "beta", "se_cluster", "n_sectors"]])

    print("\n=== Alternate exposure measure ===")
    try:
        alt_exp = alt_exposure_measure_comparison(panel)
        print(alt_exp[["exposure_col", "beta", "se_cluster"]] if "beta" in alt_exp else alt_exp)
    except Exception as e:
        print(f"(skipped -- {e})")

    print("\n=== Synthetic control for Software Development ===")
    try:
        sc = synthetic_control(panel, treated_sector="Software Development")
        print({k: v for k, v in sc.items() if k != "gap_series"})
    except Exception as e:
        print(f"(skipped -- {e})")

    placebo_df = pd.DataFrame([placebo])
    placebo_df.to_csv(OUTPUT_TABLES_DIR / "robustness_placebo_event_date.csv", index=False)
    alt_ctrl.to_csv(OUTPUT_TABLES_DIR / "robustness_alt_control_groups.csv", index=False)
    sys.exit(0)
