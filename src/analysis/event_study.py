"""
Event-study version of the binary tech DiD: replaces the single post-dummy
with a full set of lead/lag bins around the event date. Pre-period
coefficients are the parallel-trends test Sec 5 calls for -- they should be
flat and statistically indistinguishable from zero if the design is valid.
A violation looks like a pre-trend: tech postings already diverging from
control *before* Copilot GA, which would mean the "effect" partly reflects
something else (e.g. anticipation, or a slower-moving confound not fully
captured by sector/period FE).

Spec: log(postings_index)_st = sum_k beta_k * (is_tech_s * bin_k(t)) + sector FE_s + period FE_t + e_st
      bin just before the event (k = -1 bin) is the omitted reference category.

Usage:
    python -m src.analysis.event_study
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR, ensure_dirs
from src.analysis.utils import absorb_fixed_effects, cluster_robust_se, event_time_bins


def run_event_study(
    panel: pd.DataFrame,
    event_time_col: str = "weeks_since_copilot_ga",
    bin_width: int = 4,
    window: int = 24,
    control_groups: tuple[str, ...] = ("stem_control", "non_stem_control"),
) -> pd.DataFrame:
    df = panel[panel["group"].isin(("tech",) + control_groups)].copy()
    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
    df["bin"] = event_time_bins(df[event_time_col], bin_width=bin_width, window=window)

    bin_dummies = pd.get_dummies(df["bin"], prefix="bin", drop_first=False)
    ref_col = "bin_ref"
    reg_cols = [c for c in bin_dummies.columns if c != ref_col]

    interacted = bin_dummies[reg_cols].multiply(df["is_tech"], axis=0)
    interacted.columns = [f"tech_x_{c}" for c in interacted.columns]

    reg_df = pd.concat([df[["display_name", "period", "log_postings"]], interacted], axis=1)
    coef_cols = list(interacted.columns)

    resid = absorb_fixed_effects(reg_df, cols=["log_postings"] + coef_cols,
                                  fe_groups=["display_name", "period"])
    y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
    X = resid[coef_cols].to_numpy() - resid[coef_cols].mean().to_numpy()

    model = sm.OLS(y, X).fit()
    vcov_cl = cluster_robust_se(model.resid, X, df["display_name"])
    se_cl = np.sqrt(np.diag(vcov_cl))

    def bin_sort_key(name):
        raw = name.replace("tech_x_bin_", "")
        if raw == "far_pre":
            return -9999
        if raw == "far_post":
            return 9999
        sign = -1 if raw.startswith("m") else 1
        return sign * int(raw[1:])

    out = pd.DataFrame({
        "bin": coef_cols,
        "beta": model.params,
        "se_cluster": se_cl,
    })
    out["sort_key"] = out["bin"].apply(bin_sort_key)
    out = out.sort_values("sort_key").reset_index(drop=True)
    out["ci_lower"] = out["beta"] - 1.96 * out["se_cluster"]
    out["ci_upper"] = out["beta"] + 1.96 * out["se_cluster"]

    # add back the omitted reference bin at coefficient 0 for plotting
    ref_row = pd.DataFrame([{"bin": "tech_x_bin_ref", "beta": 0.0, "se_cluster": 0.0,
                              "sort_key": -bin_width, "ci_lower": 0.0, "ci_upper": 0.0}])
    out = pd.concat([out, ref_row]).sort_values("sort_key").reset_index(drop=True)
    return out


def plot_event_study(coef_df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(coef_df))
    ax.errorbar(x, coef_df["beta"], yerr=1.96 * coef_df["se_cluster"],
                fmt="o", capsize=3, color="#1f4e79")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(coef_df[coef_df["sort_key"] < 0].index.max() + 0.5,
               color="red", linestyle="--", linewidth=0.8, label="event date")
    ax.set_xticks(list(x))
    ax.set_xticklabels(coef_df["bin"].str.replace("tech_x_bin_", ""), rotation=45, ha="right")
    ax.set_ylabel("Coefficient on tech x event-time bin (log points)")
    ax.set_title("Event study: tech vs. non-tech postings around AI-tool event date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR)
    panel_path = PROCESSED_DIR / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `python -m src.data.build_panel` first.")
    panel = pd.read_csv(panel_path, parse_dates=["period"])

    coef_df = run_event_study(panel)
    print(coef_df.to_string(index=False))

    out_csv = OUTPUT_TABLES_DIR / "event_study_coefficients.csv"
    coef_df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    out_fig = OUTPUT_FIGURES_DIR / "event_study.png"
    plot_event_study(coef_df, out_fig)
    print(f"saved {out_fig}")
    sys.exit(0)
