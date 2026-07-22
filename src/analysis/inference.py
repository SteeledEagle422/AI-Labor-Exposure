"""
Inference for the few-clusters problem (Sec 5's inference plan).

With ~13-20 sector clusters, standard cluster-robust SEs (as reported by
did_baseline.py / continuous_exposure_did.py / triple_diff.py) are
unreliable -- the asymptotics assume "many clusters." Two corrections:

1. Wild cluster bootstrap (Cameron, Gelbach & Miller 2008), via the
   `wildboottest` package. This is the standard fix in applied micro and
   should be reported as the primary inference in the results tables, with
   the plain cluster-robust SE shown for reference/comparability only.

2. Randomization inference (Fisher-style permutation test): shuffle which
   sectors are "treated" (either the binary tech flag or the continuous
   exposure score, depending on spec) across sector identities many times,
   recompute the same statistic under each placebo assignment, and see
   where the real estimate falls in that null distribution. Makes no
   asymptotic-normality assumption at all -- useful as a second, more
   primitive check when even the wild bootstrap feels shaky given so few
   sectors.

Usage (see bottom of file for a worked example against the real panel):
    python -m src.analysis.inference
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import load_config, PROCESSED_DIR, OUTPUT_TABLES_DIR, ensure_dirs
from src.analysis.utils import absorb_fixed_effects, build_group_period


def wild_cluster_bootstrap_pvalue(
    y: np.ndarray,
    x: np.ndarray,
    cluster: pd.Series,
    B: int = 9999,
    weights_type: str = "rademacher",
    seed: int = 12345,
) -> dict:
    """y, x: 1-D arrays of the ALREADY FE-ABSORBED (residualized) outcome and
    single regressor of interest (see did_baseline.py etc. for how these are
    constructed). Wraps wildboottest's high-level API."""
    from wildboottest.wildboottest import wildboottest

    # wildboottest's numba-jitted backend requires a numeric cluster array --
    # factorize string sector names (e.g. "Software Development") into codes.
    cluster_codes, _ = pd.factorize(cluster)
    cluster_codes = cluster_codes.astype(np.int64)

    X = pd.DataFrame({"beta_of_interest": x})
    model = sm.OLS(y, X)
    result = wildboottest(
        model, param="beta_of_interest", cluster=cluster_codes,
        B=B, weights_type=weights_type, seed=seed, show=False,
    )
    return {
        "wild_boot_pvalue": float(result["p-value"].iloc[0]),
        "wild_boot_tstat": float(result["statistic"].iloc[0]),
        "n_clusters": cluster.nunique(),
        "B": B,
        "weights_type": weights_type,
    }


def randomization_inference(
    panel: pd.DataFrame,
    regression_fn,
    permute_col: str,
    entity_col: str = "display_name",
    n_reps: int = 5000,
    seed: int = 0,
) -> dict:
    """Generic Fisher-style permutation test.

    regression_fn: callable(panel_df) -> float, returning the point estimate
        of interest (e.g. a thin wrapper around run_baseline_did that
        returns result["beta"]). Must be cheap enough to call n_reps times --
        for the full triple-diff spec this can take a while; consider
        reducing n_reps or n periods for a first pass.
    permute_col: the sector-level column to shuffle across sector identities
        each replication (e.g. "is_tech" for the binary specs, or an
        exposure column for the continuous specs). Shuffling is done at the
        sector level (not row level) so the panel's time-series structure
        within a sector is preserved -- only *which* sector got which
        treatment status is randomized, which is the right unit of
        randomization here since sectors are the clusters.
    """
    rng = np.random.default_rng(seed)
    sector_vals = panel[[entity_col, permute_col]].drop_duplicates().set_index(entity_col)[permute_col]

    observed_beta = regression_fn(panel)

    placebo_betas = []
    sectors = sector_vals.index.to_numpy()
    values = sector_vals.to_numpy()
    for i in range(n_reps):
        shuffled = values.copy()
        rng.shuffle(shuffled)
        remap = dict(zip(sectors, shuffled))

        placebo_panel = panel.copy()
        placebo_panel[permute_col] = placebo_panel[entity_col].map(remap)
        try:
            placebo_betas.append(regression_fn(placebo_panel))
        except Exception:
            continue  # a degenerate permutation (e.g. all-one-group) can fail; skip it

    placebo_betas = np.array(placebo_betas)
    p_value = float(np.mean(np.abs(placebo_betas) >= np.abs(observed_beta)))

    return {
        "observed_beta": observed_beta,
        "n_valid_reps": len(placebo_betas),
        "placebo_beta_mean": float(placebo_betas.mean()) if len(placebo_betas) else np.nan,
        "placebo_beta_sd": float(placebo_betas.std()) if len(placebo_betas) else np.nan,
        "randomization_pvalue": p_value,
    }


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(OUTPUT_TABLES_DIR)
    panel_path = PROCESSED_DIR / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `python -m src.data.build_panel` first.")
    panel = pd.read_csv(panel_path, parse_dates=["period"])

    # --- worked example: wild bootstrap for the baseline binary DiD ---
    from src.analysis.did_baseline import run_baseline_did
    df = panel[panel["group"].isin(("tech", "stem_control", "non_stem_control"))].copy()
    df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
    df["treat_post"] = df["is_tech"] * df["post_chatgpt_launch"]
    resid = absorb_fixed_effects(df, cols=["log_postings", "treat_post"],
                                  fe_groups=["display_name", "period"])
    y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
    x = (resid["treat_post"] - resid["treat_post"].mean()).to_numpy()

    reps = cfg["inference"]["wild_bootstrap_reps"]
    print(f"Running wild cluster bootstrap (B={reps}) on baseline binary DiD ...")
    wb_result = wild_cluster_bootstrap_pvalue(y, x, df["display_name"], B=reps)
    print(wb_result)

    # --- worked example: randomization inference for the same spec ---
    def _beta_fn(p):
        return run_baseline_did(p)["beta"]

    ri_reps = min(500, cfg["inference"]["randomization_inference_reps"])  # keep the smoke test fast
    print(f"\nRunning randomization inference (n_reps={ri_reps}, this refits {ri_reps} regressions) ...")
    ri_result = randomization_inference(panel, _beta_fn, permute_col="is_tech", n_reps=ri_reps)
    print(ri_result)

    pd.DataFrame([{**wb_result, **ri_result}]).to_csv(
        OUTPUT_TABLES_DIR / "inference_baseline_did.csv", index=False
    )
    sys.exit(0)
