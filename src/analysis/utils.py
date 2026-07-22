"""
Shared utilities for the analysis scripts.

Core piece: `absorb_fixed_effects`, a generic iterative-demeaning ("alternating
projections") absorber that can net out an arbitrary list of categorical
fixed-effect groupings -- not just entity/time, but also non-nested effects
like group x period (needed for the triple-diff in Sec 5, where "the tech-wide
shock is absorbed into time fixed effects" that vary by group).

Why iterative demeaning instead of linearmodels.PanelOLS directly: PanelOLS
handles entity/time effects natively and is used for the two simpler specs
(baseline DiD, continuous-exposure DiD), but the triple-diff needs an
*additional*, non-nested group x period effect that doesn't fit its
entity/time structure. `other_effects` in PanelOLS can do this internally,
but wild cluster bootstrap inference (Cameron-Gelbach-Miller, Sec 5) needs a
plain OLS fit on the *residualized* variables to feed into `wildboottest`,
which only accepts statsmodels model objects, not linearmodels ones. So:
absorb everything via demeaning, then run OLS (no intercept) on the residuals
-- this reproduces the FE point estimates exactly (Frisch-Waugh-Lovell) and
gives us a statsmodels object to bootstrap.

This is validated against linearmodels.PanelOLS in tests/test_absorb_fe.py --
run that after editing this function.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def absorb_fixed_effects(
    df: pd.DataFrame,
    cols: list[str],
    fe_groups: list[str],
    max_iter: int = 500,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Residualize `cols` in `df` against an arbitrary set of categorical
    fixed effects in `fe_groups` (each a column name), via iterative
    demeaning. Handles any number of possibly-non-nested groupings (e.g.
    sector FE + group-by-period FE simultaneously).

    Returns a copy of df[cols] with each column replaced by its residual
    after absorbing all fe_groups (i.e. still has the same mean as before --
    only within-group variation is removed group by group, then re-added
    grand mean at the end so coefficients on OTHER included regressors are
    unaffected by an arbitrary additive constant).
    """
    out = df[cols].copy().astype(float)
    grand_means = out.mean()

    group_codes = {g: df[g] for g in fe_groups}

    for _ in range(max_iter):
        max_delta = 0.0
        for g in fe_groups:
            means = out.groupby(group_codes[g]).transform("mean")
            delta = means.abs().to_numpy().max()
            out = out - means
            max_delta = max(max_delta, delta)
        if max_delta < tol:
            break

    out = out + grand_means  # re-center so this isn't a pure-residual (mean 0) frame
    return out


def cluster_robust_se(resid: np.ndarray, X: np.ndarray, cluster: pd.Series) -> np.ndarray:
    """CRV1 cluster-robust variance-covariance matrix for an OLS fit on
    already-residualized (FE-absorbed) X, y. Used as the "textbook" SE to
    report alongside the wild-bootstrap p-value, per Sec 5's inference plan
    (report both, since few-cluster bootstrap is the trusted one)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    clusters = cluster.unique()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in clusters:
        mask = (cluster == c).to_numpy()
        Xg = X[mask]
        ug = resid[mask]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n, k = X.shape
    g = len(clusters)
    dof_adj = (g / (g - 1)) * ((n - 1) / (n - k))
    vcov = dof_adj * XtX_inv @ meat @ XtX_inv
    return vcov


def build_group_period(df: pd.DataFrame, group_col: str = "group", period_col: str = "period") -> pd.Series:
    """Combined categorical for group x period fixed effects."""
    return df[group_col].astype(str) + "__" + df[period_col].astype(str)


def event_time_bins(event_time: pd.Series, bin_width: int, window: int, ref_bin_label: str = "ref") -> pd.Series:
    """Bin a continuous event-time variable (e.g. weeks_since_event) into
    bin_width-wide bins within [-window, +window], collapsing anything
    outside the window into 'far_pre' / 'far_post' catch-all bins, and
    labeling the last pre-period bin as the omitted reference category.
    Used by event_study.py to keep the number of lead/lag dummies manageable."""
    def label(t):
        if t < -window:
            return "far_pre"
        if t > window:
            return "far_post"
        # bins like [-window, -window+bin_width), ..., includes 0 in a "post" bin
        bin_start = int(np.floor(t / bin_width) * bin_width)
        if -bin_width <= t < 0:
            return ref_bin_label  # last pre-period bin = reference, omitted
        sign = "p" if bin_start >= 0 else "m"
        return f"{sign}{abs(bin_start)}"

    return event_time.apply(label)
