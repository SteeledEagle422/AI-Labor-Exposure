"""
Validate src.analysis.utils.absorb_fixed_effects against linearmodels.PanelOLS
on synthetic data with known structure. This is the load-bearing piece for
every DiD/event-study/triple-diff spec in src/analysis -- if this test fails,
don't trust any regression output downstream of it.

Run with: python -m tests.test_absorb_fe
"""
import sys
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])  # allow `python tests/test_absorb_fe.py`

from src.analysis.utils import absorb_fixed_effects, build_group_period


def _make_synthetic_panel(n_entities=20, n_periods=30, seed=0):
    rng = np.random.default_rng(seed)
    entities = [f"e{i}" for i in range(n_entities)]
    periods = list(range(n_periods))

    rows = []
    entity_fe = {e: rng.normal(scale=2.0) for e in entities}
    period_fe = {t: rng.normal(scale=1.5) for t in periods}
    true_beta = 1.7

    for e in entities:
        group = "A" if int(e[1:]) % 2 == 0 else "B"
        x_base = rng.normal(size=n_periods)
        for t, x in zip(periods, x_base):
            y = (
                true_beta * x
                + entity_fe[e]
                + period_fe[t]
                + rng.normal(scale=0.5)
            )
            rows.append({"entity": e, "period": t, "group": group, "x": x, "y": y})
    return pd.DataFrame(rows), true_beta


def test_two_way_fe_matches_panelols():
    from linearmodels.panel import PanelOLS

    df, true_beta = _make_synthetic_panel()

    # --- our iterative demeaning ---
    resid = absorb_fixed_effects(df, cols=["y", "x"], fe_groups=["entity", "period"])
    import statsmodels.api as sm
    ours = sm.OLS(resid["y"] - resid["y"].mean(), resid["x"] - resid["x"].mean()).fit()
    beta_ours = ours.params.iloc[0]

    # --- linearmodels reference ---
    panel_df = df.set_index(["entity", "period"])
    mod = PanelOLS(panel_df["y"], panel_df[["x"]], entity_effects=True, time_effects=True)
    res = mod.fit()
    beta_ref = res.params["x"]

    print(f"true beta:        {true_beta}")
    print(f"our beta (2-way): {beta_ours:.6f}")
    print(f"PanelOLS beta:    {beta_ref:.6f}")
    assert_allclose(beta_ours, beta_ref, rtol=1e-6, atol=1e-6)
    print("PASS: two-way FE (entity + period) matches linearmodels.PanelOLS")


def test_group_period_fe_absorbs_nonested_effect():
    """Sanity check for the triple-diff case: entity FE + group x period FE
    (non-nested with entity FE). We can't cross-check against PanelOLS here
    (it doesn't support this structure directly), so instead we check the
    absorbed regressor has zero within-group-period mean, which is the
    defining property of correct absorption."""
    df, _ = _make_synthetic_panel()
    df["group_period"] = build_group_period(df, "group", "period")

    resid = absorb_fixed_effects(df, cols=["y", "x"], fe_groups=["entity", "group_period"])
    check = resid.copy()
    check["group_period"] = df["group_period"]
    # absorb_fixed_effects re-adds the grand mean at the end (by design, so
    # regressions built on its output aren't shifted by an arbitrary
    # constant) -- so within-cell means should equal the grand mean, not 0.
    within_gp_means = (check.groupby("group_period")["x"].mean() - resid["x"].mean()).abs().max()

    print(f"max within-group-period deviation from grand mean, absorbed x: {within_gp_means:.2e}")
    assert within_gp_means < 1e-6
    print("PASS: entity + group-period FE absorption zeroes within-cell means")


if __name__ == "__main__":
    test_two_way_fe_matches_panelols()
    test_group_period_fe_absorbs_nonested_effect()
    print("\nAll absorb_fixed_effects tests passed.")
