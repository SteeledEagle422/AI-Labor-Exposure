"""
Run the full pipeline end-to-end, in dependency order. Steps that need
something only reachable from your own machine (BLS API key, OEWS file,
Google Trends) are clearly marked and will not silently no-op -- they raise
or warn loudly so a partial run is obvious, not hidden.

Usage:
    python -m src.run_all                  # everything
    python -m src.run_all --skip-manual    # skip BLS/OEWS/Trends steps,
                                            # run only what works with zero
                                            # setup (Indeed + exposure fetch)
"""
import argparse
import sys
import traceback


def _step(name, fn):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        traceback.print_exc()
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manual", action="store_true",
                         help="Skip steps that need a BLS API key, a manually "
                              "downloaded OEWS file, or Google Trends access.")
    args = parser.parse_args()

    ok = True

    from src.data.fetch_indeed_hiring_lab import fetch_indeed_data
    ok &= _step("1. Fetch Indeed Hiring Lab postings (automatic)", fetch_indeed_data)

    from src.data.fetch_eloundou_exposure import fetch_eloundou_scores
    ok &= _step("2. Fetch Eloundou exposure scores (automatic)", fetch_eloundou_scores)

    from src.data.fetch_aioe_exposure import fetch_aioe_scores
    ok &= _step("3. Fetch AIOE exposure scores (automatic)", fetch_aioe_scores)

    if not args.skip_manual:
        from src.data.fetch_oews_staffing_patterns import fetch_oews_staffing_patterns
        ok &= _step(
            "4. Fetch OEWS staffing patterns (auto-attempt; manual fallback "
            "documented in the script if BLS moves the file)",
            fetch_oews_staffing_patterns,
        )

        from src.data.fetch_bls_ces import fetch_bls_ces
        ok &= _step(
            "5. Fetch BLS CES employment series (needs BLS_API_KEY env var)",
            fetch_bls_ces,
        )

        from src.data.fetch_google_trends import fetch_google_trends
        ok &= _step(
            "6. Fetch Google Trends mechanism terms (best-effort, may 429)",
            fetch_google_trends,
        )
    else:
        print("\n[SKIPPED] OEWS / BLS CES / Google Trends -- run without "
              "--skip-manual once those are set up locally.")

    from src.data.build_exposure_crosswalk import build_sector_exposure_scores
    ok &= _step(
        "7. Build OEWS-weighted sector exposure scores (needs step 4)",
        build_sector_exposure_scores,
    )

    from src.data.build_panel import build_panel
    ok &= _step("8. Build the analysis panel", build_panel)

    from src.analysis.did_baseline import run_baseline_did
    from src.analysis.continuous_exposure_did import run_continuous_exposure_did
    from src.analysis.triple_diff import run_triple_diff
    from src.utils.config import PROCESSED_DIR
    import pandas as pd

    def _run_specs():
        panel = pd.read_csv(PROCESSED_DIR / "analysis_panel.csv", parse_dates=["period"])
        print(run_baseline_did(panel))
        print(run_continuous_exposure_did(panel))
        print(run_triple_diff(panel))

    ok &= _step("9. Run baseline / continuous-exposure / triple-diff specs", _run_specs)

    from src.analysis.event_study import run_event_study
    from src.utils.config import PROCESSED_DIR as PD2

    def _run_event_study():
        panel = pd.read_csv(PD2 / "analysis_panel.csv", parse_dates=["period"])
        print(run_event_study(panel).to_string(index=False))

    ok &= _step("10. Run event study", _run_event_study)

    def _run_inference():
        from src.analysis.inference import (
            wild_cluster_bootstrap_pvalue, randomization_inference,
        )
        from src.analysis.utils import absorb_fixed_effects
        from src.utils.config import load_config, OUTPUT_TABLES_DIR, ensure_dirs
        import numpy as np

        cfg = load_config()
        ensure_dirs(OUTPUT_TABLES_DIR)
        panel = pd.read_csv(PROCESSED_DIR / "analysis_panel.csv", parse_dates=["period"])

        df = panel[panel["group"].isin(("tech", "stem_control", "non_stem_control"))].copy()
        df["log_postings"] = np.log(df["indeed_job_postings_index"].clip(lower=0.01))
        df["treat_post"] = df["is_tech"] * df["post_chatgpt_launch"]
        resid = absorb_fixed_effects(df, cols=["log_postings", "treat_post"],
                                     fe_groups=["display_name", "period"])
        y = (resid["log_postings"] - resid["log_postings"].mean()).to_numpy()
        x = (resid["treat_post"] - resid["treat_post"].mean()).to_numpy()

        wb = wild_cluster_bootstrap_pvalue(
            y, x, df["display_name"], B=cfg["inference"]["wild_bootstrap_reps"])
        print(wb)

        def _beta_fn(p):
            return run_baseline_did(p)["beta"]

        ri = randomization_inference(
            panel, _beta_fn, permute_col="is_tech",
            n_reps=cfg["inference"]["randomization_inference_reps"])
        print(ri)

        pd.DataFrame([{**wb, **ri}]).to_csv(
            OUTPUT_TABLES_DIR / "inference_baseline_did.csv", index=False)

    ok &= _step("11. Inference: wild cluster bootstrap + randomization", _run_inference)

    def _run_robustness():
        from src.analysis import robustness as rb
        from src.utils.config import OUTPUT_TABLES_DIR, ensure_dirs

        ensure_dirs(OUTPUT_TABLES_DIR)
        panel = pd.read_csv(PROCESSED_DIR / "analysis_panel.csv", parse_dates=["period"])

        placebo = rb.placebo_event_date(panel, "2021-09-01")
        print(placebo)
        pd.DataFrame([placebo]).to_csv(
            OUTPUT_TABLES_DIR / "robustness_placebo_event_date.csv", index=False)

        alt_ctrl = rb.alt_control_group_comparison(panel)
        print(alt_ctrl[["control_group_label", "beta", "se_cluster", "n_sectors"]])
        alt_ctrl.to_csv(
            OUTPUT_TABLES_DIR / "robustness_alt_control_groups.csv", index=False)

        # These two need the continuous exposure scores (step 7), so they stay
        # non-fatal -- a missing OEWS file shouldn't sink the whole step.
        for label, fn in [("alt event date", lambda: rb.alt_event_date_comparison(panel)),
                          ("alt exposure measure", lambda: rb.alt_exposure_measure_comparison(panel))]:
            try:
                print(fn())
            except Exception as e:
                print(f"  ({label} skipped -- {e})")

        try:
            sc = rb.synthetic_control(panel, treated_sector="Software Development")
            print({k: v for k, v in sc.items() if k != "gap_series"})
        except Exception as e:
            print(f"  (synthetic control skipped -- {e})")

    ok &= _step("12. Robustness: placebo, alt controls, alt exposure, synthetic control",
                _run_robustness)

    print(f"\n{'=' * 70}")
    print("Pipeline finished." if ok else "Pipeline finished WITH FAILURES -- see [FAILED] steps above.")
    print(f"{'=' * 70}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
