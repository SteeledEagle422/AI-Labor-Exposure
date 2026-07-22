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

    print(f"\n{'=' * 70}")
    print("Pipeline finished." if ok else "Pipeline finished WITH FAILURES -- see [FAILED] steps above.")
    print(f"{'=' * 70}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
