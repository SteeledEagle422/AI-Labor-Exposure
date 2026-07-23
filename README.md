# AI Labor Exposure

Does differential exposure to LLM coding tools (GitHub Copilot, ChatGPT)
predict differential job-posting declines *within* tech -- net of the
2022-23 rate-hike-driven tech-sector correction that hit at almost exactly
the same time?

This repo is the **data + analysis pipeline** for that question. It is
deliberately scoped to data/code only -- no literature review, no written
paper yet (see `docs/paper_outline.md` for the full section-by-section plan
this pipeline was built against; the write-up comes later).

## The identification problem this is built around

A naive "tech vs. non-tech postings before/after ChatGPT" comparison
conflates two shocks that landed almost on top of each other: the Fed's 2022
hiking cycle triggered a tech-wide correction independent of AI. This repo
implements three specs of increasing rigor to deal with that:

1. **Baseline binary DiD** -- tech vs. non-tech, single event date. Naive,
   confounded, included only as the "what a casual observer would compute" baseline.
2. **Continuous-exposure DiD** -- replaces the tech/non-tech dummy with each
   sector's actual LLM exposure score (Eloundou et al. 2023 / AIOE), still
   using plain sector + period fixed effects.
3. **Triple-diff** -- adds a *group x period* fixed effect (group =
   tech / STEM-control / non-STEM-control), which absorbs whatever hit all
   tech sectors equally in a given week -- exactly where the rate-hike shock
   lives. What's left identifies beta from *within-tech* differences: does
   Software Development (high exposure) fall more than Data & Analytics
   (lower exposure) in the same week, net of "tech had a bad week"?

Comparing the three -- and watching the estimate move as the macro shock
gets absorbed -- is the point.

## Repo layout

```
config/
  config.yaml              central config: dates, series IDs, sectors, terms
  sector_naics_map.csv     Indeed sector -> representative NAICS code(s)
                           (hand-curated -- this is a real judgment call, see below)
src/
  data/                    fetch + build scripts (see "Pipeline" below)
  analysis/                DiD / event-study / triple-diff / inference / robustness
  utils/config.py          shared config loader
  run_all.py               orchestrator -- runs everything in order
tests/
  test_absorb_fe.py        validates the core FE-absorption routine against
                           linearmodels.PanelOLS -- run this if you touch
                           src/analysis/utils.py
data/                      raw/interim/processed (gitignored except .gitkeep)
output/                    tables/ and figures/ (gitignored except .gitkeep)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two things need a one-time, free registration before their fetch scripts
will run at full strength:

- **BLS API key** (for `src/data/fetch_bls_ces.py`): register at
  https://data.bls.gov/registrationEngine/, then either
  `export BLS_API_KEY="your-key"` or copy `.env.example` to `.env` and drop
  the key in there (`cp .env.example .env`, then edit it) -- `.env` is
  gitignored and picked up automatically by every script via
  `src/utils/config.py`. Without it the script still runs against BLS's
  unregistered endpoint but is capped at ~10 years of history and
  25 queries/day.
- **OEWS staffing-pattern file** (for `src/data/fetch_oews_staffing_patterns.py`):
  the script attempts an automatic download; if BLS has restructured their
  URLs (they do this periodically), download the May {year} "National
  Industry-Specific Occupational Employment and Wage Estimates" (4-digit
  NAICS) file by hand from https://www.bls.gov/oes/tables.htm and drop it at
  `data/raw/oews/oesm{YY}in4.xlsx`. See the script's docstring for the exact
  expected filename.

Google Trends (`src/data/fetch_google_trends.py`) needs no key but Google
will throttle/CAPTCHA aggressive pulls -- run it locally, not in CI, and
expect to re-run it if it 429s partway through.

**None of api.bls.gov, trends.google.com, or bls.gov downloads were
reachable from the sandbox this project was scaffolded in**, so those three
scripts are written against their documented APIs but were not executed
end-to-end here -- run them yourself. Everything else (Indeed Hiring Lab,
Eloundou exposure scores, AIOE exposure scores, the crosswalk logic, every
regression, the event-study plot) **was run against the real, live data** as
part of building this repo and produced sane, plausible results (e.g. the
baseline DiD comes back at roughly -31% tech postings relative to control
post-ChatGPT, p<0.01 by wild cluster bootstrap; the event study shows flat
pre-trends and a growing post-period divergence -- see `output/` for the
actual numbers from that test run).

## Pipeline

Run everything:
```bash
python -m src.run_all              # full pipeline, needs BLS key + OEWS file set up
python -m src.run_all --skip-manual  # only the zero-setup steps (see below)
```

Or step by step, in this order:

| # | Script | Needs | Notes |
|---|--------|-------|-------|
| 1 | `src.data.fetch_indeed_hiring_lab` | nothing | outcome data, daily, 2020-present |
| 2 | `src.data.fetch_eloundou_exposure` | nothing | SOC-level LLM exposure scores |
| 3 | `src.data.fetch_aioe_exposure` | nothing | SOC-level + NAICS-level AI exposure (robustness measure) |
| 4 | `src.data.fetch_oews_staffing_patterns` | BLS download (see Setup) | SOC-employment-within-NAICS, the crosswalk weights |
| 5 | `src.data.fetch_bls_ces` | `BLS_API_KEY` | confirmatory lower-frequency outcome |
| 6 | `src.data.fetch_google_trends` | nothing, but rate-limited | mechanism data |
| 7 | `src.data.build_exposure_crosswalk` | steps 2-4 | OEWS-weighted sector exposure scores |
| 8 | `src.data.build_panel` | step 1 (+ 7 for continuous-exposure specs) | final sector x week panel |
| 9 | `src.analysis.did_baseline` / `.continuous_exposure_did` / `.triple_diff` | step 8 | the three headline specs |
| 10 | `src.analysis.event_study` | step 8 | parallel-trends check + plot |
| 11 | `src.analysis.inference` | step 8 | wild cluster bootstrap + randomization inference |
| 12 | `src.analysis.robustness` | step 8 | placebo dates, alt controls, alt exposure, synthetic control |

## The crosswalk problem (read this before trusting continuous-exposure results)

Three classification systems don't line up: Indeed's sectors are
occupation-title categories it defines itself, Eloundou/AIOE exposure scores
are SOC-coded, BLS CES is NAICS-coded. `config/sector_naics_map.csv` is a
hand-built bridge from Indeed sector -> representative NAICS code(s), with a
`crosswalk_confidence` column (`high`/`medium`/`low`) and a `group` column
that also flags sectors too heterogeneous to trust in the exposure-weighted
specs (`excluded`) vs. usable as tech/STEM-control/non-STEM-control.
**Review and adjust this file** -- it's a judgment call, not a fact, and the
paper's Limitations section should say so explicitly.

`src/data/build_exposure_crosswalk.py` then uses OEWS staffing patterns
(SOC employment within each NAICS) to turn SOC-level exposure scores into
employment-weighted sector-level scores, with a documented match-quality
fallback (exact NAICS match -> 4-digit prefix -> 3-digit prefix) and a
`naics_match_level` diagnostic column in its output so a coarse, lower-
confidence match is visible rather than silently blended in.

## Inference

With ~13-20 sector clusters, standard cluster-robust SEs are unreliable.
`src/analysis/inference.py` implements wild cluster bootstrap (Cameron-
Gelbach-Miller, via the `wildboottest` package) as the primary inference,
plus a randomization-inference (permutation) check as a second, more
primitive cross-check. Report the wild-bootstrap p-value as primary in any
write-up; the plain cluster-robust SE in the regression output tables is
there for comparability, not as the trusted number.

## What's still a judgment call / not fully automated

- `config/sector_naics_map.csv` -- see above.
- OEWS file download -- automatic attempt, manual fallback documented in
  `src/data/fetch_oews_staffing_patterns.py`.
- Google Trends -- works, but Google's rate limiting means you may need to
  re-run it or add delay between terms.
- BLS CES series list in `config/config.yaml` covers NAICS 5415, 5413, 54
  aggregate, the Professional & Business Services supersector, and total
  nonfarm -- verified against FRED as of July 2026. Add more series IDs
  there if you want additional confirmatory industries.

## Next steps (not in this pass)

- Literature review + full write-up (paper outline exists, not started)
- Once real OEWS data is in, revisit `crosswalk_confidence: low` rows in
  `sector_naics_map.csv` and consider dropping them from the continuous-
  exposure specs rather than just flagging them
- Appendix tables / full regression output for every robustness spec
