# AI Labor Exposure

Does differential exposure to LLM coding tools (GitHub Copilot, ChatGPT)
predict differential job-posting declines *within* tech -- net of the
2022-23 rate-hike-driven tech-sector correction that hit at almost exactly
the same time?

**Short answer, as of the 2026-07-23 run: no detectable effect.** The naive
tech-vs-non-tech comparison shows a large drop (-0.379 log points), but it
collapses to zero (+0.010, t = 0.25) once a group x period fixed effect absorbs
the tech-wide macro shock. Details and caveats in
[What this pipeline actually found](#what-this-pipeline-actually-found) -- in
particular, the crosswalk limits how much this can be read as a test of the
*within-tech* mechanism.

This repo is the **data + analysis pipeline** for that question. It is
deliberately scoped to data/code only -- no literature review and no written
paper yet (see `docs/paper_outline.md` for the full section-by-section plan
this pipeline was built against; the write-up comes later). Per-run results
live in `results/`.

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
   lives. What's left is meant to identify beta from *within-tech*
   differences: does Software Development (high exposure) fall more than
   Data & Analytics (lower exposure) in the same week, net of "tech had a
   bad week"?

Comparing the three -- and watching the estimate move as the macro shock
gets absorbed -- is the point.

> **In practice spec 3 does not deliver that within-tech comparison**, because
> the NAICS crosswalk assigns all four tech sectors nearly identical exposure
> scores (within-tech SD of standardized exposure is 0.014, vs 1.083 within
> the non-STEM control group). The estimate is real, but it is identified
> mostly off within-non-STEM variation. See
> [The crosswalk problem](#the-crosswalk-problem-read-this-before-trusting-any-exposure-weighted-result)
> -- this is currently the binding constraint on the whole design.

## What this pipeline actually found

Full write-up, including every robustness check and caveat:
[`results/RESULTS_2026-07-23.md`](results/RESULTS_2026-07-23.md).

Headline table (Eloundou beta exposure, standardized; 20 sectors, n=5,300):

| Spec | beta (log pts) | Cluster SE | t |
|------|---------------|-----------|---|
| 1. Baseline binary DiD | **-0.3789** | 0.0807 | -4.69 |
| 2. Continuous-exposure DiD | **-0.0711** | 0.0509 | -1.40 |
| 3. Triple-diff (main claim) | **+0.0101** | 0.0413 | **0.25** |

**The result is a null.** The naive tech-vs-non-tech comparison says tech
postings fell ~31.5%. Swapping the binary dummy for actual exposure scores cuts
that to an insignificant -0.071. Adding the group x period fixed effect -- which
absorbs the tech-wide rate-hike shock -- drives it to **+0.010, indistinguishable
from zero**. The null holds across all five exposure measures (Eloundou
alpha/beta/gamma, AIOE via crosswalk, AIOE direct NAICS) and both candidate
event dates.

Read that as: *once you absorb whatever hit all tech sectors in a given week,
differential LLM exposure has no detectable additional association with posting
declines in this panel.* It is **not** a well-powered test of the within-tech
mechanism -- see the caveat above and the crosswalk section below.

Two further findings that constrain how much weight spec 1 can carry:

- **The placebo test fails.** A fake event date of 2021-09-01 -- before ChatGPT
  and before the Fed hikes -- still yields -0.204 (t = -3.70), roughly half the
  spec-1 magnitude. Something other than the AI event moves tech postings
  relative to control across this whole sample.
- **Spec 1's significance straddles 5%** depending on the few-cluster
  correction: wild cluster bootstrap p = 0.0086, randomization inference
  p = 0.068.

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
results/                   committed write-up of what each run produced
  RESULTS_TEMPLATE.md      copy this per run
  RESULTS_<date>.md        one file per run (output/ itself is gitignored)
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
  the script attempts an automatic download, but **bls.gov rate-limits
  automated downloads and starts returning `403 Access Denied`**, so in
  practice expect to fetch this one from a browser. Download the May {year}
  "National Industry-Specific Occupational Employment and Wage Estimates"
  (all industries, 4-digit NAICS) from https://www.bls.gov/oes/tables.htm --
  for the May 2024 vintage that is
  https://www.bls.gov/oes/special-requests/oesm24in4.zip (~32 MB) -- and drop
  it at `data/raw/oews/oesm24in4.zip`. The script extracts and parses it from
  there without any further network access. **This file is the only thing
  standing between the pipeline and its two main specs** (continuous-exposure
  DiD and triple-diff), so it is worth the two minutes.

Google Trends (`src/data/fetch_google_trends.py`) needs no key but Google
will throttle/CAPTCHA aggressive pulls -- run it locally, not in CI, and
expect to re-run it if it 429s partway through.

### What has actually been run end-to-end

As of the 2026-07-23 run, **every step except Google Trends has been executed
against real, live data** -- including the BLS CES API (with a registered key)
and the full OEWS crosswalk. Numbers in this README and in `results/` come from
that run, not from a synthetic fixture.

| Step | Status |
|------|--------|
| Indeed, Eloundou, AIOE fetches | run |
| OEWS staffing patterns | run (file downloaded manually -- bls.gov 403s automated requests) |
| BLS CES API | run (720 rows, 8 series) |
| Google Trends | **not run** -- Google returns 429; mechanism data only, blocks no spec |
| Crosswalk, panel, all 3 specs, event study, inference, robustness | run |

Two caveats on the event study specifically, since an earlier version of this
README overstated it:

- Pre-trends are **flat near the event** (bins -24 through -8 weeks are all
  individually insignificant) but **not clean overall**: the `far_pre`
  catch-all bin is -0.100 (95% CI -0.185 to -0.016), i.e. tech was already
  drifting below control before the window opens.
- The event study is the *binary* tech-vs-non-tech spec, so its monotonic
  post-period decline inherits spec 1's confound. It is a diagnostic, not
  evidence of an AI effect.

`output/` is gitignored; see `results/` for the committed numbers.

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

## The crosswalk problem (read this before trusting any exposure-weighted result)

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

### This is currently the binding constraint on the design

Routing sector -> NAICS -> SOC collapses distinct Indeed sectors onto shared
NAICS codes, which flattens exactly the variation the triple-diff needs:

| Group | SD of standardized exposure | distinct values |
|-------|---------------------------|-----------------|
| tech | **0.014** | 3 (of 4 sectors) |
| stem_control | 0.127 | 2 (of 6 sectors) |
| non_stem_control | **1.083** | 10 (of 10 sectors) |

All four tech sectors map through NAICS 5415, so their exposure scores differ
by ~0.003 raw units. All five engineering/architecture sectors map to 5413 and
receive **identical** scores. The result: spec 3's beta is identified roughly
77x more by within-non-STEM variation than by the within-tech comparison the
design is built around.

**Fixing this is worth more than any additional robustness check.** The likely
route is to weight each Indeed sector by its own occupation mix directly,
rather than inheriting a NAICS industry's staffing pattern.

### A note on NAICS vintages

The May 2024 OEWS uses **NAICS 2022**. Eight codes in `sector_naics_map.csv`
predated that revision and silently matched nothing (or fell back to a coarse
prefix) until corrected -- including `5112` Software Publishers, which had
quietly reduced Software Development, the primary treated sector, to NAICS 5415
alone, and `4521` department stores, which left Retail with zero coverage and
dropped it from every exposure-weighted spec. Current mappings use `5132`,
`4550`, `5220`, `5230`, `4240`, `5161`, `5192`, `4561`.

If you move to a different OEWS vintage, re-check every code: a retired code
fails quietly, and the pipeline will still produce plausible-looking numbers.

## Inference

With 20 sector clusters (and only 4 treated), standard cluster-robust SEs are
unreliable. `src/analysis/inference.py` implements wild cluster bootstrap
(Cameron-Gelbach-Miller, via the `wildboottest` package) as the primary
inference, plus a randomization-inference (permutation) check as a second, more
primitive cross-check. Report the wild-bootstrap p-value as primary in any
write-up; the plain cluster-robust SE in the regression output tables is
there for comparability, not as the trusted number.

In the 2026-07-23 run the two corrections **disagreed across the 5% line** on
spec 1 (bootstrap p = 0.0086, RI p = 0.068). With 4 treated sectors, RI has
little power and its discreteness matters, so this isn't fatal on its own --
but quote both, not just the bootstrap.

## What's still a judgment call / not fully automated

- `config/sector_naics_map.csv` -- see above. Still the biggest open problem.
- OEWS file download -- expect to fetch it from a browser; bls.gov 403s
  automated requests after a few hits. Manual step documented in
  `src/data/fetch_oews_staffing_patterns.py`.
- Google Trends -- rate-limited; 429'd on the last run. May need re-running or
  added delay between terms.
- BLS CES series list in `config/config.yaml` covers NAICS 5415, 5413, 54
  aggregate, the Professional & Business Services supersector, and total
  nonfarm -- verified against FRED as of July 2026. Add more series IDs
  there if you want additional confirmatory industries. **Note: these series
  are fetched but not yet consumed by any analysis script** -- the
  confirmatory lower-frequency outcome is described here but not implemented.

## Next steps

Roughly in order of how much they'd change the conclusions:

1. **Replace the NAICS crosswalk with a direct sector -> occupation-mix
   weighting.** Until tech sectors get genuinely different exposure scores,
   spec 3 cannot test the within-tech mechanism it was designed for. Everything
   else is secondary.
2. **Implement the BLS CES confirmatory outcome.** The data is already being
   fetched; nothing reads it. A monthly employment-based replication would be a
   real check on an Indeed-postings-only result.
3. Investigate the failed placebo / `far_pre` drift -- what is pushing tech
   postings down relative to control before any event? Until that is named, the
   binary spec cannot be interpreted causally.
4. Revisit `crosswalk_confidence: low` rows and consider dropping them from the
   exposure-weighted specs rather than just flagging them.
5. Literature review + full write-up (paper outline exists, not started).
6. Appendix tables / full regression output for every robustness spec.
