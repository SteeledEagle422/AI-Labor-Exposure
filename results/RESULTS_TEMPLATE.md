# Results -- run of <YYYY-MM-DD>

## Run metadata

| Field | Value |
|-------|-------|
| Date run | |
| Panel span | |
| Sectors (tech / STEM ctrl / non-STEM ctrl) | |
| Indeed data vintage (rows in `sector_US.csv`) | |
| OEWS vintage | |
| `sample.end_date` in config | |
| Primary event date | |

## Status of each pipeline step

| # | Step | Status | Note |
|---|------|--------|------|
| 1 | Indeed postings | | |
| 2 | Eloundou exposure | | |
| 3 | AIOE exposure | | |
| 4 | OEWS staffing patterns | | |
| 5 | BLS CES | | |
| 6 | Google Trends | | |
| 7 | Exposure crosswalk | | |
| 8 | Build panel | | |
| 9 | Three headline specs | | |
| 10 | Event study | | |
| 11 | Inference | | |
| 12 | Robustness | | |

## Headline specs

The point of this table is the *movement* across rows as the macro shock gets
absorbed -- not any single number.

| Spec | beta (log pts) | Cluster SE | t | Wild-boot p | Interpretation |
|------|---------------|-----------|---|-------------|----------------|
| 1. Baseline binary DiD | | | | | |
| 2. Continuous-exposure DiD | | | | | |
| 3. Triple-diff (main claim) | | | | | |

## Event study

Pre-trend verdict (flat / not flat), and the bin coefficients:

| Bin | beta | SE | 95% CI |
|-----|------|----|--------|

## Inference (few-clusters corrections)

| Method | p-value | Notes |
|--------|---------|-------|
| Cluster-robust (reference only) | | |
| Wild cluster bootstrap (primary) | | |
| Randomization inference | | |

## Robustness

| Check | Result | Passes? |
|-------|--------|---------|
| Placebo event date | | |
| Alt event date (Copilot GA vs ChatGPT) | | |
| Alt control groups | | |
| Alt exposure measure | | |
| Synthetic control | | |

## Threats to validity surfaced by THIS run

<!-- Be specific and honest. A failed placebo or a pre-trend goes here even
     if the headline coefficient looks great. -->

## What could not be run, and why

<!-- Blocked steps, missing inputs, rate limits. -->

## Conclusion

<!-- What can and cannot be claimed on the strength of this run. -->
