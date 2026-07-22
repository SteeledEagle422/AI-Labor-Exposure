# Paper Outline (section-by-section requirements)

This is the original planning document this repo's data/analysis pipeline
was built against. Kept here so the literature review + write-up pass
(deferred to later) has the full spec on hand. Section numbers below map to
where each piece of this repo fits:

- Sec 4 (Data)              -> `src/data/*`, `config/config.yaml`, `config/sector_naics_map.csv`
- Sec 5 (Empirical Strategy) -> `src/analysis/did_baseline.py`, `continuous_exposure_did.py`, `triple_diff.py`, `inference.py`
- Sec 6 (Results)            -> `output/tables/*`, `output/figures/event_study.png`
- Sec 7 (Robustness)         -> `src/analysis/robustness.py`
- Sec 8 (Limitations)        -> see README "What's still a judgment call"
- Sec 10 (Appendix)          -> this file + `config/config.yaml` (series IDs, terms) + repo itself

---

## 1. Introduction
Open with the empirical question and why it matters (labor market impact of generative AI is a live policy/academic debate — motivates the paper without overselling it)
State the naive hypothesis a casual observer would test (tech vs. non-tech postings around ChatGPT's launch) and immediately flag why it's confounded — 2022–23 rate hikes triggered a tech-sector-wide correction independent of AI, so any binary pre/post comparison conflates two shocks
Preview your fix: continuous AI-exposure design + triple-diff that nets out the sector-wide macro shock, rather than abandoning the simple approach
State contribution in one sentence: not "AI affected jobs" (too broad/already claimed elsewhere) but something narrower and defensible, e.g., "differential exposure to AI coding tools predicts differential posting declines within tech, net of the 2022–23 correction"
Roadmap of paper structure

## 2. Related Literature (short — this is a portfolio paper, not a dissertation)
Eloundou, Manning, Mishkin & Rock (2023) — exposure score methodology, cite as your weighting source
Felten, Raj, Seamans (AIOE) — alternative exposure measure, mention as robustness alternative
Any existing labor economics on the 2022–23 tech layoffs (to justify why you treat it as a separate, controllable-for shock rather than ignoring it)
Prior DiD/event-study applications to tech shocks (China shock literature — Autor/Dorn/Hanson — as the methodological ancestor of your exposure-weighted design)
Position your paper: most existing AI-jobs studies either (a) ignore the macro confound or (b) use survey/self-report data; you use high-frequency postings + a design that isolates AI from the rate-hike shock

## 3. Institutional Background / Timeline
Timeline of relevant events with exact dates: GitHub Copilot GA (June 2022), Fed hiking cycle start (March 2022), major tech layoff waves (late 2022 through 2023), ChatGPT launch (Nov 30, 2022), GPT-4/Copilot Chat (early 2023)
Brief explanation of why Copilot GA is a cleaner "AI treatment" date than ChatGPT (narrower, coding-specific, less entangled with broader AI hype cycle)
This section is what lets you justify picking multiple event dates later — needs to exist so the event-date choice doesn't look arbitrary

## 4. Data
Outcome data: Indeed Hiring Lab postings index — describe frequency (daily), sector categories used, date range, any known limitations (Indeed-only, not representative of all postings platforms)
Confirmatory data: BLS CES employment series (NAICS 5415 vs. 5413 vs. 54 aggregate) — describe series IDs pulled, frequency, why it's a lower-frequency robustness check rather than primary outcome
Mechanism data: Google Trends series list and justification for each term chosen
Exposure weights: source (Eloundou et al. published scores), how mapped to your occupation/sector categories, any crosswalk issues (SOC vs. NAICS mismatch — you'll need to address this explicitly since Indeed/BLS aren't on the same classification system)
Panel construction description: unit of observation, time aggregation choice (weekly vs. monthly and why), final sample date range and N
Summary statistics table description (what it needs to show: pre-trend levels, variance across sectors, exposure score distribution)

## 5. Empirical Strategy
Naive/baseline DiD: full equation, define treatment/control, define event time
Event-study version: equation with leads/lags, explicit statement that pre-period coefficients are your parallel-trends test
Continuous exposure design: full equation, define exposure variable construction, interpretation of β
Triple-diff: equation, explain exactly what variation identifies β once the tech-wide shock is absorbed into time fixed effects — this is the section that needs the most care since it's your main identification claim
Inference plan: state the few-clusters problem explicitly, name the correction you'll use (wild cluster bootstrap or randomization inference) and cite the methodological source for it (Cameron-Gelbach-Miller for wild bootstrap)

## 6. Results
Baseline DiD point estimate + naive interpretation, followed immediately by why you don't trust it alone
Event-study plot description: what pre-trends should look like if design is valid, what a violation would look like
Exposure-weighted DiD main result — this is your headline number
Triple-diff result — compare magnitude/significance to the plain exposure DiD to show what netting out the macro shock does to the estimate
Table shell: coefficients across specifications side by side so the reader sees how the estimate moves as you add rigor (this progression is itself the "skill display")

## 7. Robustness Checks
Placebo event date (random pre-period date) — expect null result, needs description of what you'll do if it's not null
Alternate event date (Copilot GA vs. ChatGPT) — compare estimates
Alternate control group definitions (all STEM / non-tech STEM only / all occupations) — stability check
Synthetic control as alternative to binary DiD given small comparison-group N
Alternate exposure measure (AIOE vs. Eloundou et al.) — do results survive a different weighting scheme

## 8. Limitations
Indeed postings are not a random/representative sample of all job postings (platform selection)
SOC-to-NAICS crosswalk introduces measurement error in exposure merge
Exposure scores themselves are expert/LLM-assessed, not ground truth — measurement error in the regressor, likely attenuating β
Small number of clusters even after fixes — inference caveat restated
Cannot separate labor demand effects from labor supply effects (fewer postings could mean fewer needed OR fewer qualified applicants expected) — postings ≠ employment, be explicit about what the outcome variable can and can't tell you
2022–23 window still has residual confounds you can't fully rule out (return-to-office policies, outsourcing trends, etc.)

## 9. Conclusion
Restate the narrow, defensible claim (not overreaching to "AI is replacing programmers")
One sentence on policy/practical relevance
One sentence on what a cleaner future test would need (e.g., firm-level adoption data linked to firm-level postings)

## 10. Appendix
Full BLS series ID list and API query parameters (shows technical execution)
Google Trends term list and pull methodology
Exposure score crosswalk table
Full regression tables for all robustness specifications
Data pull code snippets or repo link (this is where the "API skills" get demonstrated concretely)
