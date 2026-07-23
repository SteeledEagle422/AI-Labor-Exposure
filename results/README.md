# Results space

This directory is the **write-up layer**: a durable, human-readable record of
what each pipeline run actually produced. `output/` holds the machine-generated
CSVs and figures and is gitignored; this directory is committed.

## How to use it

1. Put your BLS key in `.env` (`cp .env.example .env`, then edit).
2. Get the OEWS file in place -- see "Manual step" in the root README. This is
   the one input that cannot be fetched automatically.
3. Run the pipeline:

   ```bash
   python -m src.run_all
   ```

4. Copy `RESULTS_TEMPLATE.md` to `RESULTS_<YYYY-MM-DD>.md` and fill it in from
   `output/tables/`. Keep one file per run so results stay comparable across
   data vintages -- the Indeed index is revised, so the same spec will move a
   little between runs.

## Files

| File | What it is |
|------|-----------|
| `RESULTS_TEMPLATE.md` | Blank template -- copy this per run |
| `RESULTS_2026-07-23.md` | First real run (partial -- see its Status section) |

## Rule of thumb for filling these in

Record the spec **and** whether its identifying assumption survived, not just
the coefficient. A significant beta next to a failed placebo test is not a
result -- it is a diagnostic that something other than the treatment is moving
the outcome. Both belong in the write-up.
