# Long Health Coach Multi-Agent Workflow

**Disclaimer:** Experimental prototype for personal use only. Not health advice or medical guidance. Consult a qualified physician before making health decisions.

Self-directed, interruptible multi-agent system for longitudinal health and longevity coaching. Implements the specification provided for the Project Manager (orchestrator), Data Scientist, Medical Professional, Hypothesis-Driven Researcher, and Health & Longevity Coach agents.

## Features

- Deterministic state machine covering ingest → EDA → modeling → interpretation → coaching → hypothesis → briefing phases.
- Typed timeline store that unifies longitudinal labs, lifestyle, and supplement data in long format.
- Data Scientist agent performs ingestion, QC, exploratory analysis, and exposure→outcome modeling with reproducible artifacts.
- Medical Professional agent provides safety-aware context and clinician discussion prompts (no diagnosis or prescriptions).
- Health & Longevity Coach agent converts insights into structured habit cards and prevention nudges.
- Hypothesis-Driven Researcher maintains a registry of hypotheses with posterior updates after each run.
- Project Manager (PM) orchestrates routing, event logging, and final Weekly Health & Longevity Plan generation.
- Artifact tracking with signed references for figures, tables, and plan outputs.

## Quickstart

1. (Optional) Set up a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Prepare one or more longitudinal CSV/XLSX files. Wide sheets (metabolites as rows, dates as columns) are automatically melted; long sheets should include date/value/unit columns.
3. Run the orchestrator:
   ```bash
   python -m long_health_coach.main --subject-id demo --run-id run001 --source data/labs.csv --source data/lifestyle.csv
   ```
4. Artifacts (timeline parquet, QC report, figures, coaching plan, weekly briefing, event log) are written to the `artifacts/` directory by default.

## LLM Runtime

- This workflow runs as a local Python pipeline; it is not designed to be driven by Codex or another external agent at runtime.
- There is no built-in LLM provider integration; agent outputs are deterministic. To use an LLM, add a provider client in the agent modules and supply the necessary API keys or environment config.
- You can also run the pipeline from an external agent (e.g., Codex or Claude Code) to interpret generated artifacts and iterate on prompts, analyses, or coaching plans.

## Repository Layout

- `src/long_health_coach/core/`: configuration, state machine, events, orchestrator, run context.
- `src/long_health_coach/data/`: ingestion helpers, timeline store, QC utilities.
- `src/long_health_coach/analytics/`: exploratory analysis and modeling routines.
- `src/long_health_coach/coaching/`: behavior-change plan synthesis.
- `src/long_health_coach/agents/`: agent definitions for PM, DS, MedPro, HDR, and HLC roles.
- `src/long_health_coach/main.py`: CLI entrypoint.
- `artifacts/`: output directory (created at runtime).

## Workflow Details

1. **Ingest & QC**
   - Detects wide vs. long layout; melts wide sheets to long format.
   - Normalizes columns (`subject_id`, `measurement_date`, `variable_name`, `value`, `unit`, `source`, `metadata`).
   - Generates QC diagnostics (missingness, outliers, unit consistency) and associated figures.

2. **EDA & Modeling**
   - Computes trend summaries, highlights, and time-series visualizations.
   - Runs deterministic OLS modeling for exposure→outcome pairs with confidence intervals and lag scans.

3. **Interpretation & Coaching**
   - Provides safety-framed medical context and clinician discussion questions.
   - Generates 3–4 structured habit cards plus prevention nudges and supplement log reminders.

4. **Hypotheses & Briefing**
   - Updates hypothesis registry with posterior adjustments based on new modeling evidence.
   - PM agent assembles the Weekly Health & Longevity Plan with snapshots, key figures, clinician prompts, and the Longevity Plan section.

## Interrupts & Extensions

- The orchestrator exposes `pause`, `resume`, and `snapshot` methods (see `Orchestrator` class) to integrate `/pause`, `/why`, or branching UX.
- Preferences (e.g., specific exposures/outcomes or autonomy level) can be injected via `RunConfig.preferences`.
- Additional tools (e.g., literature retrieval, visualization backends) can be registered via new agents or extended services.

## Safety & Compliance

- Medical Professional agent appends mandatory disclaimer: “This is educational information, not medical advice. Talk with a qualified clinician before making changes.”
- No diagnosis, prescriptions, or emergency detection logic is implemented; only informational nudges are produced.
- All outputs are informational coaching artifacts intended for clinician-guided review.

## Testing & Reproducibility

- Deterministic seeds and purely numerical analytics ensure reproducible output for the same inputs.
- Event logs (`artifacts/events/events.json`) capture provenance for each phase, including artifact references and HITL flags.
- Extend with unit tests (e.g., pytest) for ingestion, QC, analytics modules to guard against regressions.

## Next Steps

- Integrate a UI layer for `/pause`, `/why`, and `/branch` commands.
- Add artifact signing and provenance controls per security requirements.
- Connect to external data streams (wearables, surveys) via additional ingestion adapters.
