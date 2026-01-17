from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..core.state_machine import Phase
from ..utils import logging
from .base import Agent, AgentResult, AgentTask

LOGGER = logging.get_logger("agent.pm")


class ProjectManagerAgent(Agent):
    name = "Project Manager"
    actor = "pm"
    supported_phases = [Phase.BRIEF]

    def run(self, task: AgentTask, *, context) -> AgentResult:  # type: ignore[override]
        LOGGER.info("Compiling Weekly Health & Longevity Plan")
        plan_dir = context.config.artifact_path("briefings")
        plan_path = plan_dir / "weekly_plan.json"
        plan_payload = self._build_plan(context)
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload, indent=2))

        artifact = context.artifacts.register(
            name="briefings/weekly_plan",
            path=plan_path,
            description="Weekly Health & Longevity Plan",
            mime_type="application/json",
        )

        summary = plan_payload["snapshot"]["highlight"]
        return AgentResult(
            summary=summary,
            artifacts=[artifact.ref],
            preview=summary,
            metadata=plan_payload,
            next_actions=["Notify user"],
        )

    def _build_plan(self, context) -> Dict[str, Any]:
        timeline = context.timeline
        latest = timeline.latest_snapshot(days=7)
        highlight = (
            f"{len(latest)} measurements processed in last 7 days" if not latest.empty else "Awaiting new data"
        )
        ds_summary = self._load_json(context.cache.get("eda_summary"))
        model_results = self._load_json(context.cache.get("model_results"))
        med_context = self._load_json(context.cache.get("medical_context"))
        longevity_plan = self._load_json(context.cache.get("longevity_plan"))
        hypothesis_registry = self._load_json(context.cache.get("hypothesis_registry"))

        ds_key_figures = []
        if isinstance(model_results, dict):
            ds_key_figures = list(model_results.get("headlines", []))
        elif isinstance(model_results, list):
            for row in model_results[:3]:
                exposure = row.get("exposure")
                outcome = row.get("outcome")
                beta = row.get("beta")
                p_value = row.get("p_value")
                if exposure and outcome and beta is not None and p_value is not None:
                    ds_key_figures.append(f"{exposure} -> {outcome} beta={float(beta):.3f} (p={float(p_value):.3g})")

        return {
            "snapshot": {
                "highlight": highlight,
                "phase": context.state.phase.value,
                "timeline_rows": int(timeline.data.shape[0]) if not timeline.data.empty else 0,
            },
            "ds_key_figures": ds_key_figures,
            "medical_context": {
                "insights": med_context.get("insights", []) if isinstance(med_context, dict) else [],
                "questions": med_context.get("questions", []) if isinstance(med_context, dict) else [],
                "disclaimer": med_context.get("safety") if isinstance(med_context, dict) else None,
            },
            "hypotheses": hypothesis_registry.get("hypotheses", []) if isinstance(hypothesis_registry, dict) else [],
            "longevity_plan": longevity_plan if isinstance(longevity_plan, dict) else {},
            "prevention": longevity_plan.get("prevention", []) if isinstance(longevity_plan, dict) else [],
        }

    def _load_json(self, path: Path | None) -> Dict[str, Any]:
        if not path or not Path(path).exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
