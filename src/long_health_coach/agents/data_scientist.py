from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..analytics import eda, modeling
from ..core.state_machine import Phase
from ..data import ingestion
from ..utils import logging
from .base import Agent, AgentResult, AgentTask

LOGGER = logging.get_logger("agent.ds")


class DataScientistAgent(Agent):
    name = "Data Scientist"
    actor = "ds"
    supported_phases = [Phase.INGEST, Phase.EDA, Phase.MODEL]

    def run(self, task: AgentTask, *, context) -> AgentResult:  # type: ignore[override]
        LOGGER.info("Running DS agent for phase %s", task.phase.value)
        if task.phase == Phase.INGEST:
            return self._handle_ingest(task.payload, context)
        if task.phase == Phase.EDA:
            return self._handle_eda(task.payload, context)
        if task.phase == Phase.MODEL:
            return self._handle_model(task.payload, context)
        raise ValueError(f"Unsupported phase for DS: {task.phase}")

    def _handle_ingest(self, payload: Dict, context) -> AgentResult:
        result = ingestion.ingest_and_normalize(
            subject_id=context.config.subject_id,
            source_paths=payload.get("source_paths", context.config.ingestion.source_paths),
            timeline_store=context.timeline,
            artifact_dir=context.config.artifact_path("ingest"),
        )
        artifacts = []
        for name, info in result["artifacts"].items():
            artifact = context.artifacts.register(
                name=f"ingest/{name}",
                path=Path(info["path"]),
                description=info.get("description", name),
                mime_type=info.get("mime_type"),
            )
            artifacts.append(artifact.ref)
        context.cache["qc_report"] = result["qc_report"]
        summary = result.get("summary", "Ingestion completed")
        return AgentResult(
            summary=summary,
            artifacts=artifacts,
            preview=result.get("preview"),
            metadata={"qc_report": result["qc_report"]},
            next_actions=["Proceed to exploratory analysis"],
        )

    def _handle_eda(self, payload: Dict, context) -> AgentResult:
        result = eda.run_eda(
            timeline=context.timeline,
            artifact_dir=context.config.artifact_path("eda"),
            preferences=context.config.preferences,
        )
        artifacts = []
        for name, info in result["artifacts"].items():
            artifact = context.artifacts.register(
                name=f"eda/{name}",
                path=Path(info["path"]),
                description=info.get("description", name),
                mime_type=info.get("mime_type"),
            )
            artifacts.append(artifact.ref)
        context.cache["eda_summary"] = result["summary_table"]
        return AgentResult(
            summary=result["summary"],
            artifacts=artifacts,
            preview=result.get("preview"),
            metadata={"eda_highlights": result.get("highlights", [])},
            next_actions=["Run exposure-outcome modeling"],
        )

    def _handle_model(self, payload: Dict, context) -> AgentResult:
        result = modeling.run_exposure_outcome_models(
            timeline=context.timeline,
            artifact_dir=context.config.artifact_path("modeling"),
            preferences=context.config.preferences,
        )
        artifacts = []
        for name, info in result["artifacts"].items():
            artifact = context.artifacts.register(
                name=f"model/{name}",
                path=Path(info["path"]),
                description=info.get("description", name),
                mime_type=info.get("mime_type"),
            )
            artifacts.append(artifact.ref)
        context.cache["model_results"] = result["results_table"]
        return AgentResult(
            summary=result["summary"],
            artifacts=artifacts,
            preview=result.get("preview"),
            metadata={"model_notes": result.get("notes")},
            next_actions=["Route to medical interpretation"],
        )
