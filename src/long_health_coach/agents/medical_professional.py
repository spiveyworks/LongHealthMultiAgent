from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..core.state_machine import Phase
from ..utils import logging
from .base import Agent, AgentResult, AgentTask

LOGGER = logging.get_logger("agent.medpro")

SAFETY_BOILERPLATE = (
    "This is educational information, not medical advice. Talk with a qualified clinician before making changes."
)


class MedicalProfessionalAgent(Agent):
    name = "Medical Professional"
    actor = "medpro"
    supported_phases = [Phase.INTERPRET]

    def run(self, task: AgentTask, *, context) -> AgentResult:  # type: ignore[override]
        LOGGER.info("Running MedPro interpretation")
        artifact_dir = context.config.artifact_path("interpretation")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_results_path = context.cache.get("model_results")
        eda_summary_path = context.cache.get("eda_summary")

        notes = self._build_notes(model_results_path, eda_summary_path)
        output_path = artifact_dir / "medical_context.json"
        output_path.write_text(json.dumps(notes, indent=2))

        artifact = context.artifacts.register(
            name="interpretation/medical_context",
            path=output_path,
            description="Medical professional context notes",
            mime_type="application/json",
        )
        context.cache["medical_context"] = output_path

        summary = notes.get("summary", "Medical context generated")
        preview = notes.get("preview")
        next_actions = ["Generate coaching plan"]
        return AgentResult(
            summary=f"{summary} {SAFETY_BOILERPLATE}",
            artifacts=[artifact.ref],
            preview=preview,
            metadata={"clinician_questions": notes.get("questions", [])},
            next_actions=next_actions,
        )

    def _build_notes(self, model_results_path: Path, eda_summary_path: Path) -> Dict:
        questions: List[str] = []
        insights: List[str] = []
        preview = None

        if model_results_path and Path(model_results_path).exists():
            model_df = pd.read_json(model_results_path)
            if not model_df.empty:
                top_effect = model_df.sort_values(by="p_value").iloc[0]
                effect_summary = (
                    f"Observed association: {top_effect['outcome']} vs {top_effect['exposure']} "
                    f"beta={top_effect['beta']:.3f} (p={top_effect['p_value']:.3g})"
                )
                insights.append(effect_summary)
                questions.append(
                    f"Discuss with clinician whether the relationship between {top_effect['exposure']} "
                    f"and {top_effect['outcome']} is clinically meaningful."
                )
                preview = effect_summary

        if eda_summary_path and Path(eda_summary_path).exists():
            eda_df = pd.read_json(eda_summary_path)
            if not eda_df.empty:
                trend = eda_df.iloc[0]
                insights.append(
                    f"Trend noted: {trend['variable_name']} {trend['trend_direction']} over the recent window."
                )

        if not insights:
            insights.append("No strong clinical patterns detected; continue monitoring key metrics.")

        return {
            "summary": insights[0],
            "preview": preview or insights[0],
            "insights": insights,
            "questions": questions,
            "safety": SAFETY_BOILERPLATE,
        }
