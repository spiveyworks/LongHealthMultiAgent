from __future__ import annotations

from pathlib import Path

from ..coaching import planner
from ..core.state_machine import Phase
from ..utils import logging
from .base import Agent, AgentResult, AgentTask

LOGGER = logging.get_logger("agent.hlc")


class HealthLongevityCoachAgent(Agent):
    name = "Health & Longevity Coach"
    actor = "hlc"
    supported_phases = [Phase.COACHING]

    def run(self, task: AgentTask, *, context) -> AgentResult:  # type: ignore[override]
        LOGGER.info("Generating longevity coaching plan")
        plan_path = context.config.artifact_path("coaching") / "longevity_plan.json"
        plan = planner.build_longevity_plan(
            timeline=context.timeline,
            eda_summary=context.cache.get("eda_summary"),
            model_results=context.cache.get("model_results"),
            medical_context=context.cache.get("medical_context"),
            preferences=context.config.preferences,
            artifact_path=plan_path,
        )
        artifact = context.artifacts.register(
            name="coaching/longevity_plan",
            path=plan_path,
            description="Weekly longevity coaching plan",
            mime_type="application/json",
        )
        context.cache["longevity_plan"] = plan_path
        summary = plan["summary"]
        preview = summary
        return AgentResult(
            summary=summary,
            artifacts=[artifact.ref],
            preview=preview,
            metadata={"habits": plan["habits"], "prevention": plan["prevention"]},
            next_actions=["Refresh hypothesis registry"],
        )
