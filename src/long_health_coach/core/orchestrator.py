from __future__ import annotations

from typing import Dict, Iterable

from ..agents.base import Agent, AgentResult, AgentTask
from ..agents.project_manager import ProjectManagerAgent
from ..core.context import RunContext
from ..core.events import EventIO
from ..core.state_machine import PHASE_ORDER, Phase
from ..utils import logging

LOGGER = logging.get_logger("orchestrator")


class Orchestrator:
    """Coordinates multi-agent workflow following the defined state machine."""

    def __init__(self, context: RunContext, agents: Iterable[Agent]) -> None:
        self.context = context
        self.agents: Dict[Phase, Agent] = {}
        for agent in agents:
            for phase in agent.supported_phases:
                if phase in self.agents:
                    raise ValueError(f"Phase {phase} already assigned to agent {self.agents[phase].name}")
                self.agents[phase] = agent
        if Phase.BRIEF not in self.agents:
            self.agents[Phase.BRIEF] = ProjectManagerAgent()

    def run(self, goal: str = "Optimize longevity") -> AgentResult:
        self.context.state.goal = goal
        LOGGER.info("Starting orchestrator goal=%s", goal)
        while True:
            if self.context.state.paused:
                LOGGER.info("State paused; returning snapshot")
                raise RuntimeError("Run paused")

            phase = self.context.state.phase
            LOGGER.info("Processing phase: %s", phase.value)
            agent = self._route(phase)
            task = AgentTask(phase=phase, objective=self._objective(phase), payload=self._payload(phase))
            result = agent.run(task, context=self.context)

            self.context.events.log(
                actor=agent.actor,
                phase=phase,
                inputs=EventIO(refs=self._inputs_for_phase(phase)),
                outputs=EventIO(refs=result.artifacts, preview=result.preview),
                next_actions=result.next_actions,
                hitl_required=result.hitl_required,
                hitl_reason=result.hitl_reason,
            )

            if phase == Phase.BRIEF:
                LOGGER.info("Reached briefing phase; run complete")
                return result

            self.context.state.advance()

    def _route(self, phase: Phase) -> Agent:
        if phase not in self.agents:
            raise KeyError(f"No agent assigned for phase {phase.value}")
        return self.agents[phase]

    def _objective(self, phase: Phase) -> str:
        objectives = {
            Phase.INGEST: "Ingest longitudinal data and run QC",
            Phase.EDA: "Summarize trends and correlations",
            Phase.MODEL: "Model exposure-outcome relationships",
            Phase.INTERPRET: "Provide medical context and clinician-oriented questions",
            Phase.COACHING: "Translate insights into longevity habit plan",
            Phase.HYPOTHESIS: "Update hypothesis registry with latest evidence",
            Phase.BRIEF: "Compile Weekly Health & Longevity Plan",
        }
        return objectives.get(phase, "Execute phase tasks")

    def _payload(self, phase: Phase) -> Dict:
        if phase == Phase.INGEST:
            return {
                "source_paths": [str(path) for path in self.context.config.ingestion.source_paths]
            }
        return {}

    def _inputs_for_phase(self, phase: Phase) -> list[str]:
        if phase == Phase.INGEST:
            return [str(path) for path in self.context.config.ingestion.source_paths]
        return []

    def pause(self, reason: str) -> None:
        self.context.state.pause(reason=reason)

    def resume(self) -> None:
        self.context.state.resume()

    def snapshot(self) -> dict:
        return self.context.snapshot()


def default_agents() -> list[Agent]:
    from ..agents.data_scientist import DataScientistAgent
    from ..agents.health_longevity_coach import HealthLongevityCoachAgent
    from ..agents.hypothesis_researcher import HypothesisDrivenResearcherAgent
    from ..agents.medical_professional import MedicalProfessionalAgent
    from ..agents.project_manager import ProjectManagerAgent

    return [
        DataScientistAgent(),
        MedicalProfessionalAgent(),
        HealthLongevityCoachAgent(),
        HypothesisDrivenResearcherAgent(),
        ProjectManagerAgent(),
    ]
