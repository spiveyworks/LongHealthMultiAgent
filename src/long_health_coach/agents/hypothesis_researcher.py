from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..core.state_machine import Phase
from ..hypothesis.registry import Hypothesis, HypothesisRegistry
from ..utils import logging
from .base import Agent, AgentResult, AgentTask

LOGGER = logging.get_logger("agent.hdr")


class HypothesisDrivenResearcherAgent(Agent):
    name = "Hypothesis Researcher"
    actor = "hdr"
    supported_phases = [Phase.HYPOTHESIS]

    def run(self, task: AgentTask, *, context) -> AgentResult:  # type: ignore[override]
        LOGGER.info("Updating hypothesis registry")
        registry: HypothesisRegistry = context.hypotheses
        model_path = context.cache.get("model_results")
        updates = self._synthesize_updates(registry, model_path)

        artifact_path = context.config.artifact_path("hypotheses", "hypotheses.json")
        artifact_path.write_text(json.dumps(updates, indent=2))
        artifact = context.artifacts.register(
            name="hypothesis/registry",
            path=artifact_path,
            description="Updated hypothesis registry",
            mime_type="application/json",
        )
        context.cache["hypothesis_registry"] = artifact_path

        summary = updates["summary"]
        return AgentResult(
            summary=summary,
            artifacts=[artifact.ref],
            preview=summary,
            metadata={"hypotheses": updates["hypotheses"]},
            next_actions=["Compile weekly plan"],
        )

    def _synthesize_updates(self, registry: HypothesisRegistry, model_path: Path | None) -> dict:
        if registry.hypotheses == {}:
            registry.register(
                Hypothesis(
                    hypothesis_id="H1",
                    description="Improving aerobic activity will reduce Dunedin Pace of Aging",
                    prior=0.4,
                    posterior=0.4,
                )
            )
        if model_path and Path(model_path).exists():
            df = pd.read_json(model_path)
            for _, row in df.iterrows():
                hyp_id = f"H_{row['outcome']}_{row['exposure']}"
                posterior = max(0.05, min(0.95, 0.5 + -0.5 * row["beta"]))
                description = f"Relationship between {row['exposure']} and {row['outcome']}"
                if hyp_id not in registry.hypotheses:
                    registry.register(
                        Hypothesis(
                            hypothesis_id=hyp_id,
                            description=description,
                            prior=0.5,
                            posterior=posterior,
                            notes="Initialized from latest modeling run",
                            last_update="modeling",
                        )
                    )
                else:
                    registry.update(
                        hyp_id,
                        posterior=posterior,
                        notes=f"Updated from beta={row['beta']:.3f}",
                        last_update="modeling",
                    )
        summary = "Hypothesis registry refreshed with latest modeling evidence."
        return {"summary": summary, "hypotheses": registry.summary()}
