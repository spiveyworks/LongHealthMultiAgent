from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .config import RunConfig
from .events import EventLogger
from .state_machine import StateMachine
from ..data.timeline import TimelineStore
from ..hypothesis.registry import HypothesisRegistry
from ..utils.artifacts import ArtifactStore


@dataclass
class RunContext:
    config: RunConfig
    state: StateMachine
    timeline: TimelineStore
    events: EventLogger
    artifacts: ArtifactStore
    hypotheses: HypothesisRegistry
    cache: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        state_snapshot = self.state.snapshot()
        return {
            "run_id": self.config.run_id,
            "phase": state_snapshot.phase.value,
            "paused": state_snapshot.paused,
            "goal": state_snapshot.goal,
            "interrupt_reason": state_snapshot.interrupt_reason,
            "timeline_rows": int(self.timeline.data.shape[0]) if hasattr(self.timeline, "data") else 0,
            "hypotheses": self.hypotheses.summary(),
            "artifacts": list(self.artifacts.as_dict().keys()),
        }
