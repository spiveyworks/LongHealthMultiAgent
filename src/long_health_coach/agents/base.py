from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.state_machine import Phase

if TYPE_CHECKING:
    from ..core.context import RunContext


@dataclass
class AgentTask:
    phase: Phase
    objective: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    summary: str
    artifacts: List[str]
    preview: Optional[str] = None
    next_actions: List[str] = field(default_factory=list)
    hitl_required: bool = False
    hitl_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    name: str = "agent"
    actor: str = "system"
    supported_phases: List[Phase] = []

    def handles(self, phase: Phase) -> bool:
        return phase in self.supported_phases

    def run(self, task: AgentTask, *, context: RunContext) -> AgentResult:  # type: ignore[name-defined]
        raise NotImplementedError
