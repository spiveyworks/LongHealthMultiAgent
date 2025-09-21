from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Phase(str, Enum):
    """Workflow phases for the longitudinal health coaching system."""

    INGEST = "ingest"
    EDA = "eda"
    MODEL = "model"
    INTERPRET = "interpret"
    COACHING = "coaching"
    HYPOTHESIS = "hypothesis"
    BRIEF = "brief"


PHASE_ORDER: List[Phase] = [
    Phase.INGEST,
    Phase.EDA,
    Phase.MODEL,
    Phase.INTERPRET,
    Phase.COACHING,
    Phase.HYPOTHESIS,
    Phase.BRIEF,
]


@dataclass
class StateSnapshot:
    phase: Phase
    paused: bool
    history: List[Phase]
    goal: str
    interrupt_reason: Optional[str]


@dataclass
class StateMachine:
    """Small reversible state machine that covers the required workflow phases."""

    phase: Phase = Phase.INGEST
    goal: str = "auto-pilot"
    history: List[Phase] = field(default_factory=list)
    paused: bool = False
    interrupt_reason: Optional[str] = None

    def advance(self) -> Phase:
        if self.paused:
            raise RuntimeError("Cannot advance while paused")
        current_index = PHASE_ORDER.index(self.phase)
        if current_index >= len(PHASE_ORDER) - 1:
            raise RuntimeError("Already at final phase; cannot advance")
        self.history.append(self.phase)
        self.phase = PHASE_ORDER[current_index + 1]
        return self.phase

    def rewind(self) -> Phase:
        if not self.history:
            raise RuntimeError("No previous phase to rewind to")
        if self.paused:
            raise RuntimeError("Cannot rewind while paused")
        self.phase = self.history.pop()
        return self.phase

    def jump_to(self, phase: Phase) -> Phase:
        if self.paused:
            raise RuntimeError("Cannot jump while paused")
        if phase not in PHASE_ORDER:
            raise ValueError(f"Unknown phase: {phase}")
        if phase == self.phase:
            return phase
        self.history.append(self.phase)
        self.phase = phase
        return self.phase

    def pause(self, reason: Optional[str] = None) -> None:
        self.paused = True
        self.interrupt_reason = reason

    def resume(self) -> None:
        self.paused = False
        self.interrupt_reason = None

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            phase=self.phase,
            paused=self.paused,
            history=list(self.history),
            goal=self.goal,
            interrupt_reason=self.interrupt_reason,
        )

    def reset(self) -> None:
        self.phase = Phase.INGEST
        self.history.clear()
        self.paused = False
        self.interrupt_reason = None
