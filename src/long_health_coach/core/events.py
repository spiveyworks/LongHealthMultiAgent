from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .state_machine import Phase


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class EventIO:
    refs: List[str]
    preview: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"refs": self.refs}
        if self.preview is not None:
            payload["preview"] = self.preview
        return payload


@dataclass
class Event:
    event_id: str
    timestamp: str
    actor: str
    phase: Phase
    inputs: EventIO
    outputs: EventIO
    next_actions: List[str]
    hitl_required: bool = False
    hitl_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["inputs"] = self.inputs.to_dict()
        payload["outputs"] = self.outputs.to_dict()
        return payload


class EventLogger:
    """In-memory event log with optional persistence to disk."""

    def __init__(self) -> None:
        self._events: List[Event] = []

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def log(self,
            *,
            actor: str,
            phase: Phase,
            inputs: Optional[EventIO] = None,
            outputs: Optional[EventIO] = None,
            next_actions: Optional[List[str]] = None,
            hitl_required: bool = False,
            hitl_reason: Optional[str] = None,
            preview: Optional[str] = None) -> Event:
        event = Event(
            event_id=str(uuid.uuid4()),
            timestamp=_utcnow(),
            actor=actor,
            phase=phase,
            inputs=inputs or EventIO(refs=[]),
            outputs=outputs or EventIO(refs=[], preview=preview),
            next_actions=next_actions or [],
            hitl_required=hitl_required,
            hitl_reason=hitl_reason,
        )
        self._events.append(event)
        return event

    def dump_json(self, path: Path) -> None:
        payload = [event.to_dict() for event in self._events]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def clear(self) -> None:
        self._events.clear()
