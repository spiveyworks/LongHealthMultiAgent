from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Hypothesis:
    hypothesis_id: str
    description: str
    prior: float
    posterior: float
    status: str = "active"
    last_update: str = "initial"
    notes: str = ""


@dataclass
class HypothesisRegistry:
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)

    def register(self, hypothesis: Hypothesis) -> None:
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis

    def update(self, hypothesis_id: str, *, posterior: float, notes: str, last_update: str) -> None:
        if hypothesis_id not in self.hypotheses:
            raise KeyError(hypothesis_id)
        hyp = self.hypotheses[hypothesis_id]
        hyp.posterior = posterior
        hyp.notes = notes
        hyp.last_update = last_update

    def mark_inactive(self, hypothesis_id: str) -> None:
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = "inactive"

    def summary(self) -> List[dict]:
        return [
            {
                "hypothesis_id": h.hypothesis_id,
                "description": h.description,
                "prior": h.prior,
                "posterior": h.posterior,
                "status": h.status,
                "last_update": h.last_update,
                "notes": h.notes,
            }
            for h in self.hypotheses.values()
        ]
