from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

TIMELINE_COLUMNS = [
    "subject_id",
    "measurement_date",
    "variable_name",
    "value",
    "unit",
    "source",
    "metadata",
]


@dataclass
class TimelineStore:
    subject_id: str
    data: pd.DataFrame

    @classmethod
    def empty(cls, subject_id: str) -> "TimelineStore":
        frame = pd.DataFrame(columns=TIMELINE_COLUMNS)
        frame["measurement_date"] = pd.to_datetime(frame["measurement_date"])
        return cls(subject_id=subject_id, data=frame)

    def append(self, records: Iterable[dict]) -> None:
        incoming = pd.DataFrame(list(records))
        if incoming.empty:
            return
        missing = set(TIMELINE_COLUMNS) - set(incoming.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        incoming["measurement_date"] = pd.to_datetime(incoming["measurement_date"], utc=True)
        incoming["subject_id"] = incoming["subject_id"].fillna(self.subject_id)
        # maintain consistent order and types
        incoming = incoming[TIMELINE_COLUMNS]
        self.data = pd.concat([self.data, incoming], ignore_index=True)
        self.data.sort_values("measurement_date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(path)
        return path

    @classmethod
    def load(cls, subject_id: str, path: Path) -> "TimelineStore":
        if not path.exists():
            return cls.empty(subject_id)
        frame = pd.read_parquet(path)
        frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], utc=True)
        return cls(subject_id=subject_id, data=frame)

    def filter_variables(self, names: Optional[List[str]] = None) -> pd.DataFrame:
        if not names:
            return self.data.copy()
        return self.data[self.data["variable_name"].isin(names)].copy()

    def latest_snapshot(self, days: int = 30) -> pd.DataFrame:
        if self.data.empty:
            return self.data.copy()
        cutoff = self.data["measurement_date"].max() - pd.Timedelta(days=days)
        return self.data[self.data["measurement_date"] >= cutoff].copy()
