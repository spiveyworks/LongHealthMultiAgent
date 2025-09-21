from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IngestionConfig:
    source_paths: List[Path] = field(default_factory=list)
    timeline_table: Path = Path("artifacts/timeline.parquet")
    cache_dir: Path = Path("artifacts/cache")


@dataclass
class AutonomyConfig:
    mode: str = "auto-pilot"  # auto-pilot | assist | pause
    confirm_each_phase: bool = False


@dataclass
class RunConfig:
    subject_id: str = "default"
    run_id: str = "local-run"
    seed: int = 202401
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    artifact_root: Path = Path("artifacts")
    enable_artifact_signing: bool = False
    preferences: Dict[str, object] = field(default_factory=dict)

    def artifact_path(self, *parts: str) -> Path:
        path = self.artifact_root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
