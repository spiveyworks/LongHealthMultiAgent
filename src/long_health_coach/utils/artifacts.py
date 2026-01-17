from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Artifact:
    ref: str
    path: Path
    description: str
    mime_type: Optional[str] = None


class ArtifactStore:
    """Tracks artifacts generated during a run and produces stable references."""

    def __init__(self, run_id: str, root: Path) -> None:
        self.run_id = run_id
        self.root = root
        self._artifacts: Dict[str, Artifact] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, *, name: str, path: Path, description: str, mime_type: Optional[str] = None) -> Artifact:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
        ref = f"run:{self.run_id}/{name}:{digest}"
        artifact = Artifact(ref=ref, path=path, description=description, mime_type=mime_type)
        self._artifacts[ref] = artifact
        return artifact

    def get(self, ref: str) -> Artifact:
        return self._artifacts[ref]

    def as_dict(self) -> Dict[str, Artifact]:
        return dict(self._artifacts)
