from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import IngestionConfig, RunConfig
from .core.context import RunContext
from .core.events import EventLogger
from .core.orchestrator import Orchestrator, default_agents
from .core.state_machine import StateMachine
from .data.timeline import TimelineStore
from .hypothesis.registry import HypothesisRegistry
from .utils.artifacts import ArtifactStore
from .utils.logging import get_logger

LOGGER = get_logger("main")


def build_context(config: RunConfig) -> RunContext:
    timeline = TimelineStore.load(config.subject_id, config.ingestion.timeline_table)
    events = EventLogger()
    artifacts = ArtifactStore(config.run_id, config.artifact_root)
    hypotheses = HypothesisRegistry()
    state = StateMachine()
    return RunContext(
        config=config,
        state=state,
        timeline=timeline,
        events=events,
        artifacts=artifacts,
        hypotheses=hypotheses,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Long Health Coach multi-agent workflow")
    parser.add_argument("--subject-id", default="demo", help="Subject identifier")
    parser.add_argument("--run-id", default="local-run", help="Run identifier")
    parser.add_argument("--source", action="append", dest="sources", help="Path(s) to data files", default=[])
    parser.add_argument("--artifact-root", default="artifacts", help="Directory for artifacts")
    parser.add_argument("--timeline", default="artifacts/timeline.parquet", help="Existing timeline file")
    parser.add_argument("--goal", default="Optimize longevity", help="Goal for this orchestrator run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestion_config = IngestionConfig(
        source_paths=[Path(path) for path in args.sources],
        timeline_table=Path(args.timeline),
        cache_dir=Path(args.artifact_root) / "cache",
    )
    config = RunConfig(
        subject_id=args.subject_id,
        run_id=args.run_id,
        ingestion=ingestion_config,
        artifact_root=Path(args.artifact_root),
    )

    context = build_context(config)
    orchestrator = Orchestrator(context, default_agents())
    try:
        result = orchestrator.run(goal=args.goal)
        LOGGER.info("Run complete: %s", result.summary)
    except RuntimeError as exc:
        LOGGER.error("Run interrupted: %s", exc)
        LOGGER.info("State snapshot: %s", orchestrator.snapshot())
    finally:
        events_path = config.artifact_path("events") / "events.json"
        context.events.dump_json(events_path)
        LOGGER.info("Event log written to %s", events_path)


if __name__ == "__main__":
    main()
