from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..data.timeline import TimelineStore


@dataclass
class HabitCard:
    habit_id: str
    title: str
    trigger: str
    behavior: str
    min_dose: str
    environment: str
    tracking: str
    expected_effect: Dict[str, object]
    category: str

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.setdefault("expected_effect", {})
        return payload


RANDOM_SEED = 2024


def build_longevity_plan(
    timeline: TimelineStore,
    *,
    eda_summary: Optional[Path],
    model_results: Optional[Path],
    medical_context: Optional[Path],
    preferences: Dict[str, object],
    artifact_path: Path,
) -> Dict[str, object]:
    random.seed(RANDOM_SEED)
    eda_info = _load_json_lines(eda_summary)
    model_info = _load_json_lines(model_results)
    med_info = _load_json(medical_context)

    cards = _select_habits(timeline, eda_info, model_info, preferences)
    prevention = _preventive_prompts(med_info)

    plan = {
        "summary": _plan_summary(cards, model_info),
        "habits": [card.to_dict() for card in cards],
        "prevention": prevention,
        "check_ins": "weekly",
        "supplement_log_prompt": "Maintain observational supplement log and review with clinician.",
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(plan, indent=2))
    return plan


def _select_habits(
    timeline: TimelineStore,
    eda_info: Optional[pd.DataFrame],
    model_info: Optional[pd.DataFrame],
    preferences: Dict[str, object],
) -> List[HabitCard]:
    categories = ["sleep", "activity", "nutrition", "stress"]
    cards: List[HabitCard] = []
    chosen_categories = set()

    def make_card(category: str, idx: int, title: str, trigger: str, behavior: str,
                  min_dose: str, environment: str, tracking: str, expected_effect: Dict[str, object]) -> HabitCard:
        habit_id = f"{category.upper()}-{idx:02d}"
        return HabitCard(
            habit_id=habit_id,
            title=title,
            trigger=trigger,
            behavior=behavior,
            min_dose=min_dose,
            environment=environment,
            tracking=tracking,
            expected_effect=expected_effect,
            category=category,
        )

    available_metrics = set(timeline.data["variable_name"].unique()) if not timeline.data.empty else set()

    if "sleep_duration" in available_metrics or "sleep_hours" in available_metrics:
        cards.append(
            make_card(
                "sleep",
                1,
                "Consistent wake window",
                "Alarm at 6:30 am",
                "Out of bed within 5 minutes and get daylight for 5 minutes",
                "5 days/week",
                "Phone across room; blinds open",
                "days_adhered_per_week",
                {"direction": "improve", "targets": ["sleep_regular"]},
            )
        )
        chosen_categories.add("sleep")

    if "exercise_minutes" in available_metrics or "steps" in available_metrics:
        cards.append(
            make_card(
                "activity",
                1,
                "Zone 2 sessions",
                "Calendar block Mon/Wed",
                "Perform 30 minutes moderate intensity cardio",
                "2 sessions/week",
                "Shoes packed night prior",
                "sessions_completed",
                {"direction": "improve", "targets": ["cardiorespiratory_fitness"]},
            )
        )
        chosen_categories.add("activity")

    if "protein_grams" in available_metrics or "fiber_grams" in available_metrics:
        cards.append(
            make_card(
                "nutrition",
                1,
                "Protein-forward lunch",
                "Lunch prep reminder at 11:30",
                "Build plate with ≥30g protein and colorful vegetables",
                "4 days/week",
                "Prep groceries on Sunday",
                "meals_meeting_goal",
                {"direction": "improve", "targets": ["nutrition_quality"]},
            )
        )
        chosen_categories.add("nutrition")

    if "stress_score" in available_metrics:
        cards.append(
            make_card(
                "stress",
                1,
                "Post-meeting breathwork",
                "Calendar alert at end of meetings",
                "3 minute box breathing",
                "5 times/week",
                "Breath app pinned on phone",
                "sessions_logged",
                {"direction": "reduce", "targets": ["stress_score"]},
            )
        )
        chosen_categories.add("stress")

    while len(cards) < 3:
        category = (set(categories) - chosen_categories) or set(categories)
        category_choice = sorted(category)[0]
        cards.append(
            make_card(
                category_choice,
                len(cards) + 1,
                "Outdoor light exposure",
                "Morning alarm",
                "Spend 10 minutes outside within an hour of waking",
                "5 days/week",
                "Keep walking shoes near door",
                "days_adhered_per_week",
                {"direction": "improve", "targets": ["sleep_regular", "circadian_alignment"]},
            )
        )
        chosen_categories.add(category_choice)

    return cards[:4]


def _preventive_prompts(med_info: Optional[Dict[str, object]]) -> List[str]:
    prompts = [
        "Discuss age-appropriate screenings (e.g., colorectal, cardiovascular) with your clinician.",
        "Confirm immunizations are up to date, including seasonal vaccines.",
    ]
    if med_info and med_info.get("questions"):
        prompts.extend(med_info["questions"])
    return prompts


def _plan_summary(cards: List[HabitCard], model_info: Optional[pd.DataFrame]) -> str:
    focus_categories = sorted({card.category for card in cards})
    summary = f"Weekly longevity focus on: {', '.join(focus_categories)}."
    if model_info is not None and not model_info.empty:
        top = model_info.sort_values(by="p_value").iloc[0]
        summary += (
            f" Prioritize {top['exposure']} given its association with {top['outcome']} (beta={top['beta']:.3f})."
        )
    return summary


def _load_json(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def _load_json_lines(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not Path(path).exists():
        return None
    return pd.read_json(path)
