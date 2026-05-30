from __future__ import annotations

import re
from typing import Any


ACTION_PATTERN = re.compile(r"Action:\s*(.*)", re.DOTALL)


def _normalize_action(text: str) -> str:
    text = (text or "").strip()
    match = ACTION_PATTERN.search(text)
    if match:
        text = match.group(1)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if data_source != "alfworld_step":
        raise ValueError(f"Unsupported reward data source: {data_source}")

    extra_info = extra_info or {}
    predicted_action = _normalize_action(solution_str)
    target_action = _normalize_action(ground_truth)

    if not predicted_action:
        feedback = "Your response must include an `Action:` line that proposes the next environment action."
        if extra_info.get("feedback_text"):
            feedback = f"{feedback}\n\n{extra_info['feedback_text']}"
        return {
            "score": 0.0,
            "feedback": feedback,
            "pred_action": predicted_action,
            "target_action": target_action,
            "matched": False,
            "format_error": True,
        }

    matched = predicted_action == target_action
    feedback = "" if matched else extra_info.get(
        "feedback_text",
        "The proposed action does not match the target transition recorded for this state.",
    )
    return {
        "score": 1.0 if matched else 0.0,
        "feedback": feedback,
        "pred_action": predicted_action,
        "target_action": target_action,
        "matched": matched,
        "format_error": False,
    }
