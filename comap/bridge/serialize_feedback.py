from __future__ import annotations

from typing import Iterable

from .schema import strip_observation_prefix


INVALID_HINTS: tuple[str, ...] = (
    "error input",
    "can't",
    "cannot",
    "nothing happens",
    "not open",
    "not found",
    "there is no",
    "you need",
)


def _detect_tags(text: str, *, done: bool, success: bool) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    if success:
        tags.append("terminal_success")
    elif done:
        tags.append("terminal_failure")

    if any(hint in lowered for hint in INVALID_HINTS):
        tags.append("invalid_or_unproductive_action")
    elif "you pick up" in lowered or "you put" in lowered or "you open" in lowered:
        tags.append("state_changed")
    else:
        tags.append("state_observation")
    return tags


def concise_env_feedback(
    *,
    observation: str,
    next_observation: str,
    reward: float,
    done: bool,
    success: bool,
    extra_notes: Iterable[str] | None = None,
) -> str:
    next_obs = strip_observation_prefix(next_observation)
    notes = list(extra_notes or [])
    tags = _detect_tags(next_obs, done=done, success=success)
    if reward:
        notes.append(f"reward={reward}")

    lines = [
        "Environment feedback summary:",
        f"- outcome_tags: {', '.join(tags)}",
        f"- next_observation: {next_obs}",
    ]
    if notes:
        lines.append(f"- notes: {'; '.join(notes)}")
    return "\n".join(lines)
