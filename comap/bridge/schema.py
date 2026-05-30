from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
}

ACTION_PATTERN = re.compile(r"Action:\s*(.*)", re.DOTALL)
GOAL_PATTERN = re.compile(r"Your task is to:\s*(.*)", re.IGNORECASE)


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role")
    if role is None:
        role = ROLE_MAP.get(message.get("from"), message.get("from", "user"))
    content = str(message.get("content") or message.get("value") or "").strip()
    return {"role": role, "content": content}


def normalize_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [normalize_message(message) for message in conversations]


def extract_goal(text: str) -> str:
    match = GOAL_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_action(action_text: str) -> str:
    match = ACTION_PATTERN.search(action_text.strip())
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())
    return re.sub(r"\s+", " ", action_text.strip())


def strip_observation_prefix(observation: str) -> str:
    if observation.startswith("Observation:"):
        return observation.split("Observation:", 1)[1].strip()
    return observation.strip()


def messages_to_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


def build_episode_id(task_name: str, task_id: str, game_file: str | None) -> str:
    if game_file:
        sanitized = game_file.replace("\\", "/").strip("/").replace("/", "__")
        return f"{task_name}__{sanitized}"
    return f"{task_name}__task_{task_id}"


@dataclass
class RolloutStepRecord:
    task_name: str
    task_id: str
    episode_id: str
    round_id: int
    step_id: int
    goal: str
    history_messages: list[dict[str, str]]
    history_text: str
    observation: str
    action: str
    action_text: str
    reward: float
    done: bool
    success: bool
    next_observation: str
    feedback_text: str
    policy_ckpt: str
    trajectory_source: str
    terminate_reason: str | None = None
    game_file: str | None = None
    output_file: str | None = None
    flow_mode: str | None = None
    draft_response: str | None = None
    draft_action: str | None = None
    draft_action_prob: float | None = None
    pred_next_state_student: str | None = None
    pred_next_state_teacher: str | None = None
    reflect_response: str | None = None
    revised_action: str | None = None
    reflect_action_prob: float | None = None
    revise_triggered: bool | None = None
    used_revised_action: bool | None = None
    wm_consistency: float | None = None
    student_world_model_path: str | None = None
    teacher_world_model_path: str | None = None
    imagined_next_state: str | None = None
    real_next_state: str | None = None
    wm_teacher_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_preamble_length(messages: list[dict[str, str]], num_steps: int) -> int:
    preamble_len = len(messages) - (2 * max(num_steps, 0))
    if preamble_len < 0:
        raise ValueError(
            f"Conversation length {len(messages)} is inconsistent with num_steps={num_steps}."
        )
    return preamble_len


def split_episode_messages(
    messages: list[dict[str, str]],
    num_steps: int,
) -> tuple[list[dict[str, str]], list[tuple[dict[str, str], dict[str, str]]]]:
    preamble_len = derive_preamble_length(messages, num_steps)
    preamble = messages[:preamble_len]
    step_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    cursor = preamble_len
    for _ in range(num_steps):
        if cursor + 1 >= len(messages):
            break
        step_pairs.append((messages[cursor], messages[cursor + 1]))
        cursor += 2
    return preamble, step_pairs


def output_file_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem
