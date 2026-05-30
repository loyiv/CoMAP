from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bridge.schema import RolloutStepRecord, strip_observation_prefix
from common import read_jsonl


PROMPT_TEMPLATE = """You are a textual world model for an interactive ALFWorld environment. Given the current textual state and an action, predict the next textual state after executing the action.

Input:
Current state:
Task goal:
{goal}

Interaction history:
{history}

Action:
{action}

Instruction: Predict only the one-step next state caused by the given action. Preserve unchanged facts and update only the parts directly affected by the action. Do not generate a new action, plan, explanation, or multi-step rollout.

Output format:
Predicted next state:
"""


def build_world_model_prompt(record: RolloutStepRecord) -> str:
    return PROMPT_TEMPLATE.format(
        goal=record.goal,
        history=record.history_text,
        action=record.action,
    )


@dataclass
class WorldModelExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    teacher_labels: list[int]
    metadata: dict


class WorldModelDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        *,
        max_prompt_length: int = 2048,
        max_target_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        raw_records = [RolloutStepRecord(**record) for record in read_jsonl(jsonl_path)]
        self.examples = [self._encode_record(record) for record in raw_records]

    def _encode_record(self, record: RolloutStepRecord) -> WorldModelExample:
        prompt = build_world_model_prompt(record)
        target = strip_observation_prefix(record.next_observation)
        teacher_target = strip_observation_prefix(record.wm_teacher_hint or "")

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=self.max_prompt_length)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=self.max_target_length)
        teacher_target_ids = (
            self.tokenizer.encode(
                teacher_target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_length,
            )
            if teacher_target
            else []
        )

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            target_ids = target_ids + [eos_token_id]
            if teacher_target_ids:
                teacher_target_ids = teacher_target_ids + [eos_token_id]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + target_ids
        if teacher_target_ids:
            teacher_labels = ([-100] * len(prompt_ids)) + teacher_target_ids
        else:
            teacher_labels = [-100] * len(input_ids)
        if len(teacher_labels) < len(input_ids):
            teacher_labels = teacher_labels + ([-100] * (len(input_ids) - len(teacher_labels)))
        elif len(teacher_labels) > len(input_ids):
            teacher_labels = teacher_labels[: len(input_ids)]

        return WorldModelExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            teacher_labels=teacher_labels,
            metadata={
                "episode_id": record.episode_id,
                "step_id": record.step_id,
                "goal": record.goal,
                "action": record.action,
                "target": target,
                "teacher_target": teacher_target or None,
            },
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict]:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
            "teacher_labels": torch.tensor(example.teacher_labels, dtype=torch.long),
            "metadata": example.metadata,
        }


class WorldModelCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list[dict]]:
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        teacher_labels = [item["teacher_labels"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        padded_teacher_labels = torch.nn.utils.rnn.pad_sequence(
            teacher_labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
            "teacher_labels": padded_teacher_labels,
            "metadata": metadata,
        }
