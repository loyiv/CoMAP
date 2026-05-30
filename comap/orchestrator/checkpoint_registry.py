from __future__ import annotations

from dataclasses import dataclass

from common import ProjectLayout


@dataclass
class CheckpointRegistry:
    layout: ProjectLayout
    base_policy_model: str | None
    base_world_model: str | None
    base_teacher_world_model: str | None = None

    def resolve_policy_input(self, round_id: int, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        if round_id == 0:
            if not self.base_policy_model:
                raise ValueError("Round 0 requires a base policy model path.")
            return self.base_policy_model
        return str(self.layout.policy_checkpoint_dir(round_id))

    def resolve_world_model_input(
        self,
        round_id: int,
        explicit: str | None = None,
        *,
        allow_policy_fallback: bool = True,
    ) -> str | None:
        if explicit:
            return explicit
        if round_id == 0:
            if self.base_world_model:
                return self.base_world_model
            if allow_policy_fallback:
                return self.resolve_policy_input(round_id)
            return None
        return str(self.layout.world_model_checkpoint_dir(round_id))

    def resolve_teacher_world_model_input(self, round_id: int, explicit: str | None = None) -> str | None:
        if explicit:
            return explicit
        if round_id == 0:
            return self.base_teacher_world_model or self.base_world_model
        return str(self.layout.teacher_world_model_checkpoint_dir(round_id))

    def next_policy_output(self, round_id: int) -> str:
        return str(self.layout.policy_checkpoint_dir(round_id + 1))

    def next_world_model_output(self, round_id: int) -> str:
        return str(self.layout.world_model_checkpoint_dir(round_id + 1))

    def next_teacher_world_model_output(self, round_id: int) -> str:
        return str(self.layout.teacher_world_model_checkpoint_dir(round_id + 1))
