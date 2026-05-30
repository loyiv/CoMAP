from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    eto_root: Path
    sdpo_root: Path
    verl_root: Path
    bridge_root: Path
    world_model_root: Path
    training_root: Path
    orchestrator_root: Path
    scripts_root: Path
    buffers_root: Path
    checkpoints_root: Path
    outputs_root: Path
    wm_teacher_root: Path

    @classmethod
    def discover(
        cls,
        *,
        root: str | Path | None = None,
        eto_root: str | Path | None = None,
        sdpo_root: str | Path | None = None,
        verl_root: str | Path | None = None,
    ) -> "ProjectLayout":
        project_root = Path(root or os.environ.get("ETO_SDPO_PROJECT_ROOT", _default_root())).expanduser().resolve()
        eto = Path(eto_root or os.environ.get("ETO_ROOT", project_root / "third_party" / "agent-eto")).expanduser().resolve()
        sdpo = Path(sdpo_root or os.environ.get("SDPO_ROOT", project_root / "third_party" / "SDPO")).expanduser().resolve()
        verl = Path(verl_root or os.environ.get("VERL_ROOT", project_root / "third_party" / "verl")).expanduser().resolve()

        return cls(
            root=project_root,
            eto_root=eto,
            sdpo_root=sdpo,
            verl_root=verl,
            bridge_root=project_root / "bridge",
            world_model_root=project_root / "world_model",
            training_root=project_root / "training",
            orchestrator_root=project_root / "orchestrator",
            scripts_root=project_root / "scripts",
            buffers_root=project_root / "buffers",
            checkpoints_root=project_root / "checkpoints",
            outputs_root=project_root / "outputs",
            wm_teacher_root=project_root / "checkpoints" / "wm_teacher",
        )

    def ensure_directories(self) -> None:
        for path in [
            self.bridge_root,
            self.world_model_root,
            self.training_root,
            self.orchestrator_root,
            self.scripts_root,
            self.buffers_root / "wm",
            self.buffers_root / "sdpo",
            self.checkpoints_root / "policy",
            self.checkpoints_root / "wm",
            self.wm_teacher_root,
            self.outputs_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def round_rollout_dir(self, round_id: int) -> Path:
        return self.outputs_root / f"round_{round_id}_rollout"

    def round_eval_dir(self, round_id: int) -> Path:
        return self.outputs_root / f"eval_round_{round_id}"

    def round_wm_buffer(self, round_id: int, split: str = "all") -> Path:
        return self.buffers_root / "wm" / f"round_{round_id}_{split}.jsonl"

    def round_sdpo_dir(self, round_id: int) -> Path:
        return self.buffers_root / "sdpo" / f"round_{round_id}"

    def policy_checkpoint_dir(self, round_id: int) -> Path:
        return self.checkpoints_root / "policy" / f"policy_round_{round_id}"

    def world_model_checkpoint_dir(self, round_id: int) -> Path:
        return self.checkpoints_root / "wm" / f"wm_round_{round_id}"

    def teacher_world_model_checkpoint_dir(self, round_id: int) -> Path:
        return self.wm_teacher_root / f"wm_teacher_round_{round_id}"

    def metrics_path(self, round_id: int) -> Path:
        return self.outputs_root / f"round_{round_id}_metrics.json"

    def as_env(self) -> dict[str, str]:
        return {
            "ETO_SDPO_PROJECT_ROOT": str(self.root),
            "ETO_ROOT": str(self.eto_root),
            "SDPO_ROOT": str(self.sdpo_root),
            "VERL_ROOT": str(self.verl_root),
            "PYTHONPATH": f"{self.root}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
        }

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "eto_root": str(self.eto_root),
            "sdpo_root": str(self.sdpo_root),
            "verl_root": str(self.verl_root),
            "bridge_root": str(self.bridge_root),
            "world_model_root": str(self.world_model_root),
            "training_root": str(self.training_root),
            "orchestrator_root": str(self.orchestrator_root),
            "scripts_root": str(self.scripts_root),
            "buffers_root": str(self.buffers_root),
            "checkpoints_root": str(self.checkpoints_root),
            "outputs_root": str(self.outputs_root),
            "wm_teacher_root": str(self.wm_teacher_root),
        }
