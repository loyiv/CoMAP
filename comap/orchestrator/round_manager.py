from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common import ProjectLayout

from .checkpoint_registry import CheckpointRegistry
from .metrics_logger import MetricsLogger


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RoundArtifacts:
    rollout_output_dir: str
    round_buffer_path: str
    world_model_output_dir: str
    teacher_world_model_output_dir: str
    sdpo_dataset_dir: str
    policy_output_dir: str
    evaluation_output_dir: str


class RoundManager:
    def __init__(self, config_path: str | Path, *, project_root: str | Path | None = None, dry_run: bool = False) -> None:
        self.layout = ProjectLayout.discover(root=project_root)
        self.layout.ensure_directories()
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = _load_yaml(self.config_path)
        self.registry = CheckpointRegistry(
            layout=self.layout,
            base_policy_model=self.config.get("models", {}).get("base_policy_model"),
            base_world_model=self.config.get("world_model", {}).get("base_model_path")
            or self.config.get("models", {}).get("base_world_model"),
            base_teacher_world_model=self.config.get("models", {}).get("base_teacher_world_model"),
        )
        self.dry_run = dry_run

    def _run(self, command: list[str], *, cwd: str | Path | None = None) -> None:
        print("Running command:")
        print(" ".join(map(str, command)))
        if self.dry_run:
            return

        env = os.environ.copy()
        env.update(self.layout.as_env())
        subprocess.run(command, cwd=cwd or self.layout.root, env=env, check=True)

    def _resolve_model_name(self, section_name: str, policy_model_path: str) -> str:
        section_cfg = self.config.get(section_name, {})
        return section_cfg.get("model_name") or Path(policy_model_path).expanduser().resolve().name

    def _resolve_agent_backend(self, section_name: str, student_world_model_path: str | None) -> str:
        section_cfg = self.config.get(section_name, {})
        backend = section_cfg.get("agent_backend", "fastchat")
        if backend == "auto":
            if student_world_model_path:
                return "coevolving_local_hf"
            return self.config.get("pipeline", {}).get("fallback_agent_backend_without_wm", "local_hf")
        if backend == "coevolving_local_hf" and not student_world_model_path:
            fallback_backend = self.config.get("pipeline", {}).get("fallback_agent_backend_without_wm")
            if fallback_backend:
                return fallback_backend
            raise ValueError(f"{section_name}.agent_backend=coevolving_local_hf requires a student world model path.")
        return backend

    def run_rollout(
        self,
        round_id: int,
        policy_model_path: str,
        *,
        student_world_model_path: str | None = None,
        teacher_world_model_path: str | None = None,
    ) -> str:
        rollout_cfg = self.config["rollout"]
        output_dir = self.layout.round_rollout_dir(round_id)
        agent_backend = self._resolve_agent_backend("rollout", student_world_model_path)
        command = [
            str(self.layout.scripts_root / "run_rollout_round.sh"),
            "--project-root",
            str(self.layout.root),
            "--eto-root",
            str(self.layout.eto_root),
            "--conda-env",
            rollout_cfg["conda_env"],
            "--task-name",
            self.config["task"]["name"],
            "--exp-config",
            self.config["task"]["exp_config"],
            "--agent-config",
            self.config["task"]["agent_config"],
            "--agent-backend",
            agent_backend,
            "--model-path",
            policy_model_path,
            "--model-name",
            self._resolve_model_name("rollout", policy_model_path),
            "--round-id",
            str(round_id),
            "--output-dir",
            str(output_dir),
            "--controller-port",
            str(rollout_cfg["controller_port"]),
            "--worker-port",
            str(rollout_cfg["worker_port"]),
            "--wait-seconds",
            str(rollout_cfg["wait_seconds"]),
            "--split",
            rollout_cfg.get("split", "train"),
            "--max-new-tokens",
            str(rollout_cfg.get("max_new_tokens", 256)),
            "--reflect-max-new-tokens",
            str(rollout_cfg.get("reflect_max_new_tokens", rollout_cfg.get("max_new_tokens", 256))),
            "--final-max-new-tokens",
            str(rollout_cfg.get("final_max_new_tokens", 16)),
            "--wm-max-new-tokens",
            str(rollout_cfg.get("wm_max_new_tokens", 128)),
            "--temperature",
            str(rollout_cfg.get("temperature", 0.7)),
            "--top-p",
            str(rollout_cfg.get("top_p", 0.9)),
            "--revise-probability-threshold",
            str(rollout_cfg.get("revise_probability_threshold", 0.5)),
            "--reflection-confidence-threshold",
            str(rollout_cfg.get("reflection_confidence_threshold", 0.6)),
            "--part-num",
            str(rollout_cfg.get("part_num", 1)),
            "--part-idx",
            str(rollout_cfg.get("part_idx", -1)),
        ]
        if rollout_cfg.get("enable_thinking", False):
            command.append("--enable-thinking")
        if rollout_cfg.get("debug", False):
            command.append("--debug")
        if rollout_cfg.get("override", False):
            command.append("--override")
        if agent_backend == "coevolving_local_hf":
            if not student_world_model_path:
                raise ValueError("Co-evolving rollout requires student_world_model_path.")
            command.extend(["--wm-student-model-path", student_world_model_path])
            if rollout_cfg.get("include_teacher_world_model", True) and teacher_world_model_path:
                command.extend(["--wm-teacher-model-path", teacher_world_model_path])
        self._run(command)
        return str(output_dir / "eto_raw")

    def export_round_buffer(self, round_id: int, raw_rollout_dir: str, policy_model_path: str) -> str:
        round_buffer_path = self.layout.round_wm_buffer(round_id, "all")
        command = [
            "python",
            "-m",
            "bridge.eto_to_round_buffer",
            "--project-root",
            str(self.layout.root),
            "--input-dir",
            raw_rollout_dir,
            "--task-name",
            self.config["task"]["name"],
            "--round-id",
            str(round_id),
            "--policy-ckpt",
            policy_model_path,
            "--output-path",
            str(round_buffer_path),
        ]
        self._run(command, cwd=self.layout.root)
        return str(round_buffer_path)

    def train_world_model(
        self,
        round_id: int,
        round_buffer_path: str,
        *,
        current_world_model_path: str | None = None,
        teacher_world_model_path: str | None = None,
        fallback_policy_model_path: str | None = None,
    ) -> str:
        wm_cfg = self.config["world_model"]
        input_model = (
            current_world_model_path
            or wm_cfg.get("base_model_path")
            or fallback_policy_model_path
            or self.registry.resolve_world_model_input(round_id, allow_policy_fallback=True)
        )
        if not input_model:
            raise ValueError("World model training requires either a current WM path or a policy checkpoint fallback.")

        output_dir = self.registry.next_world_model_output(round_id)
        command = [
            str(self.layout.scripts_root / "run_wm_update.sh"),
            "--conda-env",
            wm_cfg["conda_env"],
            "--train-jsonl",
            round_buffer_path,
            "--model-name-or-path",
            input_model,
            "--output-dir",
            output_dir,
            "--num-train-epochs",
            str(wm_cfg["num_train_epochs"]),
            "--per-device-train-batch-size",
            str(wm_cfg["per_device_train_batch_size"]),
            "--per-device-eval-batch-size",
            str(wm_cfg["per_device_eval_batch_size"]),
            "--learning-rate",
            str(wm_cfg["learning_rate"]),
            "--lr-scheduler-type",
            str(wm_cfg.get("lr_scheduler_type", "cosine")),
            "--warmup-ratio",
            str(wm_cfg.get("warmup_ratio", 0.03)),
            "--gradient-accumulation-steps",
            str(wm_cfg.get("gradient_accumulation_steps", 16)),
            "--max-prompt-length",
            str(wm_cfg["max_prompt_length"]),
            "--max-target-length",
            str(wm_cfg["max_target_length"]),
            "--distill-weight",
            str(wm_cfg.get("distill_weight", 0.0)),
            "--project-root",
            str(self.layout.root),
        ]
        if teacher_world_model_path:
            command.extend(["--teacher-model-name-or-path", teacher_world_model_path])
        if wm_cfg.get("use_lora", False):
            command.append("--use-lora")
        if wm_cfg.get("gradient_checkpointing", False):
            command.append("--gradient-checkpointing")
        self._run(command)
        return output_dir

    def update_teacher_world_model(self, round_id: int, student_world_model_output_dir: str) -> str:
        strategy = self.config.get("world_model", {}).get("teacher_update_strategy", "snapshot")
        output_dir = Path(self.registry.next_teacher_world_model_output(round_id)).expanduser().resolve()
        source_dir = Path(student_world_model_output_dir).expanduser().resolve()

        print(f"Updating teacher world model with strategy={strategy}: {source_dir} -> {output_dir}")
        if self.dry_run:
            return str(output_dir)

        if strategy not in {"snapshot", "ema"}:
            raise ValueError(f"Unsupported teacher_update_strategy: {strategy}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)
        if strategy == "ema":
            # The minimal release keeps checkpoint materialization lightweight.
            # Training-time EMA is represented by the teacher checkpoint path;
            # full parameter-wise averaging belongs in the installed training stack.
            from common import write_json

            write_json(
                {
                    "teacher_update_strategy": "ema",
                    "ema_momentum": self.config.get("world_model", {}).get("ema_momentum", 0.99),
                    "source_dir": str(source_dir),
                },
                output_dir / "teacher_update_metadata.json",
            )
        return str(output_dir)

    def build_sdpo_buffer(self, round_id: int, round_buffer_path: str) -> str:
        output_dir = self.layout.round_sdpo_dir(round_id)
        bridge_cfg = self.config["bridge"]
        command = [
            str(self.layout.scripts_root / "run_sdpo_buffer_build.sh"),
            "--conda-env",
            self.config["sdpo"]["conda_env"],
            "--project-root",
            str(self.layout.root),
            "--round-buffer-path",
            round_buffer_path,
            "--output-dir",
            str(output_dir),
            "--episode-filter",
            bridge_cfg["episode_filter"],
            "--val-ratio",
            str(bridge_cfg["val_ratio"]),
            "--policy-dataset-mode",
            bridge_cfg.get("policy_dataset_mode", "transition"),
        ]
        if not bridge_cfg.get("include_expert_anchors", True):
            command.append("--disable-expert-anchors")
        if not bridge_cfg.get("include_draft_distill", True):
            command.append("--disable-draft-distill")
        if not bridge_cfg.get("require_success_for_hq", True):
            command.append("--disable-success-only-hq")
        if not bridge_cfg.get("allow_fallback_to_all_steps", True):
            command.append("--disable-fallback-to-all-steps")
        if bridge_cfg.get("wm_consistency_min") is not None:
            command.extend(["--wm-consistency-min", str(bridge_cfg["wm_consistency_min"])])
        self._run(command)
        return str(output_dir)

    def train_policy(self, round_id: int, sdpo_dataset_dir: str, policy_model_path: str) -> str:
        output_dir = self.registry.next_policy_output(round_id)
        command = [
            str(self.layout.training_root / "launch_sdpo_round.sh"),
            "--project-root",
            str(self.layout.root),
            "--config",
            str(self.config_path),
            "--round-id",
            str(round_id),
            "--dataset-dir",
            sdpo_dataset_dir,
            "--policy-model-path",
            policy_model_path,
            "--output-dir",
            output_dir,
        ]
        self._run(command)
        checkpoint_root = Path(output_dir).expanduser().resolve()
        latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
        if latest_path.exists():
            latest_step = latest_path.read_text(encoding="utf-8").strip()
            actor_dir = checkpoint_root / f"global_step_{latest_step}" / "actor"
            lora_dir = actor_dir / "lora_adapter"
            if (lora_dir / "adapter_model.safetensors").exists():
                return str(lora_dir)
            hf_dir = actor_dir / "huggingface"
            if (hf_dir / "config.json").exists():
                return str(hf_dir)
        return output_dir

    def evaluate_policy(
        self,
        round_id: int,
        policy_model_path: str,
        *,
        student_world_model_path: str | None = None,
        teacher_world_model_path: str | None = None,
    ) -> str:
        eval_cfg = self.config["evaluation"]
        output_dir = self.layout.round_eval_dir(round_id)
        agent_backend = self._resolve_agent_backend("evaluation", student_world_model_path)
        command = [
            str(self.layout.scripts_root / "run_eval_round.sh"),
            "--project-root",
            str(self.layout.root),
            "--eto-root",
            str(self.layout.eto_root),
            "--conda-env",
            eval_cfg["conda_env"],
            "--task-name",
            self.config["task"]["name"],
            "--exp-config",
            self.config["task"]["exp_config"],
            "--agent-config",
            self.config["task"]["agent_config"],
            "--agent-backend",
            agent_backend,
            "--model-path",
            policy_model_path,
            "--model-name",
            self._resolve_model_name("evaluation", policy_model_path),
            "--round-id",
            str(round_id),
            "--output-dir",
            str(output_dir),
            "--controller-port",
            str(eval_cfg["controller_port"]),
            "--worker-port",
            str(eval_cfg["worker_port"]),
            "--wait-seconds",
            str(eval_cfg["wait_seconds"]),
            "--max-new-tokens",
            str(eval_cfg.get("max_new_tokens", 256)),
            "--reflect-max-new-tokens",
            str(eval_cfg.get("reflect_max_new_tokens", eval_cfg.get("max_new_tokens", 256))),
            "--final-max-new-tokens",
            str(eval_cfg.get("final_max_new_tokens", 16)),
            "--wm-max-new-tokens",
            str(eval_cfg.get("wm_max_new_tokens", 128)),
            "--temperature",
            str(eval_cfg.get("temperature", 0.7)),
            "--top-p",
            str(eval_cfg.get("top_p", 0.9)),
            "--revise-probability-threshold",
            str(eval_cfg.get("revise_probability_threshold", 0.5)),
            "--reflection-confidence-threshold",
            str(eval_cfg.get("reflection_confidence_threshold", 0.6)),
            "--part-num",
            str(eval_cfg.get("part_num", 1)),
            "--part-idx",
            str(eval_cfg.get("part_idx", -1)),
        ]
        if eval_cfg.get("enable_thinking", False):
            command.append("--enable-thinking")
        if eval_cfg.get("debug", False):
            command.append("--debug")
        if eval_cfg.get("override", False):
            command.append("--override")
        if agent_backend == "coevolving_local_hf":
            if not student_world_model_path:
                raise ValueError("Co-evolving evaluation requires student_world_model_path.")
            command.extend(["--wm-student-model-path", student_world_model_path])
            if eval_cfg.get("include_teacher_world_model", False) and teacher_world_model_path:
                command.extend(["--wm-teacher-model-path", teacher_world_model_path])
        self._run(command)
        return str(output_dir)

    def run_round(
        self,
        round_id: int,
        *,
        policy_model_path: str | None = None,
        world_model_path: str | None = None,
    ) -> RoundArtifacts:
        metrics = MetricsLogger.for_round(self.layout.metrics_path(round_id), round_id)
        current_policy_path = self.registry.resolve_policy_input(round_id, explicit=policy_model_path)
        current_world_model_path = self.registry.resolve_world_model_input(
            round_id,
            explicit=world_model_path,
            allow_policy_fallback=False,
        )
        current_teacher_world_model_path = self.registry.resolve_teacher_world_model_input(round_id)
        metrics.log_stage(
            "resolve_inputs",
            policy_model_path=current_policy_path,
            world_model_path=current_world_model_path,
            teacher_world_model_path=current_teacher_world_model_path,
        )

        rollout_dir = self.run_rollout(
            round_id,
            current_policy_path,
            student_world_model_path=current_world_model_path,
            teacher_world_model_path=current_teacher_world_model_path,
        )
        metrics.log_stage("rollout_complete", raw_rollout_dir=rollout_dir)

        round_buffer_path = self.export_round_buffer(round_id, rollout_dir, current_policy_path)
        metrics.log_stage("round_buffer_complete", round_buffer_path=round_buffer_path)

        world_model_output_dir = self.train_world_model(
            round_id,
            round_buffer_path,
            current_world_model_path=current_world_model_path,
            teacher_world_model_path=current_teacher_world_model_path,
            fallback_policy_model_path=current_policy_path,
        )
        metrics.log_stage("world_model_complete", world_model_output_dir=world_model_output_dir)

        teacher_world_model_output_dir = self.update_teacher_world_model(round_id, world_model_output_dir)
        metrics.log_stage("teacher_world_model_complete", teacher_world_model_output_dir=teacher_world_model_output_dir)

        sdpo_dataset_dir = self.build_sdpo_buffer(round_id, round_buffer_path)
        metrics.log_stage("sdpo_buffer_complete", sdpo_dataset_dir=sdpo_dataset_dir)

        policy_output_dir = self.train_policy(round_id, sdpo_dataset_dir, current_policy_path)
        metrics.log_stage("sdpo_training_complete", policy_output_dir=policy_output_dir)

        evaluation_output_dir = self.evaluate_policy(
            round_id + 1,
            policy_output_dir,
            student_world_model_path=world_model_output_dir,
            teacher_world_model_path=teacher_world_model_output_dir,
        )
        metrics.log_stage("evaluation_complete", evaluation_output_dir=evaluation_output_dir)

        artifacts = RoundArtifacts(
            rollout_output_dir=rollout_dir,
            round_buffer_path=round_buffer_path,
            world_model_output_dir=world_model_output_dir,
            teacher_world_model_output_dir=teacher_world_model_output_dir,
            sdpo_dataset_dir=sdpo_dataset_dir,
            policy_output_dir=policy_output_dir,
            evaluation_output_dir=evaluation_output_dir,
        )
        metrics.finalize(artifacts=artifacts.__dict__)
        return artifacts
