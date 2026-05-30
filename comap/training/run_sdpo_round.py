from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from common import ProjectLayout


def _load_yaml(path: str | Path) -> dict:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one SDPO round from a round buffer directory.")
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--config", default=None, help="Project-level SDPO YAML config.")
    parser.add_argument("--dataset-dir", default=None, help="Directory containing train.parquet/test.parquet.")
    parser.add_argument("--policy-model-path", default=None, help="Current policy checkpoint or HF model id.")
    parser.add_argument("--output-dir", default=None, help="Directory for the next policy checkpoint.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _hydra_string(value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _is_peft_adapter_dir(path: str | Path) -> bool:
    return (Path(path).expanduser().resolve() / "adapter_config.json").exists()


def _adapter_base_model_path(adapter_dir: str | Path) -> str:
    adapter_path = Path(adapter_dir).expanduser().resolve()
    with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_path / 'adapter_config.json'}")
    return base_model_path


def main() -> None:
    args = build_parser().parse_args()
    layout = ProjectLayout.discover(root=args.project_root)
    layout.ensure_directories()

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else layout.training_root / "configs" / "eto_alfworld_sdpo.yaml"
    )
    config = _load_yaml(config_path)

    dataset_dir = Path(args.dataset_dir or layout.round_sdpo_dir(args.round_id)).expanduser().resolve()
    policy_model_path = args.policy_model_path or config.get("models", {}).get("base_policy_model")
    if not policy_model_path:
        raise ValueError("A policy model path is required via --policy-model-path or models.base_policy_model.")
    policy_lora_adapter_path = policy_model_path if _is_peft_adapter_dir(policy_model_path) else None
    policy_training_base_path = (
        _adapter_base_model_path(policy_lora_adapter_path) if policy_lora_adapter_path else policy_model_path
    )

    output_dir = Path(args.output_dir or layout.policy_checkpoint_dir(args.round_id + 1)).expanduser().resolve()
    dataset_task_arg = str(dataset_dir.relative_to(layout.root)) if dataset_dir.is_relative_to(layout.root) else str(dataset_dir)

    sdpo_cfg = config["sdpo"]
    exp_name = f"ETO-ALFWORLD-round-{args.round_id}"
    overrides = [
        f"vars.dir={_hydra_string(layout.root)}",
        f"data.train_files=[{_hydra_string(dataset_dir / 'train.parquet')}]",
        f"data.val_files=[{_hydra_string(dataset_dir / 'test.parquet')}]",
        f"custom_reward_function.path={_hydra_string(layout.training_root / 'alfworld_reward.py')}",
        "custom_reward_function.name=compute_score",
        f"actor_rollout_ref.model.path={_hydra_string(policy_training_base_path)}",
        f"critic.model.path={_hydra_string(policy_training_base_path)}",
        f"actor_rollout_ref.rollout.name={sdpo_cfg['rollout_backend']}",
        f"actor_rollout_ref.rollout.n={sdpo_cfg['rollout_batch_size']}",
        f"actor_rollout_ref.actor.optim.lr={sdpo_cfg['learning_rate']}",
        f"actor_rollout_ref.actor.self_distillation.alpha={sdpo_cfg['alpha']}",
        f"actor_rollout_ref.actor.self_distillation.success_reward_threshold={sdpo_cfg['success_reward_threshold']}",
        f"actor_rollout_ref.actor.self_distillation.teacher_regularization={sdpo_cfg['teacher_regularization']}",
        f"actor_rollout_ref.actor.self_distillation.include_environment_feedback={str(sdpo_cfg['include_environment_feedback']).lower()}",
        f"actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution={str(sdpo_cfg['environment_feedback_only_without_solution']).lower()}",
        f"actor_rollout_ref.actor.self_distillation.enable_auxiliary_supervision={str(sdpo_cfg.get('enable_auxiliary_supervision', True)).lower()}",
        f"actor_rollout_ref.actor.self_distillation.reflect_main_loss_weight={sdpo_cfg.get('reflect_main_loss_weight', 1.0)}",
        f"actor_rollout_ref.actor.self_distillation.draft_distill_loss_weight={sdpo_cfg.get('draft_distill_loss_weight', 0.0)}",
        f"actor_rollout_ref.actor.self_distillation.expert_anchor_loss_weight={sdpo_cfg.get('expert_anchor_loss_weight', 0.0)}",
        f"actor_rollout_ref.actor.self_distillation.transition_fallback_loss_weight={sdpo_cfg.get('transition_fallback_loss_weight', 0.0)}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        f"data.train_batch_size={sdpo_cfg['train_batch_size']}",
        f"data.max_prompt_length={sdpo_cfg['max_prompt_length']}",
        f"data.max_response_length={sdpo_cfg['max_response_length']}",
        f"max_model_len={sdpo_cfg['max_model_len']}",
        f"trainer.project_name=ETO-SDPO",
        f"trainer.group_name={exp_name}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.default_local_dir={_hydra_string(output_dir)}",
        f"trainer.n_gpus_per_node={sdpo_cfg['n_gpus_per_node']}",
        f"trainer.nnodes={sdpo_cfg['nnodes']}",
    ]
    if policy_lora_adapter_path:
        overrides.append(f"actor_rollout_ref.model.lora_adapter_path={_hydra_string(policy_lora_adapter_path)}")
    overrides.extend(sdpo_cfg.get("extra_overrides", []))

    command = [
        "bash",
        str(layout.sdpo_root / "training" / "verl_training.sh"),
        exp_name,
        sdpo_cfg["config_name"],
        dataset_task_arg,
        *overrides,
    ]
    if sdpo_cfg.get("conda_env"):
        command = ["conda", "run", "-n", sdpo_cfg["conda_env"], *command]

    env = os.environ.copy()
    env.update(layout.as_env())
    env["USER"] = env.get("USER", "root")

    print("Launching SDPO command:")
    print(" ".join(map(str, command)))
    if args.dry_run:
        return

    subprocess.run(command, cwd=layout.sdpo_root, env=env, check=True)


if __name__ == "__main__":
    main()
