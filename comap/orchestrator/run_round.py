from __future__ import annotations

import argparse

from common import ProjectLayout

from .round_manager import RoundManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ETO x WM x SDPO round.")
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--config", default=None, help="Path to project-level YAML config.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--policy-model-path", default=None, help="Override the current policy model path.")
    parser.add_argument("--world-model-path", default=None, help="Override the current world model path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    layout = ProjectLayout.discover(root=args.project_root)
    config_path = args.config or layout.training_root / "configs" / "eto_alfworld_sdpo.yaml"

    manager = RoundManager(config_path, project_root=layout.root, dry_run=args.dry_run)
    artifacts = manager.run_round(
        args.round_id,
        policy_model_path=args.policy_model_path,
        world_model_path=args.world_model_path,
    )
    print("Round completed with artifacts:")
    for key, value in artifacts.__dict__.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
