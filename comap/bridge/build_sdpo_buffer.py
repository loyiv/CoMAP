from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from common import ProjectLayout, read_jsonl, write_json

from .schema import RolloutStepRecord
from .serialize_teacher_context import build_student_prompt_text, build_teacher_prompt_preview


def _select_episode_ids(records: list[RolloutStepRecord], episode_filter: str) -> set[str]:
    if episode_filter == "all":
        return {record.episode_id for record in records}

    successful_episode_ids = {record.episode_id for record in records if record.done and record.success}
    if episode_filter == "successful_only":
        return successful_episode_ids

    if successful_episode_ids:
        return successful_episode_ids
    return {record.episode_id for record in records}


def _episode_to_split(episode_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < val_ratio else "train"


def _build_reward_model(target_action: str) -> dict[str, Any]:
    return {
        "style": "alfworld_step",
        "ground_truth": target_action,
    }


def _build_extra_info(
    record: RolloutStepRecord,
    *,
    target_action: str,
    prompt_mode: str,
    sample_origin: str,
) -> dict[str, Any]:
    refinement_label = 1 if sample_origin in {"revise_positive", "hq_reflect"} else 0
    return {
        "index": f"{record.episode_id}:{record.step_id}",
        "task_name": record.task_name,
        "task_id": record.task_id,
        "round_id": record.round_id,
        "episode_id": record.episode_id,
        "step_id": record.step_id,
        "goal": record.goal,
        "observation": record.observation,
        "next_observation": record.next_observation,
        "feedback_text": record.feedback_text,
        "target_action": target_action,
        "target_response": record.action_text,
        "student_prompt_text": build_student_prompt_text(record),
        "teacher_prompt_preview": build_teacher_prompt_preview(record),
        "trajectory_source": record.trajectory_source,
        "policy_ckpt": record.policy_ckpt,
        "game_file": record.game_file,
        "terminate_reason": record.terminate_reason,
        "prompt_mode": prompt_mode,
        "sample_origin": sample_origin,
        "refinement_label": refinement_label,
        "flow_mode": record.flow_mode,
        "draft_action": record.draft_action,
        "revised_action": record.revised_action,
        "revise_triggered": record.revise_triggered,
        "used_revised_action": record.used_revised_action,
        "pred_next_state_student": record.pred_next_state_student,
        "pred_next_state_teacher": record.pred_next_state_teacher,
        "wm_consistency": record.wm_consistency,
        "real_next_state": record.real_next_state,
        "imagined_next_state": record.imagined_next_state,
        "wm_teacher_hint": record.wm_teacher_hint,
    }


def _build_dataset_row(
    record: RolloutStepRecord,
    *,
    prompt_messages: list[dict[str, str]],
    target_action: str,
    prompt_mode: str,
    sample_origin: str,
) -> dict[str, Any]:
    return {
        "data_source": "alfworld_step",
        "prompt": prompt_messages,
        "ability": "alfworld_step",
        "reward_model": _build_reward_model(target_action),
        "extra_info": _build_extra_info(
            record,
            target_action=target_action,
            prompt_mode=prompt_mode,
            sample_origin=sample_origin,
        ),
        "uid": f"{record.episode_id}:{record.step_id}:{sample_origin}",
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to build SDPO parquet buffers.") from exc

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _build_transition_prompt(record: RolloutStepRecord) -> list[dict[str, str]]:
    return list(record.history_messages)


def _build_reflect_prompt(record: RolloutStepRecord) -> list[dict[str, str]]:
    predicted_future = record.pred_next_state_student or record.imagined_next_state or "Unknown."
    draft_action = record.draft_action or record.action
    reflect_instruction = (
        "You already proposed this draft action:\n"
        f"Action: {draft_action}\n\n"
        "A student world model predicts that the next observation after the draft action would be:\n"
        f"{predicted_future}\n\n"
        "Reflect on the predicted future state and choose the best next action. "
        "Reply in exactly this format:\n"
        "Thought: <one short sentence>\n"
        "Action: <one valid action>"
    )
    return list(record.history_messages) + [{"role": "user", "content": reflect_instruction}]


def _target_action_for_online_sample(record: RolloutStepRecord) -> str:
    return record.revised_action or record.action


def _episode_success_map(records: list[RolloutStepRecord]) -> dict[str, bool]:
    success_by_episode: dict[str, bool] = {}
    for record in records:
        success_by_episode.setdefault(record.episode_id, False)
        if record.done and record.success:
            success_by_episode[record.episode_id] = True
    return success_by_episode


def _actions_differ(record: RolloutStepRecord) -> bool:
    draft_action = (record.draft_action or "").strip().lower()
    revised_action = (record.revised_action or "").strip().lower()
    return bool(draft_action and revised_action and draft_action != revised_action)


def _is_hq_reflect_step(
    record: RolloutStepRecord,
    *,
    episode_success: bool,
    require_success_for_hq: bool,
    wm_consistency_min: float | None,
) -> bool:
    if not record.revise_triggered:
        return False
    if record.used_revised_action is not True:
        return False
    if not _actions_differ(record):
        return False
    if require_success_for_hq and not episode_success:
        return False
    if wm_consistency_min is not None and record.wm_consistency is not None:
        if record.wm_consistency < wm_consistency_min:
            return False
    return True


def _has_draft_signal(record: RolloutStepRecord) -> bool:
    return bool((record.draft_action or "").strip())


def _build_rows_for_mode(
    records: list[RolloutStepRecord],
    *,
    policy_dataset_mode: str,
    include_expert_anchors: bool,
    include_draft_distill: bool,
    require_success_for_hq: bool,
    allow_fallback_to_all_steps: bool,
    wm_consistency_min: float | None,
) -> list[dict[str, Any]]:
    if policy_dataset_mode == "transition":
        return [
            _build_dataset_row(
                record,
                prompt_messages=_build_transition_prompt(record),
                target_action=record.action,
                prompt_mode="transition",
                sample_origin="transition",
            )
            for record in records
        ]

    success_by_episode = _episode_success_map(records)
    rows: list[dict[str, Any]] = []
    if include_expert_anchors:
        for record in records:
            rows.append(
                _build_dataset_row(
                    record,
                    prompt_messages=_build_transition_prompt(record),
                    target_action=record.action,
                    prompt_mode="expert_anchor",
                    sample_origin="expert_anchor",
                )
            )

    for record in records:
        if not _has_draft_signal(record):
            continue
        if record.used_revised_action is True and _actions_differ(record):
            rows.append(
                _build_dataset_row(
                    record,
                    prompt_messages=_build_reflect_prompt(record),
                    target_action=_target_action_for_online_sample(record),
                    prompt_mode="reflect_online",
                    sample_origin="revise_positive",
                )
            )
        else:
            rows.append(
                _build_dataset_row(
                    record,
                    prompt_messages=_build_reflect_prompt(record),
                    target_action=record.draft_action or record.action,
                    prompt_mode="reflect_keep",
                    sample_origin="keep_draft",
                )
            )

    for record in records:
        if not _is_hq_reflect_step(
            record,
            episode_success=success_by_episode.get(record.episode_id, False),
            require_success_for_hq=require_success_for_hq,
            wm_consistency_min=wm_consistency_min,
        ):
            continue

        target_action = _target_action_for_online_sample(record)
        rows.append(
            _build_dataset_row(
                record,
                prompt_messages=_build_reflect_prompt(record),
                target_action=target_action,
                prompt_mode="reflect_online",
                sample_origin="hq_reflect",
            )
        )
        if include_draft_distill:
            rows.append(
                _build_dataset_row(
                    record,
                    prompt_messages=_build_transition_prompt(record),
                    target_action=target_action,
                    prompt_mode="draft_distill",
                    sample_origin="hq_draft_distill",
                )
            )

    if rows or not allow_fallback_to_all_steps:
        return rows

    return [
        _build_dataset_row(
            record,
            prompt_messages=_build_transition_prompt(record),
            target_action=record.action,
            prompt_mode="transition_fallback",
            sample_origin="fallback_transition",
        )
        for record in records
    ]


def build_sdpo_dataset(
    *,
    round_buffer_path: str | Path,
    output_dir: str | Path,
    episode_filter: str = "successful_or_all",
    val_ratio: float = 0.1,
    policy_dataset_mode: str = "transition",
    include_expert_anchors: bool = True,
    include_draft_distill: bool = True,
    require_success_for_hq: bool = True,
    allow_fallback_to_all_steps: bool = True,
    wm_consistency_min: float | None = None,
) -> dict[str, Any]:
    raw_records = [RolloutStepRecord(**record) for record in read_jsonl(round_buffer_path)]
    if not raw_records:
        raise ValueError(f"No round records found in {round_buffer_path}.")

    selected_episode_ids = _select_episode_ids(raw_records, episode_filter)
    selected_records = [record for record in raw_records if record.episode_id in selected_episode_ids]
    if not selected_records:
        raise ValueError(
            f"No records left after applying episode_filter={episode_filter} to {round_buffer_path}."
        )

    rows = _build_rows_for_mode(
        selected_records,
        policy_dataset_mode=policy_dataset_mode,
        include_expert_anchors=include_expert_anchors,
        include_draft_distill=include_draft_distill,
        require_success_for_hq=require_success_for_hq,
        allow_fallback_to_all_steps=allow_fallback_to_all_steps,
        wm_consistency_min=wm_consistency_min,
    )
    if not rows:
        raise ValueError(
            "No SDPO rows were produced from the selected rollout records. "
            "Relax the HQ filters or enable fallback-to-all-steps."
        )

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        split = _episode_to_split(row["extra_info"]["episode_id"], val_ratio)
        if split == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows and test_rows:
        train_rows.append(test_rows[0])
    if not test_rows and train_rows:
        test_rows.append(train_rows[-1])

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    write_json(train_rows, output_path / "train.json")
    write_json(test_rows, output_path / "test.json")
    _write_parquet(train_rows, output_path / "train.parquet")
    _write_parquet(test_rows, output_path / "test.parquet")

    stats = {
        "round_buffer_path": str(Path(round_buffer_path).expanduser().resolve()),
        "output_dir": str(output_path),
        "episode_filter": episode_filter,
        "val_ratio": val_ratio,
        "policy_dataset_mode": policy_dataset_mode,
        "include_expert_anchors": include_expert_anchors,
        "include_draft_distill": include_draft_distill,
        "require_success_for_hq": require_success_for_hq,
        "allow_fallback_to_all_steps": allow_fallback_to_all_steps,
        "wm_consistency_min": wm_consistency_min,
        "num_raw_records": len(raw_records),
        "num_selected_records": len(selected_records),
        "num_output_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "num_selected_episodes": len(selected_episode_ids),
        "sample_origin_counts": dict(Counter(row["extra_info"]["sample_origin"] for row in rows)),
        "prompt_mode_counts": dict(Counter(row["extra_info"]["prompt_mode"] for row in rows)),
        "refinement_label_counts": dict(Counter(row["extra_info"]["refinement_label"] for row in rows)),
    }
    write_json(stats, output_path / "stats.json")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SDPO-compatible parquet buffers from round JSONL records.")
    parser.add_argument("--round-buffer-path", required=True, help="Step-level round buffer JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for train/test parquet.")
    parser.add_argument(
        "--episode-filter",
        default="successful_or_all",
        choices=["successful_or_all", "successful_only", "all"],
        help="Which episodes to keep before split.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--policy-dataset-mode",
        default="transition",
        choices=["transition", "coevolving"],
        help="Whether to export plain transition supervision or HQ co-evolving samples.",
    )
    parser.add_argument(
        "--disable-draft-distill",
        action="store_true",
        help="Only emit reflect-online samples when policy_dataset_mode=coevolving.",
    )
    parser.add_argument(
        "--disable-expert-anchors",
        action="store_true",
        help="Do not emit expert-anchor style supervision rows in co-evolving mode.",
    )
    parser.add_argument(
        "--disable-success-only-hq",
        action="store_true",
        help="Allow non-success episodes to contribute HQ samples in co-evolving mode.",
    )
    parser.add_argument(
        "--disable-fallback-to-all-steps",
        action="store_true",
        help="Do not fall back to transition rows when no HQ sample is found.",
    )
    parser.add_argument(
        "--wm-consistency-min",
        type=float,
        default=None,
        help="Optional minimum WM consistency score for HQ sample selection.",
    )
    parser.add_argument("--project-root", default=None, help="Optional project root override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    layout = ProjectLayout.discover(root=args.project_root)
    layout.ensure_directories()
    stats = build_sdpo_dataset(
        round_buffer_path=args.round_buffer_path,
        output_dir=args.output_dir,
        episode_filter=args.episode_filter,
        val_ratio=args.val_ratio,
        policy_dataset_mode=args.policy_dataset_mode,
        include_expert_anchors=not args.disable_expert_anchors,
        include_draft_distill=not args.disable_draft_distill,
        require_success_for_hq=not args.disable_success_only_hq,
        allow_fallback_to_all_steps=not args.disable_fallback_to_all_steps,
        wm_consistency_min=args.wm_consistency_min,
    )
    print(f"Built SDPO dataset at {stats['output_dir']}")
    print(f"train={stats['num_train_rows']} test={stats['num_test_rows']}")


if __name__ == "__main__":
    main()
