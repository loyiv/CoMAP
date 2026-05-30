from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import ProjectLayout, read_json, write_json, write_jsonl

from .schema import (
    RolloutStepRecord,
    build_episode_id,
    extract_goal,
    messages_to_text,
    normalize_conversations,
    output_file_stem,
    parse_action,
    split_episode_messages,
    strip_observation_prefix,
)
from .serialize_feedback import concise_env_feedback


def _tokenize_text(text: str) -> set[str]:
    return {token for token in str(text or "").lower().split() if token}


def _wm_consistency_score(predicted_text: str | None, next_observation: str) -> float | None:
    predicted_tokens = _tokenize_text(predicted_text)
    target_tokens = _tokenize_text(strip_observation_prefix(next_observation))
    if not predicted_tokens or not target_tokens:
        return None
    overlap = len(predicted_tokens & target_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def episode_to_step_records(
    episode_json: dict[str, Any],
    *,
    task_name: str,
    task_id: str,
    round_id: int,
    policy_ckpt: str,
    output_file: str,
    step_metadata: list[dict[str, Any]] | None = None,
) -> list[RolloutStepRecord]:
    meta = episode_json.get("meta", {})
    conversations = normalize_conversations(episode_json.get("conversations", []))
    num_steps = int(meta.get("steps") or 0)
    preamble, step_pairs = split_episode_messages(conversations, num_steps)
    if not preamble or not step_pairs:
        return []

    goal = extract_goal(preamble[-1]["content"])
    game_file = meta.get("error")
    episode_id = build_episode_id(task_name=task_name, task_id=task_id, game_file=game_file)
    episode_reward = float(meta.get("reward") or 0.0)
    episode_success = bool(meta.get("success"))
    terminate_reason = meta.get("terminate_reason")

    records: list[RolloutStepRecord] = []
    for step_id, (assistant_message, user_message) in enumerate(step_pairs):
        metadata = (step_metadata or [])
        step_meta = metadata[step_id] if step_id < len(metadata) else {}
        history_messages = conversations[: len(preamble) + (2 * step_id)]
        observation = history_messages[-1]["content"] if history_messages else preamble[-1]["content"]
        next_observation = user_message["content"]
        done = step_id == len(step_pairs) - 1
        reward = episode_reward if done else 0.0
        success = episode_success if done else False
        final_action = step_meta.get("final_action") or parse_action(assistant_message["content"])
        student_prediction = step_meta.get("pred_next_state_student") or step_meta.get("imagined_next_state")
        teacher_prediction = step_meta.get("pred_next_state_teacher")
        wm_teacher_hint = step_meta.get("wm_teacher_hint")
        wm_consistency = _wm_consistency_score(student_prediction, next_observation)
        feedback_text = concise_env_feedback(
            observation=observation,
            next_observation=next_observation,
            reward=reward,
            done=done,
            success=success,
        )
        records.append(
            RolloutStepRecord(
                task_name=task_name,
                task_id=str(task_id),
                episode_id=episode_id,
                round_id=round_id,
                step_id=step_id,
                goal=goal,
                history_messages=history_messages,
                history_text=messages_to_text(history_messages),
                observation=observation,
                action=final_action,
                action_text=assistant_message["content"],
                reward=reward,
                done=done,
                success=success,
                next_observation=next_observation,
                feedback_text=feedback_text,
                policy_ckpt=policy_ckpt,
                trajectory_source="eto_output_json",
                terminate_reason=terminate_reason,
                game_file=game_file,
                output_file=output_file,
                flow_mode=step_meta.get("flow_mode"),
                draft_response=step_meta.get("draft_response"),
                draft_action=step_meta.get("draft_action"),
                draft_action_prob=step_meta.get("draft_action_prob"),
                pred_next_state_student=student_prediction,
                pred_next_state_teacher=teacher_prediction,
                reflect_response=step_meta.get("reflect_response"),
                revised_action=step_meta.get("revised_action"),
                reflect_action_prob=step_meta.get("reflect_action_prob"),
                revise_triggered=step_meta.get("revise_triggered"),
                used_revised_action=step_meta.get("used_revised_action"),
                wm_consistency=wm_consistency,
                student_world_model_path=step_meta.get("student_world_model_path"),
                teacher_world_model_path=step_meta.get("teacher_world_model_path"),
                imagined_next_state=student_prediction,
                real_next_state=strip_observation_prefix(next_observation),
                wm_teacher_hint=wm_teacher_hint,
            )
        )
    return records


def convert_episode_dir(
    input_dir: str | Path,
    *,
    task_name: str,
    round_id: int,
    policy_ckpt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = Path(input_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    num_episodes = 0
    num_success_episodes = 0

    for episode_file in sorted(input_path.glob("*.json")):
        episode_json = read_json(episode_file)
        meta = episode_json.get("meta", {})
        num_episodes += 1
        if meta.get("success"):
            num_success_episodes += 1

        step_records = episode_to_step_records(
            episode_json,
            task_name=task_name,
            task_id=output_file_stem(episode_file),
            round_id=round_id,
            policy_ckpt=policy_ckpt,
            output_file=str(episode_file),
            step_metadata=episode_json.get("step_metadata", []),
        )
        records.extend(record.to_dict() for record in step_records)

    stats = {
        "input_dir": str(input_path),
        "task_name": task_name,
        "round_id": round_id,
        "policy_ckpt": policy_ckpt,
        "num_episodes": num_episodes,
        "num_success_episodes": num_success_episodes,
        "num_step_records": len(records),
    }
    return records, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ETO episode outputs into step-level round buffers.")
    parser.add_argument("--input-dir", required=True, help="Directory containing ETO episode JSON files.")
    parser.add_argument("--task-name", default="alfworld", help="Task name for exported records.")
    parser.add_argument("--round-id", type=int, required=True, help="Current round id.")
    parser.add_argument("--policy-ckpt", required=True, help="Policy checkpoint used to produce the rollout.")
    parser.add_argument("--output-path", default=None, help="Destination JSONL path. Defaults to buffers/wm/round_<id>_all.jsonl.")
    parser.add_argument("--stats-path", default=None, help="Optional stats JSON output path.")
    parser.add_argument("--project-root", default=None, help="Project root override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    layout = ProjectLayout.discover(root=args.project_root)
    layout.ensure_directories()
    output_path = args.output_path or layout.round_wm_buffer(args.round_id, "all")
    stats_path = args.stats_path or Path(output_path).with_suffix(".stats.json")

    records, stats = convert_episode_dir(
        args.input_dir,
        task_name=args.task_name,
        round_id=args.round_id,
        policy_ckpt=args.policy_ckpt,
    )
    write_jsonl(records, output_path)
    write_json(stats, stats_path)

    print(f"Wrote {len(records)} step records to {output_path}")
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
