from __future__ import annotations

import argparse

from bridge.schema import RolloutStepRecord
from common import read_jsonl

from .datasets import build_world_model_prompt
from .model_loading import load_generation_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-step ALFWorld world model inference.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--round-buffer-path", default=None, help="Optional round buffer JSONL.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index when using --round-buffer-path.")
    parser.add_argument("--goal", default=None)
    parser.add_argument("--history-text", default=None)
    parser.add_argument("--action-text", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser


def _prompt_from_args(args) -> str:
    if args.round_buffer_path:
        records = [RolloutStepRecord(**record) for record in read_jsonl(args.round_buffer_path)]
        record = records[args.record_index]
        return build_world_model_prompt(record)
    if not (args.goal and args.history_text and args.action_text):
        raise ValueError("Either provide --round-buffer-path or all of --goal/--history-text/--action-text.")
    temp_record = RolloutStepRecord(
        task_name="alfworld",
        task_id="adhoc",
        episode_id="adhoc",
        round_id=0,
        step_id=0,
        goal=args.goal,
        history_messages=[],
        history_text=args.history_text,
        observation="",
        action=args.action_text,
        action_text=args.action_text,
        reward=0.0,
        done=False,
        success=False,
        next_observation="",
        feedback_text="",
        policy_ckpt="",
        trajectory_source="adhoc",
    )
    return build_world_model_prompt(temp_record)


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("transformers is required for world model inference.") from exc

    model, tokenizer = load_generation_model(args.model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompt = _prompt_from_args(args)
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][model_inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
