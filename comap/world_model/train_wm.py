from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import ProjectLayout, write_json

from .datasets import WorldModelCollator, WorldModelDataset
from .losses import masked_cross_entropy
from .model_loading import load_generation_model
from .wmsd import masked_kl_distillation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the one-step ALFWorld textual world model used by COMAP.")
    parser.add_argument("--train-jsonl", required=True, help="Training JSONL buffer.")
    parser.add_argument("--eval-jsonl", default=None, help="Optional evaluation JSONL buffer.")
    parser.add_argument("--model-name-or-path", required=True, help="Base CausalLM checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Checkpoint output directory.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--teacher-model-name-or-path", default=None)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--project-root", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is required for world model training.") from exc

    class DistillationTrainer(Trainer):
        def __init__(
            self,
            *trainer_args,
            distill_weight: float = 0.0,
            teacher_model=None,
            distill_temperature: float = 1.0,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.distill_weight = distill_weight
            self.teacher_model = teacher_model
            self.distill_temperature = distill_temperature
            if self.teacher_model is not None:
                self.teacher_model.eval()

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            batch = dict(inputs)
            batch.pop("metadata", None)
            labels = batch.pop("labels")
            teacher_labels = batch.pop("teacher_labels", None)
            outputs = model(**batch)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_real = masked_cross_entropy(shift_logits, shift_labels)
            loss = loss_real

            if teacher_labels is not None:
                shift_teacher_labels = teacher_labels[..., 1:].contiguous()
                if self.teacher_model is not None:
                    self.teacher_model.to(model.device)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)
                    loss_sd = masked_kl_distillation(
                        student_logits=logits,
                        teacher_logits=teacher_outputs.logits,
                        labels=labels,
                        temperature=self.distill_temperature,
                    )
                    loss = loss_real + (self.distill_weight * loss_sd)
                elif torch.any(shift_teacher_labels.ne(-100)):
                    loss_sd = masked_cross_entropy(shift_logits, shift_teacher_labels)
                    loss = loss_real + (self.distill_weight * loss_sd)

            return (loss, outputs) if return_outputs else loss

    layout = ProjectLayout.discover(root=args.project_root)
    layout.ensure_directories()

    model, tokenizer = load_generation_model(args.model_name_or_path, is_trainable=True)
    teacher_model = None
    if args.teacher_model_name_or_path:
        teacher_model, _ = load_generation_model(args.teacher_model_name_or_path, is_trainable=False)
    if args.use_lora and not hasattr(model, "peft_config"):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required when --use-lora is enabled.") from exc

        target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_dataset = WorldModelDataset(
        args.train_jsonl,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No world-model training samples found in {args.train_jsonl}.")
    eval_dataset = None
    if args.eval_jsonl:
        eval_dataset = WorldModelDataset(
            args.eval_jsonl,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_target_length=args.max_target_length,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WorldModelCollator(tokenizer),
        distill_weight=args.distill_weight,
        teacher_model=teacher_model,
        distill_temperature=args.distill_temperature,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    write_json(
        {
            "train_jsonl": str(Path(args.train_jsonl).expanduser().resolve()),
            "eval_jsonl": str(Path(args.eval_jsonl).expanduser().resolve()) if args.eval_jsonl else None,
            "model_name_or_path": args.model_name_or_path,
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            "use_lora": args.use_lora,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_steps": args.max_steps,
            "distill_weight": args.distill_weight,
            "teacher_model_name_or_path": args.teacher_model_name_or_path,
            "distill_temperature": args.distill_temperature,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
        },
        Path(args.output_dir).expanduser().resolve() / "training_metadata.json",
    )
    print(f"Saved world model checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
