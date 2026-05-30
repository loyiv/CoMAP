from __future__ import annotations

import json
from pathlib import Path

import torch


def _load_base_generation_model(model_name_or_path: str, *, torch_dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    except Exception:
        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )


def _is_peft_adapter_dir(model_name_or_path: str) -> bool:
    return (Path(model_name_or_path).expanduser().resolve() / "adapter_config.json").exists()


def _adapter_base_model_path(adapter_dir: Path) -> str:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_dir / 'adapter_config.json'}")
    return base_model_path


def load_generation_model(model_name_or_path: str, *, is_trainable: bool = False):
    from transformers import AutoTokenizer

    model_path = Path(model_name_or_path).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if _is_peft_adapter_dir(model_name_or_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load LoRA world-model checkpoints.") from exc

        base_model = _load_base_generation_model(
            _adapter_base_model_path(model_path),
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
            is_trainable=is_trainable,
        )
    else:
        model = _load_base_generation_model(model_name_or_path, torch_dtype=torch_dtype)

    return model, tokenizer
