import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .base import LMAgent


THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\nAction:|Action:|$)", re.DOTALL)
ACTION_PATTERN = re.compile(r"Action:\s*([^\n\r]+)")
ENTITY_PATTERN = re.compile(r"\b([a-z][a-z0-9]*(?: [a-z][a-z0-9]*)* \d+)\b", re.IGNORECASE)
PUT_GOAL_PATTERN = re.compile(r"Your task is to:\s*put .*? (?:in|on|into|onto) ([a-z][a-z0-9]*)", re.IGNORECASE)

GO_ACTION_PATTERN = re.compile(r"^go to [a-z0-9][a-z0-9 ]+$")
TAKE_ACTION_PATTERN = re.compile(r"^take [a-z0-9][a-z0-9 ]+ from [a-z0-9][a-z0-9 ]+$")
PUT_ACTION_PATTERN = re.compile(r"^put [a-z0-9][a-z0-9 ]+ in/on [a-z0-9][a-z0-9 ]+$")
OPEN_CLOSE_ACTION_PATTERN = re.compile(r"^(open|close) [a-z0-9][a-z0-9 ]+$")
TOGGLE_ACTION_PATTERN = re.compile(r"^toggle [a-z0-9][a-z0-9 ]+$")
TOOL_ACTION_PATTERN = re.compile(r"^(clean|heat|cool) [a-z0-9][a-z0-9 ]+ with [a-z0-9][a-z0-9 ]+$")


def _truncate_at_stop_words(text: str, stop_words: List[str]) -> str:
    best_idx = None
    for stop in stop_words:
        idx = text.find(stop)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
    if best_idx is not None:
        return text[:best_idx]
    return text


def _normalize_action_text(action: str) -> str:
    action = action.strip().lower()
    action = action.replace(" the ", " ")
    action = action.replace(" onto ", " on ")
    action = action.replace(" into ", " in ")
    action = action.replace("pickup ", "take ")
    action = action.replace("pick up ", "take ")
    action = action.replace("grab ", "take ")
    action = action.replace("walk to ", "go to ")
    action = action.replace("move to ", "go to ")
    action = re.sub(r"\s+", " ", action)
    return action.strip(" .")


def _extract_entities(text: str) -> List[str]:
    entities = []
    seen = set()
    for match in ENTITY_PATTERN.finditer(text):
        entity = match.group(1).strip().lower()
        for prefix in [
            "observation: ",
            "you arrive at ",
            "you are at ",
            "you see a ",
            "you see an ",
            "you see ",
            "on the ",
            "in the ",
            "the ",
            "a ",
            "an ",
            "at ",
        ]:
            if entity.startswith(prefix):
                entity = entity[len(prefix):].strip()
        if entity not in seen:
            entities.append(entity)
            seen.add(entity)
    return entities


def _first_numbered_entity(text: str) -> Optional[str]:
    tokens = text.strip().lower().split()
    if not tokens:
        return None

    entity_tokens = []
    for token in tokens:
        entity_tokens.append(token)
        if token.isdigit():
            return " ".join(entity_tokens)
    return None


def _latest_observation_text(messages: List[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user" and "Observation:" in message["content"]:
            return message["content"]
    return messages[-1]["content"] if messages else ""


def _find_entity_with_base(messages: List[Dict[str, str]], base_name: str) -> Optional[str]:
    base_name = base_name.lower().strip()
    pattern = re.compile(rf"\b({re.escape(base_name)} \d+)\b")
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        match = pattern.search(message["content"].lower())
        if match:
            return match.group(1).strip()
    return None


def _infer_goal_receptacle(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages:
        return None
    first_user_text = messages[0]["content"]
    match = PUT_GOAL_PATTERN.search(first_user_text)
    if not match:
        return None
    return _find_entity_with_base(messages, match.group(1))


def _fallback_action(messages: List[Dict[str, str]]) -> str:
    current_observation = _latest_observation_text(messages)
    entities = _extract_entities(current_observation)
    if entities:
        return f"go to {entities[0]}"
    goal_receptacle = _infer_goal_receptacle(messages)
    if goal_receptacle:
        return f"go to {goal_receptacle}"
    return "go to cabinet 1"


def _canonicalize_action(action: str, messages: List[Dict[str, str]]) -> str:
    action = _normalize_action_text(action)

    if action == "go":
        return _fallback_action(messages)

    if action.startswith("go ") and not action.startswith("go to "):
        action = "go to " + action[3:].strip()

    if action.startswith("take ") and " out of " in action:
        action = action.replace(" out of ", " from ")

    tool_match = re.match(r"^(clean|heat|cool) (.+?) (?:in|on) (.+)$", action)
    if tool_match:
        action = f"{tool_match.group(1)} {tool_match.group(2).strip()} with {tool_match.group(3).strip()}"

    put_full_match = re.match(r"^put (.+?) (?:in/on|on|in) (.+)$", action)
    if put_full_match:
        action = f"put {put_full_match.group(1).strip()} in/on {put_full_match.group(2).strip()}"
    elif action.startswith("put "):
        partial_put = re.match(r"^put (.+?) (?:in/on|on|in)\s*$", action)
        if partial_put:
            target = _infer_goal_receptacle(messages)
            if target:
                action = f"put {partial_put.group(1).strip()} in/on {target}"
            else:
                return _fallback_action(messages)

    if action.startswith("toggle "):
        toggle_entity = _first_numbered_entity(action[len("toggle ") :])
        if toggle_entity:
            action = f"toggle {toggle_entity}"

    action = re.sub(r"\s+", " ", action).strip()

    if (
        GO_ACTION_PATTERN.match(action)
        or TAKE_ACTION_PATTERN.match(action)
        or PUT_ACTION_PATTERN.match(action)
        or OPEN_CLOSE_ACTION_PATTERN.match(action)
        or TOGGLE_ACTION_PATTERN.match(action)
        or TOOL_ACTION_PATTERN.match(action)
    ):
        return action

    if action.startswith(("open ", "close ", "toggle ")):
        verb, remainder = action.split(" ", 1)
        entities = _extract_entities(remainder)
        if entities:
            return f"{verb} {entities[0]}"

    if action.startswith("go "):
        entities = _extract_entities(action)
        if entities:
            return f"go to {entities[0]}"

    if action.startswith("take "):
        entities = _extract_entities(action)
        if len(entities) >= 2:
            return f"take {entities[0]} from {entities[1]}"

    if action.startswith("put "):
        entities = _extract_entities(action)
        if len(entities) >= 2:
            return f"put {entities[0]} in/on {entities[1]}"

    if action.startswith(("clean ", "heat ", "cool ")):
        entities = _extract_entities(action)
        if len(entities) >= 2:
            verb = action.split(" ", 1)[0]
            return f"{verb} {entities[0]} with {entities[1]}"

    return _fallback_action(messages)


def _extract_first_thought_and_action(text: str, messages: List[Dict[str, str]]) -> str | None:
    thought_match = THOUGHT_PATTERN.search(text)
    action_match = ACTION_PATTERN.search(text)
    if action_match is None:
        return None

    action = _canonicalize_action(action_match.group(1).strip(), messages)
    if thought_match is not None:
        thought = " ".join(thought_match.group(1).strip().split())
    else:
        prefix = text[: action_match.start()].strip()
        prefix = prefix.splitlines()[-1].strip() if prefix else ""
        thought = " ".join(prefix.split()) if prefix else "I should choose the next valid action."

    if not thought:
        thought = "I should choose the next valid action."

    return f"Thought: {thought}\nAction: {action}"


def _is_peft_adapter_dir(model_path: str) -> bool:
    return (Path(model_path).expanduser().resolve() / "adapter_config.json").exists()


def _adapter_base_model_path(adapter_dir: Path) -> str:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_dir / 'adapter_config.json'}")
    return base_model_path


class HuggingFaceLocalAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model_path = config["model_path"]
        self.model_name = config.get("model_name", os.path.basename(self.model_path.rstrip("/")))
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 1.0))
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        adapter_dir = Path(self.model_path).expanduser().resolve()
        tokenizer_path = _adapter_base_model_path(adapter_dir) if _is_peft_adapter_dir(self.model_path) else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_path = tokenizer_path
        try:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                trust_remote_code=True,
                dtype=self.torch_dtype,
            )
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(
                load_path,
                trust_remote_code=True,
                dtype=self.torch_dtype,
            )
        if _is_peft_adapter_dir(self.model_path):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        self.model = model

        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n\nassistant:"

    def _generate_text(self, prompt: str) -> str:
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][model_inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def __call__(self, messages: List[dict]) -> str:
        prompt = self._build_prompt(messages)
        text = _truncate_at_stop_words(self._generate_text(prompt), self.stop_words).strip()
        formatted = _extract_first_thought_and_action(text, messages)
        if formatted is not None:
            return formatted

        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous reply missed the required `Action:` line. "
                    "Reply again using exactly this format and nothing else:\n"
                    "Thought: <one short sentence>\n"
                    "Action: <one valid action>"
                ),
            }
        ]
        retry_prompt = self._build_prompt(retry_messages)
        retry_text = _truncate_at_stop_words(self._generate_text(retry_prompt), self.stop_words).strip()
        retry_formatted = _extract_first_thought_and_action(retry_text, retry_messages)
        if retry_formatted is not None:
            return retry_formatted

        return retry_text or text
