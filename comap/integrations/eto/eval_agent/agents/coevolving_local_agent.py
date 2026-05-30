from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

from bridge.schema import RolloutStepRecord, extract_goal, messages_to_text
from world_model.datasets import build_world_model_prompt
from world_model.model_loading import load_generation_model

from .hf_local_agent import (
    HuggingFaceLocalAgent,
    _extract_first_thought_and_action,
    _normalize_action_text,
    _truncate_at_stop_words,
)


ACTION_PATTERN = re.compile(r"Action:\s*([^\n\r]+)")
FINAL_ACTION_PATTERN = re.compile(r"Final Action:\s*([^\n\r]+)")
REVISE_PROB_PATTERN = re.compile(r"Revise Probability:\s*([01](?:\.\d+)?)")


def _normalize_prediction_text(text: str) -> str:
    text = (text or "").strip()
    for prefix in ("Next observation:", "Observation:"):
        if text.startswith(prefix):
            text = text.split(prefix, 1)[1].strip()
    return text


class CoevolvingLocalAgent(HuggingFaceLocalAgent):
    """A local policy agent with draft -> WM -> reflect execution."""

    def __init__(self, config):
        super().__init__(config)
        self.student_world_model_path = config["student_world_model_path"]
        self.teacher_world_model_path = config.get("teacher_world_model_path")
        self.reflect_max_new_tokens = int(config.get("reflect_max_new_tokens", self.max_new_tokens))
        self.final_max_new_tokens = int(config.get("final_max_new_tokens", 16))
        self.wm_max_new_tokens = int(config.get("wm_max_new_tokens", 192))
        self.revise_probability_threshold = float(config.get("revise_probability_threshold", 0.5))
        self.reflection_confidence_threshold = float(config.get("reflection_confidence_threshold", 0.6))
        self.last_step_metadata: dict[str, Any] | None = None

        self.student_world_model, self.student_world_model_tokenizer = load_generation_model(self.student_world_model_path)
        self.student_world_model.to(self.device)
        self.student_world_model.eval()

        self.teacher_world_model = None
        self.teacher_world_model_tokenizer = None
        self.teacher_world_model_shared = False
        if self.teacher_world_model_path:
            if self.teacher_world_model_path == self.student_world_model_path:
                self.teacher_world_model = self.student_world_model
                self.teacher_world_model_tokenizer = self.student_world_model_tokenizer
                self.teacher_world_model_shared = True
            else:
                self.teacher_world_model, self.teacher_world_model_tokenizer = load_generation_model(
                    self.teacher_world_model_path
                )
                self.teacher_world_model.to(self.device)
                self.teacher_world_model.eval()

    def consume_step_metadata(self) -> dict[str, Any] | None:
        metadata = self.last_step_metadata
        self.last_step_metadata = None
        return metadata

    def _generate_text_with_limit(self, prompt: str, *, max_new_tokens: int) -> str:
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][model_inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_formatted_policy_response(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        prompt = self._build_prompt(messages)
        text = _truncate_at_stop_words(
            self._generate_text_with_limit(prompt, max_new_tokens=max_new_tokens),
            self.stop_words,
        ).strip()
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
        retry_text = _truncate_at_stop_words(
            self._generate_text_with_limit(retry_prompt, max_new_tokens=max_new_tokens),
            self.stop_words,
        ).strip()
        retry_formatted = _extract_first_thought_and_action(retry_text, retry_messages)
        if retry_formatted is not None:
            return retry_formatted
        return text

    def _extract_goal_from_messages(self, messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            if "Your task is to:" in message.get("content", ""):
                return extract_goal(message["content"])
        return extract_goal(messages[-1]["content"]) if messages else ""

    def _build_world_model_prompt(self, messages: List[Dict[str, str]], action: str) -> str:
        temp_record = RolloutStepRecord(
            task_name="alfworld",
            task_id="coev",
            episode_id="coev",
            round_id=0,
            step_id=0,
            goal=self._extract_goal_from_messages(messages),
            history_messages=messages,
            history_text=messages_to_text(messages),
            observation=messages[-1]["content"] if messages else "",
            action=action,
            action_text=action,
            reward=0.0,
            done=False,
            success=False,
            next_observation="",
            feedback_text="",
            policy_ckpt=self.model_path,
            trajectory_source="coevolving_local_agent",
        )
        return build_world_model_prompt(temp_record)

    def _predict_next_state(
        self,
        messages: List[Dict[str, str]],
        action: str,
        *,
        use_teacher: bool = False,
    ) -> Optional[str]:
        model = self.teacher_world_model if use_teacher else self.student_world_model
        tokenizer = self.teacher_world_model_tokenizer if use_teacher else self.student_world_model_tokenizer
        if model is None or tokenizer is None:
            return None

        prompt = self._build_world_model_prompt(messages, action)
        model_inputs = tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=self.wm_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][model_inputs["input_ids"].shape[-1] :]
        return _normalize_prediction_text(tokenizer.decode(generated, skip_special_tokens=True).strip())

    def _extract_action_only(self, response_text: str) -> str:
        match = FINAL_ACTION_PATTERN.search(response_text or "") or ACTION_PATTERN.search(response_text or "")
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_revise_probability(self, response_text: str) -> float:
        match = REVISE_PROB_PATTERN.search(response_text or "")
        if not match:
            return 1.0 if self._extract_action_only(response_text) else 0.0
        return max(0.0, min(1.0, float(match.group(1))))

    def _reflection_confidence(self, reflect_response: str, reflect_action: str) -> float:
        if not reflect_action:
            return 0.0
        # The internal ALFWorld runs did not expose token-level confidence in the
        # ETO logger, so the public minimal code keeps a conservative textual
        # proxy: a parseable final action with an explicit keep/revise decision
        # is treated as high confidence.
        has_decision = "Decision:" in (reflect_response or "")
        return 1.0 if has_decision else 0.7

    def _should_use_refined_action(
        self,
        draft_action: str,
        reflect_action: str,
        revise_probability: float,
        reflection_confidence: float,
    ) -> bool:
        if not reflect_action:
            return False
        if revise_probability <= self.revise_probability_threshold:
            return False
        if reflection_confidence <= self.reflection_confidence_threshold:
            return False
        return _normalize_action_text(reflect_action) != _normalize_action_text(draft_action)

    def _format_final_action_response(self, final_action: str, reflect_response: str, *, used_revised_action: bool) -> str:
        if used_revised_action:
            thought = "I should revise the draft action based on the predicted future state."
        else:
            thought = "I should keep the draft action because the predicted future state is acceptable."
        if reflect_response:
            first_line = reflect_response.strip().splitlines()[0].strip()
            if first_line.startswith("Reflection:"):
                thought = first_line.split("Reflection:", 1)[1].strip() or thought
        return f"Thought: {thought}\nAction: {final_action}"

    def _build_reflect_messages(
        self,
        messages: List[Dict[str, str]],
        draft_response: str,
        student_prediction: str | None,
    ) -> List[Dict[str, str]]:
        student_prediction = student_prediction or "Unknown."
        reflect_instruction = (
            "You are revising a candidate ALFWorld action using a future-state signal.\n\n"
            "You already proposed the following draft response:\n"
            f"{draft_response}\n\n"
            "A student world model predicts that the next observation after the draft action would be:\n"
            f"{student_prediction}\n\n"
            "Revise the draft action only when the future-state signal shows that the draft action is invalid, "
            "unhelpful, harmful, redundant, or clearly worse than another available action. "
            "If the draft action already makes useful progress, keep it unchanged.\n\n"
            "Reply using exactly this format and nothing else:\n"
            "Reflection: <one short analysis of the predicted consequence>\n"
            "Decision: KEEP or REVISE\n"
            "Final Action: <one executable ALFWorld action>\n"
            "Revise Probability: <a number from 0 to 1>"
        )
        return messages + [{"role": "user", "content": reflect_instruction}]

    def _generate_reflection_response(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._build_prompt(messages)
        return _truncate_at_stop_words(
            self._generate_text_with_limit(prompt, max_new_tokens=self.reflect_max_new_tokens),
            self.stop_words,
        ).strip()

    def __call__(self, messages: List[dict]) -> str:
        self.last_step_metadata = None

        draft_response = self._generate_formatted_policy_response(messages, max_new_tokens=self.max_new_tokens)
        draft_action = self._extract_action_only(draft_response)
        if not draft_action:
            self.last_step_metadata = {
                "flow_mode": "draft_only_fallback",
                "draft_response": draft_response,
                "policy_model_path": self.model_path,
                "student_world_model_path": self.student_world_model_path,
                "teacher_world_model_path": self.teacher_world_model_path,
            }
            return draft_response

        student_prediction = self._predict_next_state(messages, draft_action, use_teacher=False)
        teacher_prediction = self._predict_next_state(messages, draft_action, use_teacher=True)

        reflect_messages = self._build_reflect_messages(messages, draft_response, student_prediction)
        reflect_response = self._generate_reflection_response(reflect_messages)
        reflect_action = self._extract_action_only(reflect_response)
        revise_probability = self._extract_revise_probability(reflect_response)
        reflection_confidence = self._reflection_confidence(reflect_response, reflect_action)

        used_revised_action = self._should_use_refined_action(
            draft_action,
            reflect_action,
            revise_probability,
            reflection_confidence,
        )
        if used_revised_action:
            final_action = reflect_action
        else:
            final_action = draft_action
        final_response = self._format_final_action_response(
            final_action,
            reflect_response,
            used_revised_action=used_revised_action,
        )

        executed_teacher_hint = teacher_prediction
        if used_revised_action and self.teacher_world_model is not None:
            executed_teacher_hint = self._predict_next_state(messages, final_action, use_teacher=True)

        self.last_step_metadata = {
            "flow_mode": "draft_student_wm_reflect",
            "draft_response": draft_response,
            "draft_action": draft_action,
            "draft_action_prob": None,
            "pred_next_state_student": student_prediction,
            "pred_next_state_teacher": teacher_prediction,
            "reflect_response": reflect_response,
            "revised_action": reflect_action or draft_action,
            "reflect_action_prob": revise_probability,
            "reflection_confidence": reflection_confidence,
            "revise_probability_threshold": self.revise_probability_threshold,
            "reflection_confidence_threshold": self.reflection_confidence_threshold,
            "final_max_new_tokens": self.final_max_new_tokens,
            "revise_triggered": used_revised_action,
            "final_action": final_action,
            "used_revised_action": used_revised_action,
            "imagined_next_state": student_prediction,
            "wm_teacher_hint": executed_teacher_hint,
            "policy_model_path": self.model_path,
            "student_world_model_path": self.student_world_model_path,
            "teacher_world_model_path": self.teacher_world_model_path,
            "teacher_world_model_shared": self.teacher_world_model_shared,
        }
        return final_response
