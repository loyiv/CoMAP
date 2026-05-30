from __future__ import annotations

from .schema import RolloutStepRecord, messages_to_text


def build_student_prompt_text(record: RolloutStepRecord) -> str:
    return record.history_text


def build_teacher_prompt_preview(record: RolloutStepRecord) -> str:
    base = messages_to_text(record.history_messages)
    if record.feedback_text:
        return (
            f"{base}\n\n"
            "The following is feedback from your unsuccessful earlier attempt:\n\n"
            f"{record.feedback_text}\n\n"
            "Correctly solve the original question."
        )
    return base
