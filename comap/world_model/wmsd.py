from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_kl_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level WMSD loss on target positions.

    This is the world-model analogue of the SDPO logit distillation hook in
    `integrations/sdpo/verl/trainer/ppo/core_algos.py`: real next-state CE
    anchors the transition, while the teacher distribution supplies soft
    guidance on the same generated/teacher-forced prefix.
    """

    shift_student = student_logits[..., :-1, :].contiguous() / temperature
    shift_teacher = teacher_logits[..., :-1, :].contiguous() / temperature
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels.ne(-100)
    if not torch.any(mask):
        return student_logits.new_zeros(())

    student_log_probs = F.log_softmax(shift_student, dim=-1)
    teacher_probs = F.softmax(shift_teacher, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

