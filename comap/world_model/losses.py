from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels.ne(-100)
    if not torch.any(mask):
        return 0.0
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels) & mask
    return correct.sum().item() / mask.sum().item()


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
