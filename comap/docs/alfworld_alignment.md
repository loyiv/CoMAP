# ALFWorld Paper Alignment

This note records how the minimal code maps to the ALFWorld part of the paper.

## Benchmark

ALFWorld is used as a text-only embodied household environment. The agent receives natural-language observations and emits admissible text actions. The paper reports the standard splits:

- train: 3,119 episodes
- test-seen: 140 episodes
- test-unseen: 171 episodes

The code only keeps ALFWorld-related prompts, rewards, rollout serialization, and training bridges.

## COMAP Components

`integrations/eto/eval_agent/agents/coevolving_local_agent.py` implements the decision loop:

- draft generation by the local policy,
- one-step future-state prediction by the student world model,
- future-aware reflection,
- action-gated execution.

`world_model/` implements one-step textual transition modeling. The prompt follows the paper's world-model template: current textual state plus action, predict only the next state.

`bridge/eto_to_round_buffer.py` converts ALFWorld episode JSON files into step-level records containing draft action, revised action, predicted next state, real next state, reward, and success.

`bridge/build_sdpo_buffer.py` builds policy-side rows for expert anchors, revise-positive supervision, keep-draft supervision, high-quality reflection, draft distillation, and fallback transition supervision.

`integrations/sdpo/` contains the SDPO/verl files that carry the policy-side self-distillation and auxiliary supervised-loss hooks.

## Training Defaults

The sanitized config in `training/configs/eto_alfworld_sdpo.yaml` follows the paper defaults:

- 3 co-evolving rounds
- max sequence length 2048
- AdamW with cosine decay and 0.03 warm-up ratio
- LoRA rank 8, alpha 16, dropout 0.05
- world-model training: 3 epochs, LR 2e-5, batch size 1, grad accumulation 16
- WMSD weight 0.5, EMA momentum 0.99, world-state threshold 0.6
- policy training: 3 epochs, LR 2e-5, batch size 1, grad accumulation 16
- revise threshold 0.5, reflection-confidence threshold 0.6
- draft/reflection/final token budgets 64/128/16
- temperature 0.7, top-p 0.9

## Scope

The repository is designed as a minimal paper-code release, not as a full benchmark mirror. It excludes generated buffers and trained checkpoints but keeps the code paths that produced those artifacts.
