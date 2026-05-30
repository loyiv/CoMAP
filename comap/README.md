# COMAP: ALFWorld Minimal Release

This repository is a minimal ALFWorld-focused code release for **COMAP: Co-Evolving World Models and Agent Policies for LLM Agents**.

The code is distilled from the internal experiment workspace used for the paper. It keeps the parts needed to describe and reproduce the ALFWorld pipeline at code level:

- one-step textual world model training and inference,
- draft action generation followed by world-model lookahead,
- future-aware reflection with an action gate,
- rollout conversion into world-model and policy training buffers,
- SDPO-style policy-side self-distillation hooks,
- round orchestration for the closed-loop policy/world-model update.

The release intentionally excludes checkpoints, generated buffers, raw rollout outputs, full third-party repositories, and machine-specific paths.

## ALFWorld Setting

The paper evaluates ALFWorld as a text-based embodied household benchmark with six compositional task families: PICK, CLEAN, HEAT, COOL, LOOK, and PICK2.

| Split | Size |
| --- | ---: |
| Train | 3,119 |
| Test-Seen | 140 |
| Test-Unseen | 171 |

The trainable COMAP setting uses Qwen3-4B or Qwen3-8B as both policy backbone and textual world-model backbone. The ALFWorld code path uses the same tokenizer template for both modules.

## Paper Defaults Reflected Here

| Group | Value |
| --- | --- |
| Maximum sequence length | 2048 |
| Optimizer / scheduler | AdamW / cosine decay |
| Warm-up ratio | 0.03 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Co-evolving rounds | 3 |
| World-model epochs / LR | 3 / 2e-5 |
| World-model batch / grad accumulation | 1 / 16 |
| WMSD weight eta | 0.5 |
| EMA momentum mu | 0.99 |
| World-state threshold tau_wm | 0.6 |
| World-model max new tokens | 192 |
| Policy epochs / LR | 3 / 2e-5 |
| Online reflection weight alpha | 1.0 |
| BCE weight beta | 0.01 |
| Revise threshold tau_p | 0.5 |
| Reflection-confidence threshold tau_q | 0.6 |
| Draft / reflection / final action tokens | 64 / 128 / 16 |
| Decoding temperature / top-p | 0.7 / 0.9 |

See `training/configs/eto_alfworld_sdpo.yaml` for the sanitized ALFWorld config.

## Repository Layout

```text
bridge/                 Rollout JSON -> round buffers -> SDPO parquet rows.
common/                 Project layout and JSON/JSONL helpers.
orchestrator/           One-round COMAP pipeline manager.
training/               ALFWorld reward and SDPO launch wrapper.
world_model/            Textual world model dataset, loading, training, inference.
scripts/                Shell entry points used by the orchestrator.
integrations/eto/       Minimal ETO-side agent patches for COMAP reflection.
integrations/sdpo/      Patch-bearing SDPO/verl files used by policy training.
docs/                   Method and data-format notes.
examples/               Tiny schema examples, not paper data.
```

## Method Flow

At each ALFWorld step, COMAP follows the paper's decision path:

1. The policy drafts an action from the current textual state.
2. The student world model predicts the one-step future state for the draft action.
3. The policy reflects over the draft action and predicted future state.
4. The action gate executes the revised action only when the revise probability, reflection confidence, and canonical action difference pass the thresholds.
5. The resulting on-policy transition is serialized for world-model update and policy-side reflection learning.

The world model is updated from real next observations plus teacher-guided distillation targets. The policy side uses SDPO-style self-distillation and auxiliary supervised rows for expert anchors, revise-positive samples, keep-draft samples, high-quality reflection, draft distillation, and fallback transitions.

## What Is Not Included

This is not a full one-command reproduction package. The following are intentionally omitted:

- ALFWorld game files and benchmark installation,
- Qwen checkpoints and trained LoRA adapters,
- generated rollout outputs and buffers,
- the complete ETO and SDPO/verl repositories.

The files under `integrations/` show the concrete local modifications that were used with those external projects.

## Minimal Usage Sketch

```bash
export COMAP_ROOT=/path/to/CoMAP
export PYTHONPATH=$COMAP_ROOT:$PYTHONPATH

python -m orchestrator.run_round \
  --round-id 0 \
  --config training/configs/eto_alfworld_sdpo.yaml \
  --dry-run
```

For a real run, install ALFWorld, place the ETO and SDPO/verl dependencies as described in `docs/integration_notes.md`, and replace the model/checkpoint placeholders in the config.
