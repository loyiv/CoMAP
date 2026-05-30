# Integration Notes

The original experiments used local copies of ETO for ALFWorld evaluation and SDPO/verl for policy optimization. This minimal release does not vendor those full repositories.

## ETO

Copy the files under `integrations/eto/eval_agent/` into the matching ETO `eval_agent` package, or keep them as a small overlay on `PYTHONPATH`.

The important ALFWorld-facing class is:

```text
integrations/eto/eval_agent/agents/coevolving_local_agent.py
```

It adds `CoevolvingLocalAgent`, which performs draft -> world model -> reflection -> action gate.

## SDPO/verl

The files under `integrations/sdpo/` are patch-bearing files from the local SDPO/verl training stack. They contain:

- `SelfDistillationConfig`
- `compute_self_distillation_loss`
- auxiliary supervised action loss hooks
- SDPO trainer config defaults used by the ALFWorld policy update

For a full run, transplant these files or port the corresponding snippets into the installed SDPO/verl version.

