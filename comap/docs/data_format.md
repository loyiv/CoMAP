# Data Format

The bridge code uses a step-level JSONL format. Each row corresponds to one ALFWorld decision step after converting an ETO episode JSON file.

Important fields:

- `goal`: task instruction.
- `history_messages`: conversation history before the current action.
- `observation`: current textual observation.
- `action`: final executed action.
- `draft_action`: policy draft before reflection.
- `revised_action`: reflected action candidate.
- `revise_triggered`: whether the action gate adopted the reflected action.
- `pred_next_state_student`: one-step next-state prediction from the student world model.
- `pred_next_state_teacher`: teacher-mode prediction when available.
- `next_observation`: real next observation returned by ALFWorld.
- `feedback_text`: compact environment feedback used by policy training.
- `wm_teacher_hint`: privileged teacher-side hint for world-model distillation.

The SDPO builder converts these records into rows with:

- `prompt`: chat messages used as policy input.
- `reward_model.ground_truth`: target ALFWorld action.
- `extra_info.sample_origin`: one of `expert_anchor`, `revise_positive`, `keep_draft`, `hq_reflect`, `hq_draft_distill`, or `fallback_transition`.
- `extra_info.refinement_label`: `1` for accepted revision targets and `0` for keep-draft/anchor/fallback targets.
