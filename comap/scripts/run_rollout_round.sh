#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ETO_ROOT="${PROJECT_ROOT}/third_party/agent-eto"
CONDA_ENV=""
TASK_NAME="alfworld"
EXP_CONFIG="alfworld"
AGENT_CONFIG_BASENAME="fastchat_round"
AGENT_BACKEND="fastchat"
SPLIT="train"
MODEL_PATH=""
MODEL_NAME=""
WM_STUDENT_MODEL_PATH=""
WM_TEACHER_MODEL_PATH=""
ROUND_ID="0"
OUTPUT_DIR=""
CONTROLLER_PORT="21001"
WORKER_PORT="21002"
WAIT_SECONDS="45"
OVERRIDE="false"
MAX_NEW_TOKENS="64"
REFLECT_MAX_NEW_TOKENS="128"
FINAL_MAX_NEW_TOKENS="16"
WM_MAX_NEW_TOKENS="192"
TEMPERATURE="0.7"
TOP_P="0.9"
REVISE_PROBABILITY_THRESHOLD="0.5"
REFLECTION_CONFIDENCE_THRESHOLD="0.6"
ENABLE_THINKING="false"
PART_NUM="1"
PART_IDX="-1"
DEBUG="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --eto-root) ETO_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --exp-config) EXP_CONFIG="$2"; shift 2 ;;
    --agent-config) AGENT_CONFIG_BASENAME="$2"; shift 2 ;;
    --agent-backend) AGENT_BACKEND="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --wm-student-model-path) WM_STUDENT_MODEL_PATH="$2"; shift 2 ;;
    --wm-teacher-model-path) WM_TEACHER_MODEL_PATH="$2"; shift 2 ;;
    --round-id) ROUND_ID="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --controller-port) CONTROLLER_PORT="$2"; shift 2 ;;
    --worker-port) WORKER_PORT="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --reflect-max-new-tokens) REFLECT_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --final-max-new-tokens) FINAL_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --wm-max-new-tokens) WM_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --revise-probability-threshold) REVISE_PROBABILITY_THRESHOLD="$2"; shift 2 ;;
    --reflection-confidence-threshold) REFLECTION_CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
    --part-num) PART_NUM="$2"; shift 2 ;;
    --part-idx) PART_IDX="$2"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING="true"; shift 1 ;;
    --debug) DEBUG="true"; shift 1 ;;
    --override) OVERRIDE="true"; shift 1 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model-path is required" >&2
  exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$(basename "$MODEL_PATH")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$SPLIT" == "test" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval_round_${ROUND_ID}"
  else
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/round_${ROUND_ID}_rollout"
  fi
fi

mkdir -p "$OUTPUT_DIR"
TMP_AGENT_DIR="$(mktemp -d)"
AGENT_CONFIG_PATH="${TMP_AGENT_DIR}/${AGENT_CONFIG_BASENAME}.json"
EXP_SUFFIX="_round_${ROUND_ID}_${SPLIT}"
if [[ "$PART_NUM" != "1" || "$PART_IDX" != "-1" ]]; then
  EXP_SUFFIX="${EXP_SUFFIX}_p${PART_IDX}of${PART_NUM}"
fi
if [[ "$DEBUG" == "true" ]]; then
  EXP_SUFFIX="${EXP_SUFFIX}_debug"
fi

cleanup() {
  if [[ -n "${WORKER_PID:-}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill "${WORKER_PID}" 2>/dev/null || true
    wait "${WORKER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CONTROLLER_PID:-}" ]] && kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
    kill "${CONTROLLER_PID}" 2>/dev/null || true
    wait "${CONTROLLER_PID}" 2>/dev/null || true
  fi
  rm -rf "$TMP_AGENT_DIR"
}
trap cleanup EXIT

run_in_env() {
  if [[ -n "$CONDA_ENV" ]]; then
    conda run -n "$CONDA_ENV" "$@"
  else
    "$@"
  fi
}

if [[ "$AGENT_BACKEND" == "fastchat" ]]; then
  cat > "$AGENT_CONFIG_PATH" <<EOF
{
  "agent_class": "FastChatAgent",
  "config": {
    "controller_address": "http://localhost:${CONTROLLER_PORT}",
    "model_name": "${MODEL_NAME}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P}
  }
}
EOF
elif [[ "$AGENT_BACKEND" == "local_hf" ]]; then
  cat > "$AGENT_CONFIG_PATH" <<EOF
{
  "agent_class": "HuggingFaceLocalAgent",
  "config": {
    "model_path": "${MODEL_PATH}",
    "model_name": "${MODEL_NAME}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P},
    "enable_thinking": ${ENABLE_THINKING}
  }
}
EOF
elif [[ "$AGENT_BACKEND" == "coevolving_local_hf" ]]; then
  if [[ -z "$WM_STUDENT_MODEL_PATH" ]]; then
    echo "--wm-student-model-path is required when --agent-backend=coevolving_local_hf" >&2
    exit 1
  fi
  cat > "$AGENT_CONFIG_PATH" <<EOF
{
  "agent_class": "CoevolvingLocalAgent",
  "config": {
    "model_path": "${MODEL_PATH}",
    "model_name": "${MODEL_NAME}",
    "student_world_model_path": "${WM_STUDENT_MODEL_PATH}",
    "teacher_world_model_path": "${WM_TEACHER_MODEL_PATH}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "reflect_max_new_tokens": ${REFLECT_MAX_NEW_TOKENS},
    "final_max_new_tokens": ${FINAL_MAX_NEW_TOKENS},
    "wm_max_new_tokens": ${WM_MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P},
    "revise_probability_threshold": ${REVISE_PROBABILITY_THRESHOLD},
    "reflection_confidence_threshold": ${REFLECTION_CONFIDENCE_THRESHOLD},
    "enable_thinking": ${ENABLE_THINKING}
  }
}
EOF
else
  echo "Unsupported --agent-backend: ${AGENT_BACKEND}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${ETO_ROOT}:${PYTHONPATH:-}"

if [[ "$AGENT_BACKEND" == "fastchat" ]]; then
  (
    cd "$ETO_ROOT"
    run_in_env python -u -m fastchat.serve.controller --port "$CONTROLLER_PORT" > "${OUTPUT_DIR}/controller.log" 2>&1
  ) &
  CONTROLLER_PID=$!
  sleep 5

  (
    cd "$ETO_ROOT"
    run_in_env python -u -m fastchat.serve.model_worker \
      --model-path "$MODEL_PATH" \
      --model-names "$MODEL_NAME" \
      --port "$WORKER_PORT" \
      --worker-address "http://localhost:${WORKER_PORT}" \
      --controller-address "http://localhost:${CONTROLLER_PORT}" > "${OUTPUT_DIR}/model_worker.log" 2>&1
  ) &
  WORKER_PID=$!

  sleep "$WAIT_SECONDS"
fi

EVAL_CMD=(python -m eval_agent.main
  --agent_path "$TMP_AGENT_DIR"
  --agent_config "$AGENT_CONFIG_BASENAME"
  --exp_config "$EXP_CONFIG"
  --exp_name "$EXP_SUFFIX"
  --split "$SPLIT"
  --part_num "$PART_NUM"
  --part_idx "$PART_IDX"
  --model_name "$MODEL_NAME")

if [[ "$OVERRIDE" == "true" ]]; then
  EVAL_CMD+=(--override)
fi
if [[ "$DEBUG" == "true" ]]; then
  EVAL_CMD+=(--debug)
fi

(
  cd "$ETO_ROOT"
  run_in_env "${EVAL_CMD[@]}" | tee "${OUTPUT_DIR}/eval_stdout.log"
)

ETO_OUTPUT_DIR="${ETO_ROOT}/outputs/${MODEL_NAME}/${EXP_CONFIG}${EXP_SUFFIX}"
python - <<PY
from pathlib import Path
import shutil

src = Path(r"""${ETO_OUTPUT_DIR}""").resolve()
dst = Path(r"""${OUTPUT_DIR}""").resolve() / "eto_raw"
if not src.exists():
    raise SystemExit(f"Expected ETO output directory not found: {src}")
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"Copied raw ETO outputs to {dst}")
PY
