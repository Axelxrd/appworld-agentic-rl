#!/usr/bin/env bash
set -euo pipefail

# Run frozen final evaluation for a checkpoint already selected on dev_enforce.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the selected checkpoint model directory}"
MODEL_NAME="${MODEL_NAME:-appworld-sft-best}"
MODEL_REVISION="${MODEL_REVISION:-sft-selected}"
TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-$MODEL_PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
APPWORLD_ROOT="${APPWORLD_ROOT:?Set APPWORLD_ROOT to the AppWorld runtime root}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
PORT="${PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-160000}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-50}"
EVAL_MAX_MODEL_TURNS="${EVAL_MAX_MODEL_TURNS:-48}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-2048}"
EVAL_MAX_CONTEXT_TOKENS="${EVAL_MAX_CONTEXT_TOKENS:-$((MAX_MODEL_LEN - 4096))}"
MEMORY_TRIGGER_TOKENS="${MEMORY_TRIGGER_TOKENS:-65536}"
MEMORY_TARGET_TOKENS="${MEMORY_TARGET_TOKENS:-49152}"
START_SCRIPT="${START_SCRIPT:-$ROOT/scripts/start_vllm_server.sh}"
STOP_SCRIPT="${STOP_SCRIPT:-$ROOT/scripts/stop_vllm_server.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/evaluations/sft_selected}"
FINAL_MANIFEST="${FINAL_MANIFEST:-$ROOT/artifacts/evaluations/final_manifest_v1.jsonl}"
WANDB_EVAL_LOG="${WANDB_EVAL_LOG:-1}"
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${WANDB_PROJECT:-appworld-agentic-rl}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
  else
    command -v -- "$candidate"
  fi
}

EVAL_PYTHON="$(resolve_executable "$EVAL_PYTHON")" || { echo "Missing eval python: $EVAL_PYTHON" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/tokenizer_config.json" ]] || { echo "Missing tokenizer config: $TOKENIZER_PATH" >&2; exit 1; }
[[ -d "$APPWORLD_ROOT" ]] || { echo "Missing AppWorld runtime root: $APPWORLD_ROOT" >&2; exit 1; }
[[ -f "$FINAL_MANIFEST" ]] || { echo "Missing final manifest: $FINAL_MANIFEST" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true
ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" MODEL_NAME="$MODEL_NAME" \
  PORT="$PORT" CUDA_DEVICES="$CUDA_DEVICES" TP_SIZE="$TP_SIZE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" bash "$START_SCRIPT"

run_eval() {
  local label="$1" manifest="$2" split="$3" mode="$4"
  local output="$OUTPUT_ROOT/$label"
  local resume=()
  if [[ -d "$output" && ! -f "$output/run_config.json" && ! -f "$output/raw.jsonl" ]]; then
    rmdir "$output"
  fi
  [[ -f "$output/run_config.json" ]] && resume=(--resume)
  "$EVAL_PYTHON" -m appworld_agentic_rl.cli evaluate-frozen \
    --model "$MODEL_NAME" --model-revision "$MODEL_REVISION" \
    --base-url "http://127.0.0.1:$PORT/v1" --condition "$label" \
    --manifest "$manifest" --output-dir "$output" \
    --appworld-root "$APPWORLD_ROOT" --split "$split" \
    --experiment-name "appworld_$label" --bc-guard-mode "$mode" \
    --max-steps "$EVAL_MAX_STEPS" --max-model-turns "$EVAL_MAX_MODEL_TURNS" \
    --temperature 0 --max-tokens "$EVAL_MAX_TOKENS" \
    --disable-thinking \
    --context-budget-mode tokens --tokenizer-path "$TOKENIZER_PATH" \
    --max-context-tokens "$EVAL_MAX_CONTEXT_TOKENS" \
    --context-compaction-trigger-tokens "$MEMORY_TRIGGER_TOKENS" \
    --context-compaction-target-tokens "$MEMORY_TARGET_TOKENS" \
    --timeout 180 --request-retries 2 --provider-disable-parallel-tool-calls "${resume[@]}"
  if [[ "$WANDB_EVAL_LOG" == "1" && -f "$output/summary.json" ]]; then
    "$EVAL_PYTHON" - "$output/summary.json" "$label" "$WANDB_EVAL_PROJECT" <<'PY'
import json
import sys

try:
    import wandb
    summary = json.load(open(sys.argv[1], encoding="utf-8"))
    run = wandb.init(project=sys.argv[3], name="appworld-eval-" + sys.argv[2], job_type="evaluation", reinit="finish")
    run.log({
        "evaluation/task_success_rate": summary.get("task_success_rate", 0.0),
        "evaluation/task_success_rate_over_expected": summary.get("task_success_rate_over_expected", 0.0),
        "evaluation/clean_task_success_rate": summary.get("clean_task_success_rate", 0.0),
        "evaluation/clean_task_success_rate_over_expected": summary.get("clean_task_success_rate_over_expected", 0.0),
        "evaluation/requirement_pass_rate": summary.get("requirement_pass_rate"),
        "evaluation/infrastructure_error_rate": summary.get("infrastructure_error_rate", 0.0),
        "evaluation/execution_error_rate": summary.get("execution_error_rate", 0.0),
        "evaluation/collateral_damage_rate": summary.get("collateral_damage_rate", 0.0),
        "evaluation/mean_model_turns": summary.get("mean_model_turns"),
        "evaluation/mean_action_attempts": summary.get("mean_action_attempts"),
        "evaluation/mean_environment_steps": summary.get("mean_environment_steps"),
        "evaluation/compaction_rate": summary.get("compaction_rate", 0.0),
        "evaluation/post_compaction_name_error_total": summary.get("post_compaction_name_error_total", 0),
        "evaluation/tasks_with_post_compaction_name_error": summary.get("tasks_with_post_compaction_name_error", 0),
        "evaluation/successful_tasks": summary.get("successful_tasks", 0),
        "evaluation/completed_tasks": summary.get("completed_tasks", 0),
        "evaluation/infrastructure_error_tasks": summary.get("infrastructure_error_tasks", 0),
    })
    run.finish()
except Exception as exc:
    print("W&B evaluation logging skipped: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
PY
  fi
}

run_eval final_enforce "$FINAL_MANIFEST" test_challenge enforce
ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true
echo "Selected checkpoint evaluation complete: $OUTPUT_ROOT"
