#!/usr/bin/env bash
set -euo pipefail

# Evaluate one RL checkpoint on dev enforce only. The caller uses this summary
# for checkpoint selection; frozen final evaluation is deliberately separate.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a checkpoint model directory}"
MODEL_NAME="${MODEL_NAME:-appworld-grpo-dev-candidate}"
MODEL_REVISION="${MODEL_REVISION:-dev-candidate}"
TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-$MODEL_PATH}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for this checkpoint dev results}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/artifacts/evaluations/dev_full_v1/dev.jsonl}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

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
[[ -f "$DEV_MANIFEST" ]] || { echo "Missing dev manifest: $DEV_MANIFEST" >&2; exit 1; }
[[ -d "$APPWORLD_ROOT" ]] || { echo "Missing AppWorld runtime root: $APPWORLD_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true
ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" MODEL_NAME="$MODEL_NAME" \
  PORT="$PORT" CUDA_DEVICES="$CUDA_DEVICES" TP_SIZE="$TP_SIZE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" bash "$START_SCRIPT" >"$OUTPUT_ROOT/vllm.log" 2>&1

run_eval() {
  local label="$1" mode="$2"
  local output="$OUTPUT_ROOT/$label"
  local resume=()
  if [[ -d "$output" && ! -f "$output/run_config.json" && ! -f "$output/raw.jsonl" ]]; then
    rmdir "$output"
  fi
  [[ -f "$output/run_config.json" ]] && resume=(--resume)
  "$EVAL_PYTHON" -m appworld_agentic_rl.cli evaluate-frozen \
    --model "$MODEL_NAME" --model-revision "$MODEL_REVISION" \
    --base-url "http://127.0.0.1:$PORT/v1" --condition "$label" \
    --manifest "$DEV_MANIFEST" --output-dir "$output" \
    --appworld-root "$APPWORLD_ROOT" --split dev \
    --experiment-name "appworld_$label" --bc-guard-mode "$mode" \
    --max-steps "$EVAL_MAX_STEPS" --max-model-turns "$EVAL_MAX_MODEL_TURNS" \
    --temperature 0 --max-tokens "$EVAL_MAX_TOKENS" \
    --disable-thinking \
    --context-budget-mode tokens --tokenizer-path "$TOKENIZER_PATH" \
    --max-context-tokens "$EVAL_MAX_CONTEXT_TOKENS" \
    --context-compaction-trigger-tokens "$MEMORY_TRIGGER_TOKENS" \
    --context-compaction-target-tokens "$MEMORY_TARGET_TOKENS" \
    --timeout 180 --request-retries 2 --provider-disable-parallel-tool-calls "${resume[@]}" \
    >"$OUTPUT_ROOT/${label}.log" 2>&1
}

run_eval dev_enforce enforce
ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true
[[ -f "$OUTPUT_ROOT/dev_enforce/summary.json" ]] || { echo "Missing dev_enforce summary" >&2; exit 1; }
printf '%s\n' "$OUTPUT_ROOT/dev_enforce/summary.json"
