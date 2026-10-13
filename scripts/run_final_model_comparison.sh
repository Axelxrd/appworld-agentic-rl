#!/usr/bin/env bash
set -euo pipefail

# Evaluate Base, SFT-final, and RL-best on one frozen manifest, then aggregate
# their summaries into JSON, CSV, and Markdown tables.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON_BIN}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the Qwen3.5-4B Base directory}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the Qwen3.5-4B SFT-final directory}"
RL_MODEL_PATH="${RL_MODEL_PATH:?Set RL_MODEL_PATH to the merged RL-best directory}"
APPWORLD_ROOT="${APPWORLD_ROOT:?Set APPWORLD_ROOT to the AppWorld runtime root}"
FINAL_MANIFEST="${FINAL_MANIFEST:-$ROOT/artifacts/evaluations/final_manifest_v1.jsonl}"
OUTPUT_ROOT="${FINAL_COMPARISON_OUTPUT:-$ROOT/artifacts/evaluations/final_model_comparison}"
EVALUATION_SCRIPT="${FINAL_EVALUATION_SCRIPT:-$ROOT/scripts/evaluate_selected_checkpoint.sh}"

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
  else
    command -v -- "$candidate"
  fi
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="$(resolve_executable "$PYTHON_BIN")" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
EVAL_PYTHON="$(resolve_executable "$EVAL_PYTHON")" || { echo "Missing eval Python: $EVAL_PYTHON" >&2; exit 1; }
[[ -d "$APPWORLD_ROOT" ]] || { echo "Missing AppWorld runtime root: $APPWORLD_ROOT" >&2; exit 1; }
[[ -f "$FINAL_MANIFEST" ]] || { echo "Missing frozen final manifest: $FINAL_MANIFEST" >&2; exit 1; }
RL_MODEL_PATH="$("$PYTHON_BIN" -m appworld_agentic_rl.checkpointing --checkpoint-dir "$RL_MODEL_PATH")"
for model_path in "$BASE_MODEL_PATH" "$SFT_MODEL_PATH" "$RL_MODEL_PATH"; do
  [[ -f "$model_path/config.json" ]] || { echo "Missing model config: $model_path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT"

evaluate_model() {
  local key="$1" model_path="$2" model_name="$3" revision="$4" tokenizer_path="$5"
  local model_output="$OUTPUT_ROOT/$key"
  echo "Evaluating $model_name on frozen final"
  ROOT="$ROOT" MODEL_PATH="$model_path" MODEL_NAME="$model_name" \
    MODEL_REVISION="$revision" APPWORLD_ROOT="$APPWORLD_ROOT" \
    EVAL_TOKENIZER_PATH="$tokenizer_path" \
    EVAL_PYTHON="$EVAL_PYTHON" \
    FINAL_MANIFEST="$FINAL_MANIFEST" OUTPUT_ROOT="$model_output" \
    bash "$EVALUATION_SCRIPT"
  [[ -f "$model_output/final_enforce/summary.json" ]] || {
    echo "Missing final summary for $model_name" >&2; exit 1;
  }
}

evaluate_model base "$BASE_MODEL_PATH" "Qwen3.5-4B-Base" "base-final" "$BASE_MODEL_PATH"
evaluate_model sft "$SFT_MODEL_PATH" "Qwen3.5-4B-SFT-final" "sft-final" "$SFT_MODEL_PATH"
evaluate_model rl "$RL_MODEL_PATH" "Qwen3.5-4B-RL-best" "rl-best" "$SFT_MODEL_PATH"

"$PYTHON_BIN" -m appworld_agentic_rl.result_reporting \
  --base-summary "$OUTPUT_ROOT/base/final_enforce/summary.json" \
  --sft-summary "$OUTPUT_ROOT/sft/final_enforce/summary.json" \
  --rl-summary "$OUTPUT_ROOT/rl/final_enforce/summary.json" \
  --output-dir "$OUTPUT_ROOT"

echo "Final model comparison complete: $OUTPUT_ROOT/comparison.md"
