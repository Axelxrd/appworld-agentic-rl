#!/usr/bin/env bash
set -euo pipefail

# Compare full-history prompting with the production structured-memory policy
# on one fixed, dev-only manifest. This script never reads a frozen final
# manifest and writes one independent run contract per condition.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"
APPWORLD_ROOT="${APPWORLD_ROOT:?Set APPWORLD_ROOT to the AppWorld runtime root}"
COMPARISON_MANIFEST="${CONTEXT_COMPARISON_MANIFEST:?Set CONTEXT_COMPARISON_MANIFEST to a dev-only manifest}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
MODEL_REVISION="${MODEL_REVISION:-context-comparison}"
TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-$MODEL_PATH}"
OUTPUT_ROOT="${CONTEXT_COMPARISON_OUTPUT:-$ROOT/artifacts/evaluations/context_memory_comparison}"
PORT="${PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-160000}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-50}"
EVAL_MAX_MODEL_TURNS="${EVAL_MAX_MODEL_TURNS:-48}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-2048}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-$((MAX_MODEL_LEN - 4096))}"
MEMORY_TRIGGER_TOKENS="${MEMORY_TRIGGER_TOKENS:-65536}"
MEMORY_TARGET_TOKENS="${MEMORY_TARGET_TOKENS:-49152}"
MEMORY_RECENT_GROUPS="${MEMORY_RECENT_GROUPS:-4}"
START_SCRIPT="${START_SCRIPT:-$ROOT/scripts/start_vllm_server.sh}"
STOP_SCRIPT="${STOP_SCRIPT:-$ROOT/scripts/stop_vllm_server.sh}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export WANDB_MODE="${WANDB_MODE:-offline}"

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
  else
    command -v -- "$candidate"
  fi
}

EVAL_PYTHON="$(resolve_executable "$EVAL_PYTHON")" || { echo "Missing eval Python: $EVAL_PYTHON" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/tokenizer_config.json" ]] || { echo "Missing tokenizer: $TOKENIZER_PATH" >&2; exit 1; }
[[ -d "$APPWORLD_ROOT" ]] || { echo "Missing AppWorld runtime root: $APPWORLD_ROOT" >&2; exit 1; }
[[ -f "$COMPARISON_MANIFEST" ]] || { echo "Missing comparison manifest: $COMPARISON_MANIFEST" >&2; exit 1; }
[[ "$MEMORY_TARGET_TOKENS" -lt "$MEMORY_TRIGGER_TOKENS" ]] || {
  echo "MEMORY_TARGET_TOKENS must be smaller than MEMORY_TRIGGER_TOKENS" >&2; exit 1;
}

mkdir -p "$OUTPUT_ROOT"
ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true
cleanup() { ROOT="$ROOT" bash "$STOP_SCRIPT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" MODEL_NAME="$MODEL_NAME" PORT="$PORT" \
  CUDA_DEVICES="$CUDA_DEVICES" TP_SIZE="$TP_SIZE" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  bash "$START_SCRIPT" >"$OUTPUT_ROOT/vllm.log" 2>&1

run_condition() {
  local condition="$1"
  shift
  local output="$OUTPUT_ROOT/$condition"
  local resume=()
  if [[ -d "$output" && ! -f "$output/run_config.json" && ! -f "$output/raw.jsonl" ]]; then
    rmdir "$output"
  fi
  [[ -f "$output/run_config.json" ]] && resume=(--resume)
  "$EVAL_PYTHON" -m appworld_agentic_rl.cli evaluate-frozen \
    --model "$MODEL_NAME" --model-revision "$MODEL_REVISION" \
    --base-url "http://127.0.0.1:$PORT/v1" --condition "$condition" \
    --manifest "$COMPARISON_MANIFEST" --output-dir "$output" \
    --appworld-root "$APPWORLD_ROOT" --split dev \
    --experiment-name "appworld_context_comparison_$condition" --bc-guard-mode enforce \
    --max-steps "$EVAL_MAX_STEPS" --max-model-turns "$EVAL_MAX_MODEL_TURNS" \
    --temperature 0 --max-tokens "$EVAL_MAX_TOKENS" \
    --disable-thinking \
    --context-budget-mode tokens --tokenizer-path "$TOKENIZER_PATH" \
    --max-context-tokens "$MAX_CONTEXT_TOKENS" \
    --timeout 180 --request-retries 2 --provider-disable-parallel-tool-calls \
    "$@" "${resume[@]}"
  [[ -f "$output/summary.json" ]] || { echo "Missing summary for $condition" >&2; exit 1; }
}

run_condition full_context \
  --no-context-compaction-enable \
  --context-compaction-trigger-tokens "$MEMORY_TRIGGER_TOKENS" \
  --context-compaction-target-tokens "$MEMORY_TARGET_TOKENS" \
  --context-preserve-recent-groups "$MEMORY_RECENT_GROUPS"

run_condition structured_memory \
  --context-compaction-enable \
  --context-compaction-trigger-tokens "$MEMORY_TRIGGER_TOKENS" \
  --context-compaction-target-tokens "$MEMORY_TARGET_TOKENS" \
  --context-preserve-recent-groups "$MEMORY_RECENT_GROUPS"

"$EVAL_PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = (
    "task_success_rate", "clean_task_success_rate", "requirement_pass_rate",
    "execution_error_rate", "infrastructure_error_rate", "collateral_damage_rate",
    "mean_model_turns", "mean_action_attempts", "compaction_rate",
    "post_compaction_name_error_total", "tasks_with_post_compaction_name_error",
)
summaries = {
    name: json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
    for name in ("full_context", "structured_memory")
}
result = {
    "schema_version": "appworld-context-memory-comparison-v1",
    "conditions": summaries,
    "structured_memory_minus_full_context": {
        key: float(summaries["structured_memory"].get(key) or 0.0)
        - float(summaries["full_context"].get(key) or 0.0)
        for key in keys
    },
}
(root / "comparison.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
lines = [
    "# Context Memory Comparison",
    "",
    "| Condition | Task Success | Clean Success | Execution Error | Post-compaction NameError |",
    "|---|---:|---:|---:|---:|",
]
for name, label in (("full_context", "Full context"), ("structured_memory", "Structured memory")):
    summary = summaries[name]
    lines.append(
        "| {label} | {task:.2%} | {clean:.2%} | {execution:.2%} | {name_errors} |".format(
            label=label,
            task=float(summary.get("task_success_rate") or 0.0),
            clean=float(summary.get("clean_task_success_rate") or 0.0),
            execution=float(summary.get("execution_error_rate") or 0.0),
            name_errors=int(summary.get("post_compaction_name_error_total") or 0),
        )
    )
(root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Context-memory comparison complete: $OUTPUT_ROOT/comparison.json and comparison.md"
