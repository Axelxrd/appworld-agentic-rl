#!/usr/bin/env bash
set -euo pipefail

# Full one-epoch AppWorld GRPO run, dev-based checkpoint selection, and frozen-test evaluation.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${APPWORLD_MODEL:-$ROOT/models/Qwen3.5-4B-sft-final}"
TRAIN="${APPWORLD_GRPO_TRAIN:-$ROOT/artifacts/appworld_grpo/train_custom.parquet}"
VAL="${APPWORLD_GRPO_VAL:-$ROOT/artifacts/appworld_grpo/dev_custom.parquet}"
OUTPUT="${APPWORLD_GRPO_OUTPUT:-$ROOT/artifacts/appworld_grpo/checkpoints_formal}"
CONFIG="${APPWORLD_GRPO_CONFIG:-appworld_grpo_lightweight_event_credit_160k}"
LOG_DIR="${APPWORLD_LOG_DIR:-$ROOT/logs}"
TRAIN_LOG="${APPWORLD_TRAIN_LOG:-$LOG_DIR/appworld_grpo_formal.log}"
EVAL_OUTPUT="${APPWORLD_EVAL_OUTPUT:-$ROOT/artifacts/evaluations/appworld_grpo_formal}"
DEV_SELECTION_OUTPUT="${APPWORLD_DEV_SELECTION_OUTPUT:-$EVAL_OUTPUT/dev_checkpoint_selection}"
DEV_SELECTION_SCRIPT="${APPWORLD_DEV_SELECTION_SCRIPT:-$ROOT/scripts/evaluate_checkpoint_dev.sh}"

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
  else
    command -v -- "$candidate"
  fi
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-appworld-agentic-rl}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-appworld-grpo-formal}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Runtime watchdogs are environment variables because veRL's AgentLoopConfig
# is a strict dataclass and rejects project-specific YAML keys.
export APPWORLD_ROLLOUT_TIMEOUT="${APPWORLD_ROLLOUT_TIMEOUT:-1200}"
export APPWORLD_GROUP_WAIT_TIMEOUT="${APPWORLD_GROUP_WAIT_TIMEOUT:-1500}"

PYTHON_BIN="$(resolve_executable "$PYTHON_BIN")" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$TRAIN" ]] || { echo "Missing train parquet: $TRAIN" >&2; exit 1; }
[[ -f "$VAL" ]] || { echo "Missing validation parquet: $VAL" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is not set. Export it or use WANDB_MODE=offline." >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run passed: config=$CONFIG train=$TRAIN val=$VAL model=$MODEL output=$OUTPUT dev_selection=$DEV_SELECTION_OUTPUT eval=$EVAL_OUTPUT wandb=$WANDB_MODE"
  exit 0
fi
mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "Starting formal AppWorld GRPO: config=$CONFIG output=$OUTPUT"
APPWORLD_MODEL="$MODEL" APPWORLD_GRPO_TRAIN="$TRAIN" APPWORLD_GRPO_VAL="$VAL" \
  APPWORLD_GRPO_OUTPUT="$OUTPUT" APPWORLD_GRPO_CONFIG="$CONFIG" \
  "$PYTHON_BIN" "$ROOT/scripts/train_appworld_grpo_160k.py" >"$TRAIN_LOG" 2>&1

prepare_model_dir() {
  local step_dir="$1"
  "$PYTHON_BIN" -m appworld_agentic_rl.checkpointing --checkpoint-dir "$step_dir"
}

mapfile -t checkpoint_steps < <(find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V)
[[ "${#checkpoint_steps[@]}" -gt 0 ]] || { echo "No checkpoint found under $OUTPUT" >&2; exit 1; }
mkdir -p "$DEV_SELECTION_OUTPUT"
selection_rows=()
for step_name in "${checkpoint_steps[@]}"; do
  step="${step_name#global_step_}"
  if ! model_dir="$(prepare_model_dir "$OUTPUT/$step_name")"; then
    echo "step=$step could not be prepared for evaluation; skipping" >&2
    continue
  fi
  candidate_output="$DEV_SELECTION_OUTPUT/step_$step"
  echo "Evaluating dev for checkpoint step=$step model=$model_dir"
  summary_path="$(ROOT="$ROOT" MODEL_PATH="$model_dir" MODEL_NAME="appworld-grpo-step-$step" \
    MODEL_REVISION="appworld-grpo-dev-step-$step" OUTPUT_ROOT="$candidate_output" \
    EVAL_TOKENIZER_PATH="$MODEL" \
  EVAL_MAX_MODEL_TURNS=48 bash "$DEV_SELECTION_SCRIPT")"
  selection_rows+=("$step|$summary_path")
done
[[ "${#selection_rows[@]}" -gt 0 ]] || { echo "No checkpoint could be evaluated on dev" >&2; exit 1; }

best_json="$DEV_SELECTION_OUTPUT/ranking.json"
BEST_JSON="$best_json" SELECTION_ROOT="$DEV_SELECTION_OUTPUT" \
  "$PYTHON_BIN" - "${selection_rows[@]}" <<'PY'
import json, os, sys
from pathlib import Path

rows = []
for item in sys.argv[1:]:
    step, summary_path = item.split("|", 1)
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    score = (
        float(data.get("task_success_rate_over_expected", data.get("task_success_rate", 0.0)) or 0.0),
        float(data.get("clean_task_success_rate_over_expected", data.get("clean_task_success_rate", 0.0)) or 0.0),
        float(data.get("requirement_pass_rate", 0.0) or 0.0),
        -float(data.get("infrastructure_error_rate", 0.0) or 0.0),
        -float(data.get("collateral_damage_rate", 0.0) or 0.0),
        -int(step),
    )
    rows.append({"step": int(step), "summary": summary_path, "score": list(score), "metrics": data})
rows.sort(key=lambda row: tuple(row["score"]), reverse=True)
out = {"selection_metric": "task_success_rate_over_expected", "ranking": rows, "best": rows[0]}
Path(os.environ["BEST_JSON"]).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(os.environ["SELECTION_ROOT"], "best.json").write_text(json.dumps({"best_step": rows[0]["step"], "score": rows[0]["score"], "summary": rows[0]["summary"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
best_step="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["step"])' "$best_json")"
best_name="global_step_$best_step"
MODEL_PATH="$(prepare_model_dir "$OUTPUT/$best_name")" || { echo "Best checkpoint model missing: $best_name" >&2; exit 1; }
echo "Evaluating frozen final for best checkpoint step=$best_step model=$MODEL_PATH"
ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" MODEL_NAME="appworld-grpo-$best_name" \
  MODEL_REVISION="appworld-grpo-formal-$best_name" OUTPUT_ROOT="$EVAL_OUTPUT" \
  EVAL_TOKENIZER_PATH="$MODEL" \
  EVAL_MAX_MODEL_TURNS=48 bash "$ROOT/scripts/evaluate_selected_checkpoint.sh"
cp "$DEV_SELECTION_OUTPUT/best.json" "$EVAL_OUTPUT/best_checkpoint.json"
echo "Formal GRPO training and evaluation complete. Best step=$best_step. Train log: $TRAIN_LOG"
