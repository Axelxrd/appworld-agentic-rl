#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a verl FSDP checkpoint directory}"
TARGET_DIR="${TARGET_DIR:?Set TARGET_DIR to the merged HuggingFace output directory}"

[[ -f "$CHECKPOINT_DIR/fsdp_config.json" ]] || { echo "Missing fsdp_config.json: $CHECKPOINT_DIR" >&2; exit 1; }
world_size="$($PYTHON_BIN - "$CHECKPOINT_DIR/fsdp_config.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["world_size"])
PY
)"
for ((rank=0; rank<world_size; rank++)); do
  [[ -f "$CHECKPOINT_DIR/model_world_size_${world_size}_rank_${rank}.pt" ]] || {
    echo "Missing FSDP rank file $rank in $CHECKPOINT_DIR" >&2; exit 1;
  }
done
[[ ! -e "$TARGET_DIR" ]] || { echo "Target already exists: $TARGET_DIR" >&2; exit 1; }

"$PYTHON_BIN" -m verl.model_merger merge \
  --backend fsdp --local_dir "$CHECKPOINT_DIR" --target_dir "$TARGET_DIR"
[[ -f "$TARGET_DIR/config.json" ]] || { echo "Merged model has no config.json" >&2; exit 1; }
compgen -G "$TARGET_DIR/*.safetensors" >/dev/null || {
  echo "Merged model has no safetensors weights" >&2; exit 1;
}
echo "Merged HuggingFace model: $TARGET_DIR"
