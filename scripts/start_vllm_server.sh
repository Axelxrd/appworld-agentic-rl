#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the downloaded model directory}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-106496}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DTYPE="${DTYPE:-auto}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/v1/models}"
LOG_DIR="$ROOT/artifacts/model_server"
PID_FILE="$LOG_DIR/vllm.pid"
LOG_FILE="$LOG_DIR/vllm.log"

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
  else
    command -v -- "$candidate"
  fi
}

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if ! VLLM_BIN="$(resolve_executable "$VLLM_BIN")"; then
  echo "vLLM is not installed: $VLLM_BIN" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "vLLM is already running with PID $old_pid" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

args=(
  serve "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
)
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  args+=(--language-model-only)
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$ROOT"
nohup "$VLLM_BIN" "${args[@]}" >"$LOG_FILE" 2>&1 &
server_pid=$!
echo "$server_pid" >"$PID_FILE"
echo "vLLM started: pid=$server_pid model=$MODEL_NAME host=$HOST port=$PORT cuda=$CUDA_DEVICES"
echo "log=$LOG_FILE"

deadline=$((SECONDS + STARTUP_TIMEOUT))
while ! curl --fail --silent --show-error "$HEALTH_URL" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM is still running but did not become ready within ${STARTUP_TIMEOUT}s." >&2
    echo "Inspect the log with: tail -f $LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done

echo "vLLM is ready: $HEALTH_URL"
curl --fail --silent --show-error "$HEALTH_URL"
echo
