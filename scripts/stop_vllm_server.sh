#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="$ROOT/artifacts/model_server/vllm.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "vLLM is not running"
  exit 0
fi
pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
fi
rm -f "$PID_FILE"
echo "vLLM stopped: pid=$pid"
