#!/usr/bin/env bash
set -euo pipefail

# Set MODEL_PATH to the local Qwen3.5-4B directory on the new server.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
# Leave this unset to use the verl package installed in the active venv.
# Set it explicitly only when testing another local verl checkout.
VERL_ROOT="${VERL_ROOT:-}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-4B directory}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/artifacts/sft/appworld_teacher_clean_all_v2/train.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/checkpoints/appworld_sft_qwen3.5_4b}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAX_LENGTH="${MAX_LENGTH:-106496}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NPROC_PER_NODE * MICRO_BATCH_PER_GPU))}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-true}"
FUSED_KERNEL_BACKEND="${FUSED_KERNEL_BACKEND:-torch}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[model]}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-true}"
SAVE_FREQ="${SAVE_FREQ:-25}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$VERL_ROOT" ]]; then
  export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"
fi

CHECK_ARGS=(
  --model "$MODEL_PATH"
  --data "$DATA_PATH"
  --project-root "$PROJECT_ROOT"
  --max-length "$MAX_LENGTH"
)
if [[ -n "$VERL_ROOT" ]]; then
  CHECK_ARGS+=(--verl-root "$VERL_ROOT")
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_appworld_sft_env.py" "${CHECK_ARGS[@]}"

if [[ -n "$VERL_ROOT" ]]; then
  cd "$VERL_ROOT"
else
  cd "$PROJECT_ROOT"
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_PATH" \
  data.val_files=null \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.max_length="$MAX_LENGTH" \
  data.max_token_len_per_gpu="$MAX_LENGTH" \
  data.truncation=left \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.pad_mode=no_padding \
  data.custom_cls.path="$PROJECT_ROOT/appworld_agentic_rl/verl_sft_dataset.py" \
  data.custom_cls.name=AppWorldMultiTurnSFTDataset \
  model.path="$MODEL_PATH" \
  model.tokenizer_path="$MODEL_PATH" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  model.enable_activation_offload="$ACTIVATION_OFFLOAD" \
  model.use_remove_padding="$USE_REMOVE_PADDING" \
  +model.override_config.attn_implementation="$ATTN_IMPLEMENTATION" \
  model.lora_rank=0 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  model.use_fused_kernels="$USE_FUSED_KERNELS" \
  model.fused_kernel_options.impl_backend="$FUSED_KERNEL_BACKEND" \
  engine.strategy=fsdp \
  engine.model_dtype=bfloat16 \
  engine.use_torch_compile=false \
  engine.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  optim.lr=1e-5 \
  optim.betas='[0.9,0.95]' \
  optim.lr_warmup_steps_ratio=0.1 \
  optim.lr_scheduler_type=cosine \
  trainer.default_local_dir="$OUTPUT_PATH" \
  trainer.project_name=appworld-agentic-rl \
  trainer.experiment_name=appworld-sft-qwen3.5-4b \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.resume_mode=disable \
  checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
  trainer.logger='["console"]' \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.profile_interval='[-1,-1]'
