# 可复现实验说明

## 1. 版本

- Python 3.11+
- AppWorld commit：`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`
- veRL：`0.9.0`
- Transformers：`7ea2320c76117e6742364808a666ef6f2fb40a67`
- Model：Qwen3.5-4B
- vLLM、PyTorch 与 CUDA 按目标 GPU 环境匹配

每次正式实验保存 `pip freeze`、配置快照、Git revision、manifest hash 和 GPU 信息。

## 2. 外部输入

运行时通过以下环境变量加载外部输入：

```text
APPWORLD_ROOT=/path/to/appworld-runtime
APPWORLD_MODEL=/path/to/Qwen3.5-4B-sft-final
APPWORLD_GRPO_TRAIN=/path/to/train_custom.parquet
APPWORLD_GRPO_VAL=/path/to/dev_custom.parquet
APPWORLD_GRPO_OUTPUT=/path/to/checkpoints
```

这些路径分别提供 AppWorld runtime 与任务数据库、模型权重、训练与验证数据，以及 checkpoint 输出目录。Teacher 原始轨迹由数据构建命令生成并保存在配置的产物目录中。

## 3. 数据与 Teacher

```bash
python -m appworld_agentic_rl.cli split-data --help
python -m appworld_agentic_rl.cli collect-teacher --help
python -m appworld_agentic_rl.cli replay-teacher --help
python -m appworld_agentic_rl.cli build-sft-data --help
```

Teacher 数据构建顺序：

```text
collect raw trajectories
  -> AppWorld evaluator
  -> execution-error / clean-state gate
  -> deduplicate exact attempt slots
  -> action-only SFT JSONL/parquet
```

Teacher 通过远程 OpenAI-compatible 接口采集时不假设能够取得服务端 tokenizer，使用 `131072 chars` 触发、压缩到 `98304 chars`，并保留最近 4 个完整 interaction group。字符预算是保守安全边界，不与 token 数作严格等价。

最终 SFT 输入为 446 条 clean trajectory、195 个 unique task。

## 4. SFT

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
DATA_PATH=/path/to/train.parquet \
OUTPUT_PATH=/path/to/sft-checkpoint \
PYTHON_BIN=/path/to/python \
bash scripts/run_appworld_sft.sh
```

训练完成后：

```bash
bash scripts/merge_sft_checkpoint.sh
```

合并目录作为 `APPWORLD_MODEL`，即 Qwen3.5-4B-sft-final。

## 5. RL

正式配置：

```yaml
train_batch_size: 4
rollout.n: 4
agent.num_workers: 16
ppo_mini_batch_size: 4
ppo_micro_batch_size_per_gpu: 1
ppo_epochs: 1
prompt_length: 60000
response_length: 100000
max_model_len: 160000
gpu_memory_utilization: 0.70
save_freq: 10
test_freq: -1
val_before_train: false
context_compaction_trigger_tokens: 65536
context_compaction_target_tokens: 49152
context_preserve_recent_groups: 4
```

启动：

```bash
APPWORLD_MODEL=/path/to/Qwen3.5-4B-sft-final \
APPWORLD_GRPO_TRAIN=/path/to/train_custom.parquet \
APPWORLD_GRPO_VAL=/path/to/dev_custom.parquet \
APPWORLD_ROOT=/path/to/appworld-runtime \
PYTHON_BIN=/path/to/verl-python \
WANDB_MODE=offline \
bash scripts/run_appworld_grpo_formal.sh
```

## 6. Checkpoint 选择

训练每 10 step 保存 checkpoint。训练完成后，脚本依次调用：

```text
evaluate_checkpoint_dev.sh
  -> dev_enforce/summary.json
  -> ranking.json
  -> best.json
  -> evaluate_selected_checkpoint.sh
  -> final_enforce/summary.json
```

排序优先级为 Task Success、Clean Success、Requirement Pass、较低 Infrastructure Error、较低 Collateral Damage。

## 7. Memory 对照

```bash
MODEL_PATH=/path/to/RL-best \
EVAL_TOKENIZER_PATH=/path/to/Qwen3.5-4B-sft-final \
APPWORLD_ROOT=/path/to/appworld-runtime \
CONTEXT_COMPARISON_MANIFEST=/path/to/dev_long_context.jsonl \
EVAL_PYTHON=/path/to/eval-python \
bash scripts/run_context_memory_comparison.sh
```

输出：

```text
full_context/run_config.json
full_context/results.jsonl
full_context/summary.json
structured_memory/run_config.json
structured_memory/results.jsonl
structured_memory/summary.json
comparison.json
```

## 8. 最终评测合同

Base、SFT-final 和 RL-best 均使用：

- 同一 180-task frozen manifest；
- `enforce` Harness；
- 同一 structured-memory schema；
- temperature 0；
- 最大 48 model turns；
- 使用各 checkpoint 对应的 Qwen tokenizer 精确执行 `65536 → 49152 tokens` 的压缩合同；
- 相同 max tokens、context limit 和 tool parser；
- 独立 AppWorld session。

Dev、final 和 Memory 对照会将 messages、chat template 与工具 schema 一起编码后计数。veRL 合并目录没有复制 tokenizer 文件时，通过 `EVAL_TOKENIZER_PATH` 指向 `sft-final`；Base 和 SFT 评测默认使用各自模型目录中的 tokenizer。

## 9. 保存的实验产物

完整实验保留：

- 环境与依赖版本；
- 训练配置和 manifest hash；
- W&B offline run；
- 每 10 step checkpoint；
- dev checkpoint ranking；
- final `results.jsonl` 与 `summary.json`；
- Memory comparison 的逐任务结果；
- GPU 时间、终止原因和基础设施错误统计。

最终指标见[最终结果](final_results.md)。
