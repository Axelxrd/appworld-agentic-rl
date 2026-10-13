# AppWorld Agentic RL

AppWorld Agentic RL 是一套面向长程工具智能体的完整后训练系统。项目使用 Qwen3.5-4B 作为基础模型，贯通 Teacher 轨迹采集、程序化验证、action-only SFT、在线 GRPO、轻量化过程奖励、事件级信用分配、structured memory、dev checkpoint 选择和 frozen final 评测。

## 上游 AppWorld

本项目基于 Stony Brook NLP 发布的 AppWorld 环境，并在其任务、运行时状态、API 文档和程序化 evaluator 之上构建 Teacher、SFT、在线 RL、长上下文管理与冻结评测流水线。本仓库是独立的研究实现，不是 AppWorld 官方仓库。

- 官方仓库：[StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld)
- 原始论文：[AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901)（ACL 2024 Best Resource Paper）
- 本项目固定使用的 AppWorld revision：`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`

```text
AppWorld task/state
  -> Agent 自主工具调用
  -> verifier-filtered Teacher trajectories
  -> action-only SFT
  -> online GRPO
  -> event-aligned training
  -> dev selection
  -> frozen final evaluation
```

工具集合由环境提供，但工具选择、参数、调用次数、错误恢复路线和终止时机均由 Agent 根据任务和最新 observation 自主决定，不使用预设执行路线。

## 核心结果

统一的 180-task `final_enforce` 评测结果如下：

| Model | Task Success | Clean Success | Requirement Pass | Execution Error |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Base | 0 / 180 (0.00%) | 0 / 180 (0.00%) | 21 / 180 (11.67%) | 72.22% |
| Qwen3.5-4B SFT-final | 96 / 180 (53.33%) | 89 / 180 (49.44%) | 125 / 180 (69.44%) | 28.33% |
| Qwen3.5-4B RL-best | 117 / 180 (65.00%) | 110 / 180 (61.11%) | 148 / 180 (82.22%) | 15.56% |

RL-best 相对 SFT-final 多完成 21 个任务，Task Success 提升 11.67 个百分点，同时将 execution error 从 28.33% 降至 15.56%。

## 实验流程

- 以 task/scenario 为单位划分 train、dev 和 frozen final；
- 采集 Teacher 原始轨迹并使用 AppWorld evaluator 验证；
- 构建 446 条、覆盖 195 个 unique task 的 action-only SFT 数据；
- 训练并合并 Qwen3.5-4B-sft-final；
- 运行 16 路异步 AppWorld rollout 和在线 GRPO；
- 接入轻量化过程奖励、参数 provenance 和事件级信用分配；
- 完成 structured-memory 长上下文训练适配；
- 按 dev_enforce 选择 best checkpoint；
- 完成 Base、SFT-final、RL-best frozen final 对比；
- 完成 full-context 与 structured-memory 对照。

## 文档导航

- [项目详述](docs/project_overview.md)：问题、环境、数据和端到端训练流程。
- [方法设计](docs/method_design.md)：Harness、奖励、信用分配和 Memory。
- [复现实验](docs/reproducibility.md)：依赖、配置、命令与运行产物。
- [最终结果](docs/final_results.md)：主结果、Memory 对照和结果分析。

## 目录职责

```text
appworld_agentic_rl/   环境 adapter、AgentLoop、memory、reward 与评测实现
configs/               SFT/RL 与 AgentLoop 配置
scripts/               数据、训练、评测和服务入口
tests/                 单元测试与接口回归测试
docs/                  项目、方法、复现和结果文档
```
