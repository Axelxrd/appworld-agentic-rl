# 最终实验结果

## 1. 评测设置

最终结果使用冻结的 180-task AppWorld final manifest。Base、SFT-final 和 RL-best 采用相同的 `final_enforce` 合同，包括 Harness、structured memory、temperature 0、最大 48 model turns 和相同模型服务参数。

RL-best 根据 `dev_enforce` 指标选择；final split 只用于选定 checkpoint 的最终评测。

## 2. Base vs SFT vs RL

| Model | Task Success | Clean Success | Requirement Pass | Collateral Damage | Execution Error |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 0 / 180 (0.00%) | 0 / 180 (0.00%) | 21 / 180 (11.67%) | 8.89% | 72.22% |
| Qwen3.5-4B SFT-final | 96 / 180 (53.33%) | 89 / 180 (49.44%) | 125 / 180 (69.44%) | 5.00% | 28.33% |
| Qwen3.5-4B RL-best | 117 / 180 (65.00%) | 110 / 180 (61.11%) | 148 / 180 (82.22%) | 2.78% | 15.56% |

## 3. 阶段提升

### 3.1 Base → SFT

SFT 将 Task Success 从 0.00% 提升到 53.33%。Verifier-filtered action-only 数据使模型获得以下能力：

- 正确输出 function call；
- 自主搜索 API 和核对 schema；
- 区分只读查询与写入；
- 管理认证和共享代码变量；
- 根据 observation 完成多步任务。

Execution Error 从 72.22% 降至 28.33%，主要减少了工具协议、API 名称和参数格式错误。

### 3.2 SFT → RL

RL-best 相对 SFT-final：

| 指标 | 变化 |
|---|---:|
| Successful tasks | +21 |
| Task Success | +11.67 pt |
| Clean Success | +11.67 pt |
| Requirement Pass | +12.78 pt |
| Execution Error | -12.77 pt |
| Collateral Damage | -2.22 pt |

RL 的收益主要来自更稳定的长轨迹决策、具有参数依据的写入、错误恢复，以及在 evaluator 已满足后及时终止。

## 4. Structured Memory 对照

Memory 对照使用固定的 36-task dev long-context subset。模型、Harness、采样参数和服务设置保持一致，只改变上下文管理策略。

| Condition | Task Success | Clean Success | Execution Error | Post-compaction NameError |
|---|---:|---:|---:|---:|
| Full context | 19 / 36 (52.78%) | 17 / 36 (47.22%) | 27.78% | 5 |
| Structured memory | 25 / 36 (69.44%) | 23 / 36 (63.89%) | 13.89% | 1 |

Structured Memory 带来：

- Task Success 提升 16.66 pt；
- Clean Success 提升 16.67 pt；
- Execution Error 降低 13.89 pt；
- Post-compaction `NameError` 从 5 次降至 1 次。

提升主要集中在实际发生 compaction、需要跨多轮复用鉴权、实体、schema 或 Python 变量的任务。短任务通常不触发压缩，因此两种策略在该部分的行为接近。

## 5. 错误类型变化

Base 的失败主要来自工具协议、API 名称猜测、参数错误和错误恢复失败。SFT 后，失败更多集中在长轨迹中的实体绑定、重复搜索和写入时机。RL-best 进一步减少以下问题：

- 把写 API 放入只读工具；
- 引用未成功创建的变量；
- 完全重复工具调用；
- 未读取目标状态便直接写入；
- 已经完成任务后继续操作；
- 状态修改满足核心要求但产生附带破坏。

## 6. 结论

Qwen3.5-4B 经 action-only SFT 获得可用的 AppWorld 工具智能体能力，再通过在线 GRPO 将 Task Success 从 53.33% 提升到 65.00%。轻量化过程奖励与事件级信用分配改善了稀疏终局反馈，Structured Memory 降低了压缩后的变量和认证信息丢失。

这些结果表明，AppWorld 可以支持包含数据构建、SFT、在线 RL、长上下文管理和冻结评测的长程 Agentic RL 训练流程。

复现设置见[可复现实验说明](reproducibility.md)，方法细节见[方法设计](method_design.md)。
