# 方法设计

## 1. Agent Harness

Harness 负责管理 AppWorld session、将 function call 转换为环境 action、记录 observation 和状态变化、维护上下文、执行安全 guard，并将轨迹交给奖励模块与 evaluator。Harness 不规定工具调用路线，也不替 Agent 决策。

### 1.1 Guard

始终强制的规则包括：

- 拒绝未知工具或非法参数；
- 拒绝危险 runtime access；
- 拒绝在只读工具中调用写 API；
- 拒绝没有实际写 API 的写入工具调用；
- 拒绝绕过上一条 observation 的并行 action；
- 拒绝终止后的继续操作。

行为 guard 支持 `enforce` 与 `shadow`。正式训练和最终评测采用 `enforce`；`shadow` 用于分析模型在不拦截动作时的原始行为。

### 1.2 Observation Projection

Projection 控制单次工具返回的体积，在保留结果结构和关键字段的前提下压缩冗长 observation。它处理单次 observation 过长问题；跨多轮的信息维护由 structured memory 负责。

## 2. 终局奖励

终局奖励由 AppWorld evaluator 产生。Evaluator 检查任务要求、目标数据库状态和状态完整性。模型输出“完成”不等于任务成功，成功信号以 evaluator 结果为准。

## 3. 轻量化过程奖励

过程奖励只读取 actor 可见的轨迹证据，为每个 action 生成有界的 event delta。

正向信号包括：

- 新的有效 API discovery；
- 核对准确 schema；
- 有效状态读取；
- 参数来源可验证的写入；
- 从错误中恢复；
- evaluator 成功后的及时终止。

负向信号包括：

- 工具与参数完全相同的重复 action；
- 不同 query 返回相同 API 集合的重复搜索；
- execution error；
- required parameter 缺少可验证来源的写入；
- 不必要或超出任务范围的状态修改。

过程分数设置上限并聚合 event delta，使回报规模不随低价值调用数量线性增长。

## 4. 参数级 Provenance

在执行 `run_write_code` 前，adapter 解析写 API 和 required parameters，并判断每个参数是否来自：

- 任务初始信息；
- 已执行 action 的 observation；
- 已确认的变量绑定；
- 明确 literal。

只有全部 required parameters 可追溯时，写入才记为 grounded。Provenance 仅引用当前 action 之前已经存在的证据，当前 action 产生的信息不参与本次判断。

## 5. 事件级信用分配

每个 action event 保存：

```text
context_ids
action_ids
action_log_probs
action_span
step_index
event_reward
```

训练时，每个 action 在生成时的 immutable context 下形成独立 actor-loss row。局部 event reward 只分配到该 action token；prompt、observation 和 padding 不参与 actor loss。终局 outcome 再依据事件位置与长度尺度向各 action 分配。

这一设计带来两项性质：

1. 后续 action 的错误不会等量惩罚此前所有正确 action；
2. Structured Memory 压缩前的 action 不会因 response window 重建而从 loss 中消失。

### 5.1 同质 Group Fallback

若同一 GRPO group 的终局 reward 没有方差，则使用局部 event reward 的组间差异；若终局和局部信号都同质，则跳过该 group。

## 6. Structured Memory

简单左截断可能删除 system protocol、鉴权事实、实体 ID、schema 或 Python 变量定义。RL、dev 和 final 使用对应 Qwen checkpoint 的 chat template 与 tokenizer，把工具 schema 一并编码后精确计数：上下文达到 65,536 tokens 时构造 `[APPWORLD MEMORY]`，目标为 49,152 tokens，并保留最近 4 个完整 interaction group。

Teacher 可能由不公开 tokenizer 的远程服务提供，因此不把字符数声明为精确 token 数。Teacher 使用保守的 `131,072 → 98,304 chars` 安全预算；其压缩内容、变量绑定、API discovery 去重和最近交互保留规则与 RL/评测一致，区别只在长度计量器。

Memory 保存：

- 任务目标和已确认约束；
- 非敏感实体和 ID；
- 认证状态的可用性；
- 已查看 schema 与 required parameters；
- 按 `app.api` 去重的 API discovery；
- Python 变量名、语义、来源 event 和可用状态；
- 最近的完整交互。

真实 password、access token 和 API key 不写入 Memory。

例如：

```python
venmo_token = login(...)
```

Memory 记录变量名 `venmo_token`、认证语义、来源事件与可用状态，但不记录 token 值。如果创建变量的 action 被 guard 拒绝，则该绑定标记为不可用。

## 7. 压缩后的训练对齐

压缩前已经生成的 action 持有原始 context snapshot；压缩后的 action 使用 compact context。训练按 event replay 原始生成上下文，因此不会用压缩后的 Memory 重新解释压缩前 action，也不会产生 rollout 与 policy loss 的上下文错位。

## 8. 方法结果

完整方法将 SFT-final 的 Task Success 从 53.33% 提升到 65.00%，并将 Execution Error 从 28.33% 降至 15.56%。Structured Memory 在 long-context dev 子集中将成功率从 52.78% 提升到 69.44%。详细结果见[最终结果](final_results.md)。
