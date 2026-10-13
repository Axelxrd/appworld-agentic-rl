# 项目详述

## 1. 项目动机

静态问答只要求模型生成文本，而 AppWorld 任务要求 Agent 在带状态的应用环境中完成多步操作。Agent 必须理解目标、发现 API、核对 schema、读取实体、管理认证信息、执行状态修改、恢复错误，并确保最终数据库状态满足任务要求且不存在附带破坏。

本项目将 AppWorld 从评测环境扩展为完整的长程 Agentic RL 训练环境。训练奖励和最终指标来自环境交互与程序化 evaluator。

## 2. 环境与 Agent Tools

环境向 Agent 提供八个高层工具：

| Tool | 职责 |
|---|---|
| `inspect_task` | 查看任务、时间和允许访问的应用。 |
| `search_api_docs` | 检索候选 API。 |
| `inspect_api_schema` | 获取准确参数和返回结构。 |
| `query_state` | 执行只读状态查询。 |
| `run_code` | 执行非写入计算。 |
| `run_write_code` | 执行有明确意图的状态修改。 |
| `complete_task` | 提交结果并结束任务。 |
| `finish_without_completion` | 无安全解法时终止。 |

固定的是工具接口和安全语义，不固定的是行动路线。Agent 可以根据最新 observation 自主决定先查询、检索、核对 schema、写入、验证、恢复或结束。

## 3. 数据隔离

数据以 task/scenario 为隔离单位：

| 阶段 | 数据 |
|---|---|
| Teacher collection 与 SFT | train |
| 奖励调试、checkpoint 选择、Memory 对照 | dev |
| 最终评测 | frozen `test_challenge` manifest |

同一个任务不会同时出现在 Teacher、SFT、在线 RL 和 final test 中。Final split 在 dev 选出最佳 checkpoint 后用于冻结评测。

## 4. Teacher 与 SFT

Teacher 轨迹采用 raw-first 保存。每条轨迹完成后，由 AppWorld evaluator 检查目标状态，再经过 execution-error 与 clean-state gate。Verifier-passing 且无执行错误的轨迹进入 SFT。

最终 SFT 数据包含：

```text
446 clean trajectories
195 unique tasks
```

SFT 使用 action-only loss：用户消息和环境 observation 作为上下文保留，只在 Assistant 工具调用 token 上计算损失；Teacher 隐藏思考和环境返回不作为监督目标。

SFT 从 Qwen3.5-4B Base 初始化，训练完成后合并为 Qwen3.5-4B-sft-final。该模型已经掌握工具协议、API 检索、只读/写入区分、共享代码变量和任务终止行为。

## 5. 在线强化学习

RL actor 从 SFT-final 初始化，使用 veRL 0.9 异步 AgentLoop 和在线 GRPO。每个 batch 包含 4 个 prompt，每个 prompt 采样 4 条轨迹，共由 16 个隔离 AppWorld session 并发执行。

```text
4 prompts × 4 rollouts = 16 independent Agent sessions
```

一个 prompt 的四条 rollout 构成 GRPO group。每批 rollout 进行一次参数更新。训练每 10 step 保存 checkpoint，训练结束后在独立进程中完成 dev_enforce 评测与排序。

## 6. 训练配置

| 参数 | 设置 |
|---|---:|
| Base / SFT / RL model | Qwen3.5-4B |
| Train prompt batch | 4 |
| Rollouts per prompt | 4 |
| Agent workers | 16 |
| PPO mini-batch | 4 |
| PPO micro-batch / GPU | 1 |
| PPO epochs | 1 |
| Prompt budget | 60,000 tokens |
| Response budget | 100,000 tokens |
| Model context | 160,000 tokens |
| RL / Dev / Final memory trigger | 65,536 tokens（tokenizer 精确计数） |
| RL / Dev / Final memory target | 49,152 tokens |
| Teacher memory trigger | 131,072 chars（保守安全预算） |
| Teacher memory target | 98,304 chars |
| Recent groups | 4 |
| Maximum model turns | 48 |
| vLLM GPU utilization | 0.70 |

## 7. 评测流程

训练与评测解耦：

```text
save every 10 steps
  -> evaluate all checkpoints on dev_enforce
  -> rank by success, clean success, requirement pass,
     infrastructure error and collateral damage
  -> run final_enforce on the selected checkpoint
```

Base、SFT-final 和 RL-best 使用同一 final manifest、Harness、structured-memory 策略、temperature 0、最大 turns 和模型服务参数。这些设置构成统一的最终评测合同。

## 8. 完成状态

项目已经走通任务划分、Teacher、SFT、GRPO、事件级训练、Memory、dev 选 best 和 frozen final 的完整闭环。方法细节见[方法设计](method_design.md)，运行方式见[复现实验](reproducibility.md)，最终指标见[最终结果](final_results.md)。
