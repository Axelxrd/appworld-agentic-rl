"""veRL AgentLoop for AppWorld agent-tool online GRPO."""
from __future__ import annotations
import json
from copy import deepcopy
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, ToolListWrap, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS, AppWorldAdapter
from appworld_agentic_rl.context_memory import MemoryState, context_snapshot, serialize_memory, update_memory
from appworld_agentic_rl.process_reward import score_trajectory
from appworld_agentic_rl.teacher_rollout import redact_sensitive_data
from appworld_agentic_rl.verl_compat import install_appworld_advantage_estimators
install_appworld_advantage_estimators()

def _event_credits(info: dict[str, Any]) -> dict[int, float]:
    raw = {int(x.get("step_index", -1)): float(x.get("delta") or 0.0)
           for x in (info.get("events") or []) if isinstance(x, dict)}
    target = float(info.get("process_score") or 0.0)
    if not raw:
        return {}
    total = sum(raw.values())
    if abs(total) > 1e-8:
        scale = target / total
        return {key: value * scale for key, value in raw.items()}
    last = next(reversed(raw))
    return {key: target if key == last else 0.0 for key in raw}


def _recent_interaction_messages(messages: list[dict[str, Any]], base_count: int, groups: int) -> list[dict[str, Any]]:
    """Keep the most recent assistant/tool interaction groups verbatim."""

    tail = messages[base_count:]
    if not tail or int(groups) <= 0:
        return []
    return deepcopy(tail[-int(groups) * 2 :])


def _compact_messages(
    base_messages: list[dict[str, Any]], memory: MemoryState, recent: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Inject memory into the existing system message without reordering roles."""

    messages = deepcopy(base_messages)
    memory_text = serialize_memory(memory)
    system_index = next((i for i, msg in enumerate(messages) if msg.get("role") == "system"), None)
    if system_index is None:
        messages.insert(0, {"role": "system", "content": memory_text})
    else:
        original = messages[system_index].get("content") or ""
        messages[system_index]["content"] = f"{original}\n\n{memory_text}"
    messages.extend(deepcopy(recent))
    return messages

class AppWorldAgentLoopOutput(AgentLoopOutput):
    """Serialize AppWorld terminal and process rewards as a token tensor."""
    def as_dict(self) -> dict[str, Any]:
        data = super().as_dict()
        info = self.extra_fields.get("appworld_reward") or {}
        mask = data["response_mask"]
        rewards = torch.zeros_like(mask, dtype=torch.float32)
        if not bool(info.get("infrastructure_invalid", False)):
            spans = {int(x.get("step_index", -1)): (max(0, int(x.get("start", 0))),
                    min(rewards.shape[0], int(x.get("end", 0))))
                    for x in (info.get("action_spans") or []) if isinstance(x, dict)}
            process_lambda = float(info.get("process_reward_lambda") or 0.0)
            for step_index, credit in _event_credits(info).items():
                start, end = spans.get(step_index, (0, 0))
                if end <= start:
                    continue
                active = torch.nonzero(mask[start:end] > 0, as_tuple=False).flatten()
                if active.numel():
                    rewards[start + int(active[0])] += process_lambda * float(credit)
            active_positions = torch.nonzero(mask > 0, as_tuple=False).flatten()
            if active_positions.numel():
                rewards[int(active_positions[-1])] += float(info.get("terminal_reward") or 0.0)
        else:
            mask.zero_()
        data["response_mask"] = mask
        data["rm_scores"] = rewards
        return data

def _agent_tool_carriers() -> list[FunctionTool]:
    carriers = []
    for raw in AGENT_TOOL_SCHEMAS:
        schema = OpenAIFunctionToolSchema.model_validate(raw)
        carriers.append(FunctionTool(name=schema.function.name, fn=lambda **_: None, tool_schema=schema))
    return carriers

@register("appworld_agent")
class AppWorldAgentLoop(ToolAgentLoop):
    """Let one policy choose actions from the same tool set used for SFT."""
    def __init__(self, *args, appworld_root: str | Path | None = None,
                 experiment_prefix: str = "appworld_grpo", max_steps: int = 50,
                 process_reward_lambda: float = 0.3, split: str = "train",
                 context_compaction_enable: bool = True,
                 context_compaction_trigger_tokens: int | None = None,
                 context_compaction_target_tokens: int | None = None,
                 context_compaction_min_generation_tokens: int = 4096,
                 context_preserve_recent_groups: int = 4, **kwargs):
        kwargs.pop("tools", None)
        super().__init__(*args, tools=ToolListWrap(_agent_tool_carriers()), **kwargs)
        self.appworld_root = str(appworld_root) if appworld_root is not None else None
        self.experiment_prefix = str(experiment_prefix)
        self.max_steps = int(max_steps)
        self.process_reward_lambda = float(process_reward_lambda)
        self.default_split = str(split)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.context_compaction_trigger_tokens = (
            int(context_compaction_trigger_tokens) if context_compaction_trigger_tokens is not None else None
        )
        self.context_compaction_target_tokens = (
            int(context_compaction_target_tokens) if context_compaction_target_tokens is not None else None
        )
        self.context_compaction_min_generation_tokens = max(1, int(context_compaction_min_generation_tokens))
        self.context_preserve_recent_groups = max(1, int(context_preserve_recent_groups))
        self._adapter: AppWorldAdapter | None = None

    def _model_context_limit(self) -> int:
        configured = getattr(self.rollout_config, "max_model_len", None)
        return max(1, int(configured or (self.prompt_length + self.response_length)))

    def _compaction_limits(self) -> tuple[int, int]:
        """Return context-based trigger/target limits, independent of prompt_length."""

        model_limit = self._model_context_limit()
        trigger = self.context_compaction_trigger_tokens
        if trigger is None:
            trigger = model_limit - self.context_compaction_min_generation_tokens
        target = self.context_compaction_target_tokens
        if target is None:
            target = min(self.prompt_length, int(model_limit * 0.65))
        if trigger <= 0 or target <= 0:
            raise ValueError("context compaction token limits must be positive")
        if target > self.prompt_length:
            raise ValueError(
                "context_compaction_target_tokens must not exceed rollout.prompt_length "
                "because the compact prompt is emitted in the fixed prompt tensor"
            )
        if target >= trigger:
            raise ValueError("context_compaction_target_tokens must be smaller than trigger_tokens")
        return int(trigger), int(target)
    @staticmethod
    def _response_span(response_mask: list[int]) -> tuple[int, int]:
        end = len(response_mask)
        while end and int(response_mask[end - 1]) == 0:
            end -= 1
        start = end
        while start and int(response_mask[start - 1]) == 1:
            start -= 1
        return start, end

    async def _handle_pending_state(self, agent_data: Any, sampling_params: dict[str, Any]) -> AgentState:
        """Capture the immutable task prompt before messages grow."""

        agent_data.extra_fields.setdefault("_appworld_base_messages", deepcopy(agent_data.messages))
        agent_data.extra_fields.setdefault("_appworld_base_message_count", len(agent_data.messages))
        agent_data.extra_fields.setdefault("_appworld_memory", MemoryState())
        agent_data.extra_fields.setdefault("_appworld_memory_step_count", 0)
        agent_data.extra_fields.setdefault("_appworld_total_response_tokens", 0)
        agent_data.extra_fields.setdefault("appworld_context_snapshots", [])
        agent_data.extra_fields.setdefault("appworld_event_records", [])
        agent_data.extra_fields.setdefault("appworld_rollout_uid", "")
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _compact_context_if_needed(self, agent_data: Any) -> int:
        """Replace old interaction groups by deterministic structured memory.

        The complete trajectory remains in AppWorldAdapter.  Resetting the
        response window here makes only the post-compaction suffix trainable;
        its prompt is exactly the compact context used for generation.
        """

        if not self.context_compaction_enable or self._adapter is None:
            return 0
        prompt_limit = int(self.prompt_length)
        model_limit = self._model_context_limit()
        before = len(agent_data.prompt_ids)
        trigger, target = self._compaction_limits()
        hard_trigger = model_limit - self.context_compaction_min_generation_tokens
        if before < min(trigger, hard_trigger):
            return 0
        trajectory = self._adapter.export_trajectory()
        memory = agent_data.extra_fields.setdefault("_appworld_memory", MemoryState())
        processed = int(agent_data.extra_fields.get("_appworld_memory_step_count", 0))
        for step in (trajectory.get("steps") or [])[processed:]:
            if isinstance(step, dict):
                update_memory(memory, step)
        agent_data.extra_fields["_appworld_memory_step_count"] = len(trajectory.get("steps") or [])

        base_messages = deepcopy(agent_data.extra_fields.get("_appworld_base_messages") or agent_data.messages[:1])
        recent_groups = self.context_preserve_recent_groups
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        base_count = int(agent_data.extra_fields.get("_appworld_base_message_count", len(base_messages)))
        recent = _recent_interaction_messages(agent_data.messages, base_count, recent_groups)
        compact_messages = _compact_messages(base_messages, memory, recent)
        compact_ids = await self.apply_chat_template(compact_messages, tools=schemas)
        while len(compact_ids) > target and recent_groups > 0:
            recent_groups -= 1
            recent = _recent_interaction_messages(agent_data.messages, base_count, recent_groups)
            compact_messages = _compact_messages(base_messages, memory, recent)
            compact_ids = await self.apply_chat_template(compact_messages, tools=schemas)
        if len(compact_ids) > prompt_limit:
            raise RuntimeError(
                f"structured AppWorld memory still exceeds prompt_length: {len(compact_ids)} > {prompt_limit}"
            )

        removed = max(0, before - len(compact_ids))
        agent_data.prompt_ids = compact_ids
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.extra_fields["appworld_action_spans"] = []
        agent_data.extra_fields["appworld_context_snapshots"].append(
            context_snapshot(
            turn=int(agent_data.assistant_turns),
            context_tokens_before=before,
            context_token_limit=model_limit,
            tokens_removed=removed,
            memory=memory,
            recent_groups=recent_groups,
            )
        )
        agent_data.extra_fields["appworld_compaction_count"] = int(
            agent_data.extra_fields.get("appworld_compaction_count", 0)
        ) + 1
        agent_data.extra_fields["appworld_trainable_suffix_only"] = True
        return removed

    async def _handle_generating_state(
        self, agent_data: Any, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        total = int(agent_data.extra_fields.get("_appworld_total_response_tokens", 0))
        if total >= self.response_length:
            return AgentState.TERMINATED
        model_limit = self._model_context_limit()
        compacted = await self._compact_context_if_needed(agent_data)
        if not compacted:
            memory = agent_data.extra_fields.setdefault("_appworld_memory", MemoryState())
            agent_data.extra_fields.setdefault("appworld_context_snapshots", []).append(
                context_snapshot(
                    turn=int(agent_data.assistant_turns),
                    context_tokens_before=len(agent_data.prompt_ids),
                    context_token_limit=model_limit,
                    tokens_removed=0,
                    memory=memory,
                    recent_groups=self.context_preserve_recent_groups,
                )
            )
        context_tokens = len(agent_data.prompt_ids)
        # Capture the exact token context used by vLLM.  This snapshot is kept
        # independently of the mutable prompt buffer so events before a later
        # compaction remain trainable.
        event_context_ids = list(agent_data.prompt_ids)
        remaining = max(1, self.response_length - total)
        available = max(1, model_limit - context_tokens)
        sampling_params = {**sampling_params, "max_tokens": min(remaining, available)}
        local_before = len(agent_data.response_mask)
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=ignore_termination)
        generated_ids = list(getattr(agent_data, "response_ids", []) or [])
        generated_logprobs = list(getattr(agent_data, "response_logprobs", []) or [])
        if generated_ids:
            agent_data.extra_fields["_appworld_pending_event"] = {
                "context_ids": event_context_ids,
                "action_ids": generated_ids,
                "action_log_probs": generated_logprobs[-len(generated_ids) :] if generated_logprobs else [],
                "assistant_turn": int(agent_data.assistant_turns),
            }
        agent_data.extra_fields["_appworld_total_response_tokens"] = total + max(
            0, len(agent_data.response_mask) - local_before
        )
        if agent_data.extra_fields["_appworld_total_response_tokens"] >= self.response_length:
            return AgentState.TERMINATED
        return state

    async def _handle_processing_tools_state(self, agent_data: Any) -> AgentState:
        # In non-continuous mode veRL keeps assistant turns only in token ids.
        # Maintain a message-form copy for future deterministic compaction;
        # this does not alter the token stream used by the base loop.
        if not self.enable_continuous_token and agent_data.tool_calls:
            try:
                agent_data.messages.append(self._build_assistant_message("", agent_data))
            except (TypeError, ValueError):
                agent_data.messages.append({"role": "assistant", "content": ""})
        local_before = len(agent_data.response_mask)
        state = await super()._handle_processing_tools_state(agent_data)
        total = int(agent_data.extra_fields.get("_appworld_total_response_tokens", 0))
        agent_data.extra_fields["_appworld_total_response_tokens"] = total + max(
            0, len(agent_data.response_mask) - local_before
        )
        if agent_data.extra_fields["_appworld_total_response_tokens"] >= self.response_length:
            return AgentState.TERMINATED
        if agent_data.extra_fields.get("_appworld_terminal_tool"):
            return AgentState.TERMINATED
        return state
    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: Any):
        adapter = self._adapter
        if adapter is None:
            return ToolResponse(text="AppWorld adapter is not initialized."), 0.0, {}
        span_start, span_end = self._response_span(agent_data.response_mask)
        try:
            arguments = json.loads(tool_call.arguments or "{}")
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must decode to an object")
            compact_call = {"name": tool_call.name, "arguments": arguments}
            violations = adapter.validate_tool_call(compact_call)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            arguments = {}
            violations = [{"code": "invalid_tool_arguments", "message": str(exc)}]
            compact_call = {"name": tool_call.name, "arguments": arguments}
        step_index = len(adapter.trajectory.steps) if adapter.trajectory is not None else 0
        if violations:
            observation = {"error": "agent action rejected by AppWorld guard", "violations": violations}
            if adapter.trajectory is not None:
                adapter._append_step(tool_call.name, arguments, observation, "validation_failed", False, False,
                                     metadata={"violations": redact_sensitive_data(violations)})
            result = {"tool_name": tool_call.name, "observation": observation, "error": "validation_failed",
                      "state_change": False, "terminated": False}
        else:
            result = adapter.step(compact_call)
        pending = agent_data.extra_fields.pop("_appworld_pending_event", None)
        if isinstance(pending, dict):
            agent_data.extra_fields.setdefault("appworld_event_records", []).append(
                {
                    **pending,
                    "step_index": int(step_index),
                    "tool_name": str(tool_call.name),
                    "action_span": {"start": int(span_start), "end": int(span_end)},
                    "executed": not bool(violations),
                }
            )
        agent_data.extra_fields.setdefault("appworld_action_spans", []).append(
            {"step_index": int(step_index), "start": int(span_start), "end": int(span_end),
             "tool_name": str(tool_call.name)})
        if result.get("terminated"):
            agent_data.extra_fields["_appworld_terminal_tool"] = True
        text = json.dumps(redact_sensitive_data(result), ensure_ascii=False, sort_keys=True)
        return ToolResponse(text=text), 0.0, result
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        task_id = str(kwargs.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("AppWorld RL dataset row must contain task_id")
        split = str(kwargs.get("split") or self.default_split)
        self._adapter = AppWorldAdapter(experiment_name=f"{self.experiment_prefix}_{task_id}_{uuid4().hex}",
            appworld_root=self.appworld_root, max_steps=self.max_steps)
        adapter = self._adapter
        try:
            adapter.reset(task_id, split=split)
            output = await super().run(sampling_params, **kwargs)
            # veRL v1's TransferQueue tags are built from these fields. The
            # stock loop injects them in its output path, but a custom loop
            # must preserve the generation policy version explicitly.
            try:
                global_steps = int(kwargs.get("global_steps", 0))
            except (TypeError, ValueError):
                global_steps = 0
            output.extra_fields.setdefault("global_steps", global_steps)
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            context_snapshots = output.extra_fields.get("appworld_context_snapshots") or []
            if os.environ.get("APPWORLD_CONTEXT_DEBUG") == "1":
                compacted = [item for item in context_snapshots if item.get("truncated")]
                print(
                    "[APPWORLD_CONTEXT] "
                    f"task={task_id} snapshots={len(context_snapshots)} "
                    f"compactions={len(compacted)} "
                    f"removed={[item.get('tokens_removed', 0) for item in compacted]}",
                    flush=True,
                )
            evaluation_result = adapter.evaluate(suppress_errors=True)
            trajectory = redact_sensitive_data(adapter.export_trajectory())
            terminal_reward = float(evaluation_result.get("terminal_reward") or 0.0)
            prm = score_trajectory(trajectory, terminal_reward=terminal_reward)
            process_score = float(prm.process_score)
            total_reward = terminal_reward + self.process_reward_lambda * process_score
            evaluation = redact_sensitive_data(evaluation_result.get("evaluation") or {})
            infrastructure_invalid = bool(isinstance(evaluation, dict) and evaluation.get("error"))
            output.reward_score = total_reward
            output.extra_fields["appworld_reward"] = {"task_id": task_id, "split": split,
                "terminal_reward": terminal_reward, "process_score": process_score, "total_reward": total_reward,
                "process_reward_lambda": self.process_reward_lambda, "infrastructure_invalid": infrastructure_invalid,
                "events": prm.to_dict()["events"], "metrics": prm.to_dict()["metrics"],
                "action_spans": output.extra_fields.get("appworld_action_spans", []),
                "trajectory": trajectory, "evaluation": evaluation}
            output.extra_fields["appworld_event_records"] = output.extra_fields.get(
                "appworld_event_records", []
            )
            group_uid = str(kwargs.get("index") or kwargs.get("uid") or task_id)
            output.extra_fields["appworld_group_uid"] = group_uid
            output.extra_fields["appworld_rollout_uid"] = f"{group_uid}::{uuid4().hex}"
            # Internal compression state is not part of the TransferQueue
            # schema; retain only JSON-serializable audit snapshots.
            for key in (
                "_appworld_base_messages",
                "_appworld_base_message_count",
                "_appworld_memory",
                "_appworld_memory_step_count",
                "_appworld_total_response_tokens",
            ):
                output.extra_fields.pop(key, None)
            return AppWorldAgentLoopOutput.model_validate(output.model_dump())
        finally:
            adapter.close()
            self._adapter = None
__all__ = ["AppWorldAgentLoop", "AppWorldAgentLoopOutput"]
