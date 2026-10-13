"""Deterministic protocol, guard, and context utilities for teacher rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal

from appworld_agentic_rl.context_memory import MemoryState, serialize_memory
from appworld_agentic_rl.context_policy import (
    DEFAULT_CONTEXT_POLICY,
    TEACHER_CHARS_PER_TOKEN_ESTIMATE,
    TEACHER_CONTEXT_TARGET_CHARS,
    TEACHER_CONTEXT_TRIGGER_CHARS,
)


HARNESS_VERSION = "appworld-harness-v3-unit-aware-memory"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherHarnessConfig:
    """Frozen behavior and context contract shared by collection and evaluation."""

    bc_guard_mode: Literal["shadow", "enforce"] = "enforce"
    max_action_attempts: int = 80
    invalid_action_limit: int = 6
    max_identical_action_repeats: int = 1
    context_budget_mode: Literal["chars", "tokens"] = "chars"
    max_context_chars: int = TEACHER_CONTEXT_TRIGGER_CHARS
    max_observation_chars: int = DEFAULT_CONTEXT_POLICY.max_observation_chars
    context_compaction_enable: bool = True
    context_compaction_trigger_chars: int | None = TEACHER_CONTEXT_TRIGGER_CHARS
    context_compaction_target_chars: int | None = TEACHER_CONTEXT_TARGET_CHARS
    max_context_tokens: int = 155_904
    context_compaction_trigger_tokens: int | None = DEFAULT_CONTEXT_POLICY.trigger_tokens
    context_compaction_target_tokens: int | None = DEFAULT_CONTEXT_POLICY.target_tokens
    context_preserve_recent_groups: int = DEFAULT_CONTEXT_POLICY.preserve_recent_groups

    def __post_init__(self) -> None:
        if self.bc_guard_mode not in {"shadow", "enforce"}:
            raise ValueError("bc_guard_mode must be 'shadow' or 'enforce'")
        if self.context_budget_mode not in {"chars", "tokens"}:
            raise ValueError("context_budget_mode must be 'chars' or 'tokens'")
        for name in (
            "max_action_attempts",
            "invalid_action_limit",
            "max_identical_action_repeats",
            "max_context_chars",
            "max_observation_chars",
            "max_context_tokens",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        trigger = self.context_compaction_trigger_chars or self.max_context_chars
        target = self.context_compaction_target_chars or max(1, int(trigger * 0.65))
        if trigger < 1 or target < 1 or target >= trigger:
            raise ValueError("context compaction target must be positive and smaller than trigger")
        if (
            self.context_budget_mode == "chars"
            and self.context_compaction_enable
            and self.max_context_chars < trigger
        ):
            raise ValueError(
                "max_context_chars must be greater than or equal to "
                "context_compaction_trigger_chars when compaction is enabled"
            )
        if self.context_preserve_recent_groups < 1:
            raise ValueError("context_preserve_recent_groups must be positive")
        object.__setattr__(self, "context_compaction_trigger_chars", int(trigger))
        object.__setattr__(self, "context_compaction_target_chars", int(target))
        token_trigger = self.context_compaction_trigger_tokens or self.max_context_tokens
        token_target = self.context_compaction_target_tokens or max(1, int(token_trigger * 0.75))
        if token_trigger < 1 or token_target < 1 or token_target >= token_trigger:
            raise ValueError(
                "token context compaction target must be positive and smaller than trigger"
            )
        if (
            self.context_budget_mode == "tokens"
            and self.context_compaction_enable
            and self.max_context_tokens < token_trigger
        ):
            raise ValueError(
                "max_context_tokens must be greater than or equal to "
                "context_compaction_trigger_tokens when token compaction is enabled"
            )
        object.__setattr__(self, "context_compaction_trigger_tokens", int(token_trigger))
        object.__setattr__(self, "context_compaction_target_tokens", int(token_target))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        policy = DEFAULT_CONTEXT_POLICY.to_dict()
        policy["evaluation_budget_mode"] = self.context_budget_mode
        policy["teacher_chars_per_token_safety_estimate"] = (
            TEACHER_CHARS_PER_TOKEN_ESTIMATE
        )
        policy["effective_trigger_chars"] = int(
            self.context_compaction_trigger_chars or self.max_context_chars
        )
        policy["effective_target_chars"] = int(
            self.context_compaction_target_chars or self.max_context_chars
        )
        policy["effective_max_tokens"] = self.max_context_tokens
        policy["effective_trigger_tokens"] = int(
            self.context_compaction_trigger_tokens or self.max_context_tokens
        )
        policy["effective_target_tokens"] = int(
            self.context_compaction_target_tokens or self.max_context_tokens
        )
        payload["context_policy"] = policy
        return payload


def make_violation(
    code: str,
    message: str,
    *,
    component: str,
    always_enforced: bool = False,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "component": component,
        "always_enforced": always_enforced,
    }


def _bounded_string(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    tail = min(800, max(100, limit // 5))
    head = max(0, limit - tail)
    return value[:head] + f"\n<TRUNCATED {len(value) - limit} CHARS>\n" + value[-tail:]


def _project_value(value: Any, *, string_limit: int, list_limit: int, depth: int) -> Any:
    if depth <= 0:
        return "<MAX_DEPTH_REACHED>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_string(value, string_limit)
    if isinstance(value, Mapping):
        priority = {
            "error": 0,
            "message": 1,
            "status": 2,
            "task_id": 3,
            "instruction": 4,
            "app": 5,
            "api": 6,
            "parameters": 7,
            "results": 8,
            "data": 9,
            "next_page": 10,
            "page": 11,
        }
        items = sorted(value.items(), key=lambda item: (priority.get(str(item[0]), 100), str(item[0])))
        return {
            str(key): _project_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth - 1,
            )
            for key, item in items
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
        if len(rows) <= list_limit:
            selected = rows
        else:
            tail_count = min(3, max(1, list_limit // 5))
            selected = [
                *rows[: list_limit - tail_count],
                {"truncated_items": len(rows) - list_limit},
                *rows[-tail_count:],
            ]
        return [
            _project_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth - 1,
            )
            for item in selected
        ]
    return _bounded_string(str(value), string_limit)


def project_observation(value: Any, *, max_chars: int) -> Any:
    """Project an observation deterministically while retaining IDs, errors, and list tails."""

    for string_limit, list_limit, depth in (
        (4_000, 30, 8),
        (2_000, 16, 7),
        (800, 8, 6),
        (300, 4, 5),
    ):
        projected = _project_value(
            value,
            string_limit=string_limit,
            list_limit=list_limit,
            depth=depth,
        )
        if len(canonical_json(projected)) <= max_chars:
            return projected

    rendered = canonical_json(projected)
    tail = min(1_000, max(100, max_chars // 5))
    head = max(0, max_chars - tail - 200)
    return {
        "observation_truncated": True,
        "original_projected_chars": len(rendered),
        "preview_head": rendered[:head],
        "preview_tail": rendered[-tail:],
    }


def context_size_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return len(canonical_json(list(messages)))


def build_bounded_context(
    *,
    base_messages: Sequence[Mapping[str, Any]],
    complete_groups: Sequence[Sequence[Mapping[str, Any]]],
    max_chars: int,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Drop only oldest complete assistant/tool groups to satisfy an input budget."""

    base = [dict(message) for message in base_messages]
    groups = [[dict(message) for message in group] for group in complete_groups]
    dropped = 0
    while True:
        messages = [*base, *(message for group in groups for message in group)]
        size = context_size_chars(messages)
        if size <= max_chars:
            return messages, dropped, size, False
        if len(groups) <= 1:
            return messages, dropped, size, True
        groups.pop(0)
        dropped += 1


def build_memory_context(
    *,
    base_messages: Sequence[Mapping[str, Any]],
    complete_groups: Sequence[Sequence[Mapping[str, Any]]],
    memory: MemoryState,
    max_chars: int,
    trigger_chars: int,
    target_chars: int,
    preserve_recent_groups: int = 4,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], bool, int, bool]:
    """Build a teacher request using the same structured-memory policy as RL.

    Before the trigger, the complete history is presented unchanged. Once the
    trigger is crossed, the base system/task messages are retained, a bounded
    deterministic memory block is injected, and only recent complete groups are
    retained. The old group-drop helper remains available solely for legacy
    artifact readers and is not used by the rollout path.
    """

    base = [dict(message) for message in base_messages]
    groups = [[dict(message) for message in group] for group in complete_groups]
    full = [*base, *(message for group in groups for message in group)]
    full_size = context_size_chars(full)
    if not enabled or full_size < trigger_chars:
        return full, False, full_size, full_size > max_chars

    recent = groups[-max(1, int(preserve_recent_groups)) :]
    memory_text = serialize_memory(memory)
    compact_base = [dict(message) for message in base]
    if compact_base and compact_base[0].get("role") == "system":
        compact_base[0]["content"] = str(compact_base[0].get("content") or "") + "\n\n" + memory_text
    else:
        compact_base.insert(0, {"role": "system", "content": memory_text})

    def render() -> list[dict[str, Any]]:
        return [*compact_base, *(message for group in recent for message in group)]

    messages = render()
    while context_size_chars(messages) > target_chars and len(recent) > 1:
        recent.pop(0)
        messages = render()
    size = context_size_chars(messages)
    return messages, True, size, size > max_chars


def build_memory_token_context(
    *,
    base_messages: Sequence[Mapping[str, Any]],
    complete_groups: Sequence[Sequence[Mapping[str, Any]]],
    memory: MemoryState,
    token_counter: Callable[[Sequence[Mapping[str, Any]]], int],
    max_tokens: int,
    trigger_tokens: int,
    target_tokens: int,
    preserve_recent_groups: int = 4,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], bool, int, bool]:
    """Build the same memory context using an exact model tokenizer budget."""

    base = [dict(message) for message in base_messages]
    groups = [[dict(message) for message in group] for group in complete_groups]
    full = [*base, *(message for group in groups for message in group)]
    full_size = int(token_counter(full))
    if not enabled or full_size < trigger_tokens:
        return full, False, full_size, full_size > max_tokens

    recent = groups[-max(1, int(preserve_recent_groups)) :]
    memory_text = serialize_memory(memory)
    compact_base = [dict(message) for message in base]
    if compact_base and compact_base[0].get("role") == "system":
        compact_base[0]["content"] = (
            str(compact_base[0].get("content") or "") + "\n\n" + memory_text
        )
    else:
        compact_base.insert(0, {"role": "system", "content": memory_text})

    def render() -> list[dict[str, Any]]:
        return [*compact_base, *(message for group in recent for message in group)]

    messages = render()
    size = int(token_counter(messages))
    while size > target_tokens and recent:
        recent.pop(0)
        messages = render()
        size = int(token_counter(messages))
    return messages, True, size, size > max_tokens


def correction_payload(
    violations: Sequence[Mapping[str, Any]],
    *,
    executed: bool,
) -> dict[str, Any]:
    return {
        "harness_correction": True,
        "executed": executed,
        "violations": [
            {"code": str(item.get("code")), "message": str(item.get("message"))}
            for item in violations
        ],
        "instruction": (
            "Choose one grounded tool action using the task, latest observation, and exact API schema. "
            "There is no prescribed tool order."
        ),
    }
