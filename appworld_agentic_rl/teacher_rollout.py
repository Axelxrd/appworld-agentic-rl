"""Teacher-driven agent-tool rollouts over the AppWorld adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import re
from typing import Any

from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS, parse_tool_call
from appworld_agentic_rl.reward import infer_evaluation_success
from appworld_agentic_rl.teacher_client import (
    TeacherModelClient,
    TeacherModelError,
    TeacherProtocolError,
    TeacherProviderError,
    TeacherQuotaError,
    normalize_assistant_message,
)
from appworld_agentic_rl.harness import (
    HARNESS_VERSION,
    TeacherHarnessConfig,
    build_memory_context,
    build_memory_token_context,
    context_size_chars,
    correction_payload,
    make_violation,
    project_observation,
    stable_hash,
)
from appworld_agentic_rl.context_memory import MemoryState, update_memory
from appworld_agentic_rl.context_policy import TEACHER_CHARS_PER_TOKEN_ESTIMATE


TEACHER_ROLLOUT_SCHEMA_VERSION = "appworld-teacher-rollout-v2"

SYSTEM_PROMPT = """You are a teacher agent solving one AppWorld task through agent-selected tools.

Your objective is to satisfy the task instruction with the smallest safe, evidence-grounded
sequence of actions. Use only the supplied tools. The environment defines the available tool
interfaces, but it does not prescribe their order: choose each next tool, its arguments, whether
to revisit earlier evidence, and when to finish from the task and latest observation. Never read local task files, benchmark ground
truth, solution modules, evaluator code, Python package source, or runtime internals. Do not use
dir(), inspect, getsource, imports of appworld, HTTP requests, or another client namespace.

STRICT SERIAL TOOL PROTOCOL:
- Return exactly one tool call in every assistant turn. Never return multiple tool_calls.
- Wait for the tool observation before choosing the next action; all tools share one mutable
  AppWorld session, so an unobserved later action is invalid rather than an optimization.
- Do not output a final answer or claim success in plain text. End with complete_task(...) or,
  only when no safe valid completion exists, finish_without_completion(...).

NON-SEQUENTIAL TOOL-SELECTION PRINCIPLES:
- Use inspect_task whenever the task context or allowed apps is not yet clear.
- For an unfamiliar API, use search_api_docs and inspect_api_schema for the exact
   app/API returned by the documentation. Never invent names such as
   supervisor.get_account_info, supervisor.login, app_world, apps, client, api, or call_api.
   A target app's login API is valid only after its exact schema has been returned by the docs.
- Use query_state for read-only AppWorld calls and evidence gathering. Use run_code only for
   non-mutating computation or formatting. Use run_write_code only for an intended state change.
- Choose whether more evidence, a write, verification, recovery, or termination is appropriate
  from the current state. After a write, verify the requested change before claiming success.
  For question tasks, compute the answer from observed entities and pass it as
  complete_task(answer=...).

AGENT CODE CONTRACT:
query_state, run_code, and run_write_code execute Python in one persistent session with a
preloaded global named `apis`. Call every documented API exactly as `apis.<app>.<api>(...)` and
use keyword arguments matching its schema. Variables created in one agent code call persist
to later calls. When authentication is required, first discover the exact supervisor/account
API names from documentation, consume passwords and tokens only inside code, and retain tokens
in variables for subsequent documented calls. In the standard supervisor contract,
`show_account_passwords()` returns a list of records with `account_name` and `password`, while
`show_profile()` provides the supervisor email; a password record does not contain `username`.
Use that email as the username for the target app's documented login API. Never call a guessed
`supervisor.login` or invent credential fields.

ERROR RECOVERY:
Read the complete latest observation after every tool call. If a tool returns an error, do not
repeat the identical call. Correct one grounded issue using the error and API schema, or search
the documentation again. Do not guess a replacement API or parameter. Avoid repeated searches,
repeated writes, and loops that produce no new evidence.

The simulated environment may expose account passwords and access tokens. Never print, quote,
summarize, or return credentials, access tokens, API keys, authorization values, or their literal
values in assistant text or code output. Inspect state before a write and make only necessary
state changes."""

# Frozen policy evaluation uses the same tool contract, but must not frame the
# evaluated model as a teacher. Keep this separate to preserve collection provenance.
POLICY_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "You are a teacher agent solving one AppWorld task through agent-selected tools.",
    "You are an AppWorld agent solving one task through agent-selected tools.",
    1,
)

_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:password|access_token|api_key|authorization|secret|credential)(?:$|_)",
    re.IGNORECASE,
)
_QUOTED_SECRET_RE = re.compile(
    r"(?i)((?:['\"])?\b(?:password|access[_ -]?token|api[_ -]?key|authorization|secret)\b"
    r"(?:['\"])?\s*[:=]\s*['\"])([^'\"]*)(['\"])",
)
_BARE_SECRET_RE = re.compile(
    r"(?im)((?:['\"])?\b(?:password|access[_ -]?token|api[_ -]?key|authorization|secret)\b"
    r"(?:['\"])?\s*[:=]\s*)([^\s,;\}\]]+)",
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;\}\]]+")


def _is_sensitive_key(key: object) -> bool:
    return bool(_SENSITIVE_KEY_RE.search(str(key).casefold()))


def _redact_text(value: str, *, aggressive: bool) -> str:
    output = _QUOTED_SECRET_RE.sub(r"\1<REDACTED>\3", value)
    output = _BEARER_RE.sub(r"\1<REDACTED>", output)
    if aggressive:
        output = _BARE_SECRET_RE.sub(r"\1<REDACTED>", output)
    return output


def redact_sensitive_data(value: Any, *, field_name: str | None = None) -> Any:
    """Remove simulated credentials before data is persisted or returned to the model."""

    if isinstance(value, Mapping):
        return {
            str(key): "<REDACTED>"
            if _is_sensitive_key(key)
            else redact_sensitive_data(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item, field_name=field_name) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_data(item, field_name=field_name) for item in value]
    if isinstance(value, str):
        # Code can legitimately refer to a variable such as spotify_password. Redact
        # quoted literals there, but retain the variable reference needed for SFT replay.
        return _redact_text(value, aggressive=field_name != "code")
    return value


def sanitize_assistant_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare a model message for history without changing the live tool call."""

    clean = redact_sensitive_data(dict(message))
    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list) or not isinstance(clean.get("tool_calls"), list):
        return clean

    for raw_call, clean_call in zip(raw_tool_calls, clean["tool_calls"], strict=True):
        if not isinstance(raw_call, Mapping) or not isinstance(clean_call, dict):
            continue
        raw_function = raw_call.get("function")
        clean_function = clean_call.get("function")
        if not isinstance(raw_function, Mapping) or not isinstance(clean_function, dict):
            continue
        arguments = raw_function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            clean_function["arguments"] = redact_sensitive_data(arguments)
            continue
        if isinstance(parsed_arguments, Mapping):
            clean_function["arguments"] = json.dumps(
                redact_sensitive_data(parsed_arguments),
                ensure_ascii=False,
                sort_keys=True,
            )
    return clean


def _visible_task_context(initial_observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": initial_observation.get("task_id"),
        "instruction": initial_observation.get("instruction"),
        "current_datetime": initial_observation.get("current_datetime"),
        "allowed_apps": initial_observation.get("allowed_apps"),
        "agent_code_contract": {
            "api_call_syntax": "apis.<app>.<api>(...)",
            "preloaded_global": "apis",
            "state_persistence": "Python variables persist across agent code calls.",
            "forbidden_alternatives": [
                "app_world",
                "apps",
                "client",
                "call_api",
                "HTTP requests",
                "runtime introspection",
            ],
        },
    }


def _tool_name(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "invalid_tool_call")
    return str(tool_call.get("name") or "invalid_tool_call")


def _failure_record(
    *,
    task_id: str,
    split: str,
    error: Exception,
    harness_config: TeacherHarnessConfig,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": TEACHER_ROLLOUT_SCHEMA_VERSION,
        "task_id": task_id,
        "split": split,
        "rollout": {
            "end_reason": "environment_error",
            "interaction_end_reason": "environment_error",
            "model_turns": 0,
            "action_attempts": 0,
            "environment_steps": 0,
            "invalid_actions": 0,
            "dropped_tool_calls": [],
            "errors": [type(error).__name__ + ": " + str(error)],
        },
        "harness": {
            "version": HARNESS_VERSION,
            "config": harness_config.to_dict(),
            "system_prompt_hash": stable_hash(system_prompt),
            "tool_schema_hash": stable_hash(AGENT_TOOL_SCHEMAS),
            "guard_events": [],
            "context_events": [],
        },
        "conversation": [],
        "observations": [],
        "trajectory": None,
        "evaluation": {"error": type(error).__name__ + ": " + str(error)},
        "terminal_reward": None,
        "infrastructure_invalid": True,
    }


def _telemetry_snapshot(client: TeacherModelClient) -> dict[str, Any]:
    snapshot = getattr(client, "telemetry_snapshot", None)
    if not callable(snapshot):
        return {}
    value = snapshot()
    return dict(value) if isinstance(value, Mapping) else {}


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("request_count", "retry_count", "latency_seconds"):
        old = before.get(key, 0)
        new = after.get(key, 0)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            value = new - old
            output[key] = round(value, 6) if isinstance(value, float) else value
    before_usage = before.get("usage") if isinstance(before.get("usage"), Mapping) else {}
    after_usage = after.get("usage") if isinstance(after.get("usage"), Mapping) else {}
    usage: dict[str, int] = {}
    for key in set(before_usage) | set(after_usage):
        old = before_usage.get(key, 0)
        new = after_usage.get(key, 0)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            usage[str(key)] = int(new - old)
    output["usage"] = dict(sorted(usage.items()))
    return output


def _guard_tool_call(adapter: Any, tool_call: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        parse_tool_call(tool_call)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return [
            make_violation(
                "invalid_tool_arguments",
                type(exc).__name__ + ": " + str(exc),
                component="protocol",
                always_enforced=True,
            )
        ]
    validator = getattr(adapter, "validate_tool_call", None)
    if not callable(validator):
        return []
    violations = validator(tool_call)
    if not isinstance(violations, list):
        raise TypeError("adapter.validate_tool_call() must return a list")
    return [dict(item) for item in violations if isinstance(item, Mapping)]


def _not_executed_result(
    *,
    tool_name: str,
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "observation": correction_payload(violations, executed=False),
        "error": None,
        "state_change": False,
        "terminated": False,
        "executed": False,
    }


def run_teacher_rollout(
    *,
    adapter: Any,
    client: TeacherModelClient,
    task_id: str,
    split: str = "train",
    max_model_turns: int = 25,
    harness_config: TeacherHarnessConfig | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    context_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
) -> dict[str, Any]:
    """Run one teacher policy until it terminates, fails, or reaches a bounded budget."""

    if max_model_turns < 1:
        raise ValueError("max_model_turns must be positive")
    config = harness_config or TeacherHarnessConfig()
    if config.context_budget_mode == "tokens" and context_token_counter is None:
        raise ValueError("token context budgeting requires context_token_counter")
    telemetry_before = _telemetry_snapshot(client)

    try:
        initial_observation = adapter.reset(task_id, split=split)
    except Exception as exc:
        return redact_sensitive_data(
            _failure_record(
                task_id=task_id,
                split=split,
                error=exc,
                harness_config=config,
                system_prompt=system_prompt,
            )
        )
    if not isinstance(initial_observation, Mapping):
        initial_observation = {"instruction": str(initial_observation)}

    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                _visible_task_context(initial_observation), ensure_ascii=False, sort_keys=True
            ),
        },
    ]
    complete_groups: list[list[dict[str, Any]]] = []
    training_messages: list[dict[str, Any]] = [dict(message) for message in base_messages]
    memory = MemoryState()
    interaction_end_reason = "max_model_turns"
    errors: list[str] = []
    dropped_tool_calls: list[dict[str, Any]] = []
    guard_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    model_turns = 0
    action_attempts = 0
    environment_steps = 0
    invalid_actions = 0
    previous_tool_signature: str | None = None
    identical_repeats = 0
    terminal_reached = False
    infrastructure_invalid = False

    for turn_index in range(max_model_turns):
        if config.context_budget_mode == "tokens":
            assert context_token_counter is not None
            request_messages, compacted, context_tokens, exhausted = (
                build_memory_token_context(
                    base_messages=base_messages,
                    complete_groups=complete_groups,
                    memory=memory,
                    token_counter=context_token_counter,
                    max_tokens=config.max_context_tokens,
                    trigger_tokens=int(
                        config.context_compaction_trigger_tokens
                        or config.max_context_tokens
                    ),
                    target_tokens=int(
                        config.context_compaction_target_tokens
                        or config.max_context_tokens
                    ),
                    preserve_recent_groups=config.context_preserve_recent_groups,
                    enabled=config.context_compaction_enable,
                )
            )
            context_chars = context_size_chars(request_messages)
        else:
            request_messages, compacted, context_chars, exhausted = build_memory_context(
                base_messages=base_messages,
                complete_groups=complete_groups,
                memory=memory,
                max_chars=config.max_context_chars,
                trigger_chars=int(
                    config.context_compaction_trigger_chars or config.max_context_chars
                ),
                target_chars=int(
                    config.context_compaction_target_chars or config.max_context_chars
                ),
                preserve_recent_groups=config.context_preserve_recent_groups,
                enabled=config.context_compaction_enable,
            )
            context_tokens = math.ceil(
                context_chars / TEACHER_CHARS_PER_TOKEN_ESTIMATE
            )
        if compacted:
            # This is the exact context used for the next teacher decision. It
            # becomes the SFT-visible stream after the first compaction.
            training_messages = [dict(message) for message in request_messages]
        context_events.append(
            {
                "turn_index": turn_index,
                "budget_mode": config.context_budget_mode,
                "input_chars": context_chars,
                "input_tokens": context_tokens if config.context_budget_mode == "tokens" else None,
                "estimated_input_tokens": context_tokens,
                "token_count_is_exact": config.context_budget_mode == "tokens",
                "compacted": compacted,
                "context_policy": "structured_memory" if compacted else "full_context",
                "memory_version": memory.version,
                "preserved_recent_groups": config.context_preserve_recent_groups if compacted else len(complete_groups),
            }
        )
        if exhausted:
            interaction_end_reason = "context_exhausted"
            infrastructure_invalid = True
            if config.context_budget_mode == "tokens":
                errors.append(
                    "minimum context requires "
                    f"{context_tokens} tokens, budget is {config.max_context_tokens}"
                )
            else:
                errors.append(
                    f"minimum context requires {context_chars} chars, "
                    f"budget is {config.max_context_chars}"
                )
            break
        model_turns += 1
        try:
            assistant_message = normalize_assistant_message(
                client.generate(request_messages, AGENT_TOOL_SCHEMAS)
            )
        except TeacherQuotaError:
            # A quota circuit breaker belongs to collection accounting, where the
            # current attempt can be left pending and resumed without a raw row.
            raise
        except TeacherProviderError as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            interaction_end_reason = "provider_error"
            infrastructure_invalid = True
            break
        except (TeacherProtocolError, ValueError, TypeError) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            interaction_end_reason = "protocol_error"
            infrastructure_invalid = True
            break
        except TeacherModelError as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            interaction_end_reason = "provider_error"
            infrastructure_invalid = True
            break
        except Exception as exc:  # Keep collection resumable across provider-specific failures.
            errors.append(type(exc).__name__ + ": " + str(exc))
            interaction_end_reason = "provider_error"
            infrastructure_invalid = True
            break

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            group = [sanitize_assistant_message(assistant_message)]
            violation = make_violation(
                "missing_terminal_tool",
                "Assistant stopped without complete_task or finish_without_completion.",
                component="behavior_guard",
            )
            enforced = config.bc_guard_mode == "enforce"
            guard_events.append(
                {
                    **violation,
                    "turn_index": turn_index,
                    "action_attempt_index": None,
                    "enforced": enforced,
                }
            )
            if enforced:
                invalid_actions += 1
                group.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            correction_payload([violation], executed=False),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
                complete_groups.append(group)
                if invalid_actions >= config.invalid_action_limit:
                    interaction_end_reason = "invalid_action_limit"
                    break
                continue
            complete_groups.append(group)
            interaction_end_reason = "model_stopped_without_tool"
            break

        call_ids = [str(call.get("id") or "") for call in tool_calls]
        if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
            errors.append("assistant tool_call IDs must be present and unique within a turn")
            interaction_end_reason = "protocol_error"
            infrastructure_invalid = True
            break

        group = [sanitize_assistant_message(assistant_message)]
        batch_terminal = False
        environment_failed = False
        parallel_drop: dict[str, Any] | None = None
        if len(tool_calls) > 1 and config.bc_guard_mode == "enforce":
            parallel_drop = {
                "turn_index": turn_index,
                "count": len(tool_calls) - 1,
                "tool_names": [_tool_name(call) for call in tool_calls[1:]],
            }
            dropped_tool_calls.append(parallel_drop)

        for call_index, tool_call in enumerate(tool_calls):
            action_attempts += 1
            parsed_name = _tool_name(tool_call)
            parsed_args: dict[str, Any] = {}
            violations: list[dict[str, Any]] = []
            if action_attempts > config.max_action_attempts:
                violations.append(
                    make_violation(
                        "max_action_attempts",
                        f"Maximum action attempts {config.max_action_attempts} reached.",
                        component="protocol",
                        always_enforced=True,
                    )
                )
            if len(tool_calls) > 1 and call_index > 0:
                violations.append(
                    make_violation(
                        "parallel_tool_calls",
                        "Tool actions require the preceding observation and must be serial.",
                        component="behavior_guard",
                    )
                )
            try:
                parsed_name, parsed_args = parse_tool_call(tool_call)
                signature = json.dumps(
                    {"name": parsed_name, "arguments": parsed_args},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                signature = None
            if signature is not None and signature == previous_tool_signature:
                identical_repeats += 1
            else:
                identical_repeats = 0
            previous_tool_signature = signature
            if identical_repeats >= config.max_identical_action_repeats:
                violations.append(
                    make_violation(
                        "repeated_action",
                        "This exact action was already attempted against the same latest state.",
                        component="behavior_guard",
                    )
                )
            try:
                violations.extend(_guard_tool_call(adapter, tool_call))
            except Exception as exc:
                errors.append("guard_error: " + type(exc).__name__ + ": " + str(exc))
                interaction_end_reason = "environment_error"
                infrastructure_invalid = True
                environment_failed = True
                violations.append(
                    make_violation(
                        "guard_failure",
                        "AppWorld guard could not validate this action.",
                        component="core_safety",
                        always_enforced=True,
                    )
                )

            blocked_by_terminal = batch_terminal
            blocked_by_parallel = config.bc_guard_mode == "enforce" and call_index > 0
            blocked_by_violation = any(
                bool(item.get("always_enforced"))
                or config.bc_guard_mode == "enforce"
                for item in violations
            )
            execute = not (
                blocked_by_terminal
                or blocked_by_parallel
                or blocked_by_violation
                or environment_failed
            )
            for violation in violations:
                guard_events.append(
                    {
                        **violation,
                        "turn_index": turn_index,
                        "action_attempt_index": action_attempts,
                        "tool_call_id": str(tool_call.get("id")),
                        "tool_name": parsed_name,
                        "enforced": not execute,
                    }
                )
            if blocked_by_terminal:
                terminal_violation = make_violation(
                    "terminal_already_reached",
                    "A prior call in this batch terminated the environment.",
                    component="protocol",
                    always_enforced=True,
                )
                violations.append(terminal_violation)
                guard_events.append(
                    {
                        **terminal_violation,
                        "turn_index": turn_index,
                        "action_attempt_index": action_attempts,
                        "tool_call_id": str(tool_call.get("id")),
                        "tool_name": parsed_name,
                        "enforced": True,
                    }
                )
            if execute:
                try:
                    result = adapter.step(tool_call)
                    environment_steps += 1
                    if not isinstance(result, Mapping):
                        raise TypeError("adapter.step() must return an object")
                    result = dict(result)
                    result["executed"] = True
                except Exception as exc:
                    error_text = type(exc).__name__ + ": " + str(exc)
                    result = {
                        "tool_name": parsed_name,
                        "observation": {"error": error_text},
                        "error": error_text,
                        "state_change": False,
                        "terminated": False,
                        "executed": False,
                    }
                    errors.append("environment_error: " + error_text)
                    interaction_end_reason = "environment_error"
                    infrastructure_invalid = True
                    environment_failed = True
            else:
                invalid_actions += 1
                result = _not_executed_result(
                    tool_name=parsed_name,
                    violations=violations,
                )

            if result.get("error") and not result.get("terminated"):
                result["agent_code_reminder"] = (
                    "Use exact documented calls as apis.<app>.<api>(...) with schema-matching "
                    "keyword arguments; use a new grounded action rather than repeating the failure."
                )
            raw_result = redact_sensitive_data(result)
            visible_result = project_observation(
                raw_result,
                max_chars=config.max_observation_chars,
            )
            observations.append(
                {
                    "turn_index": turn_index,
                    "action_attempt_index": action_attempts,
                    "tool_call_id": str(tool_call.get("id")),
                    "tool_name": parsed_name,
                    "executed": bool(result.get("executed")),
                    "raw": raw_result,
                    "visible": visible_result,
                }
            )
            update_memory(
                memory,
                {
                    "step_index": len(complete_groups),
                    "tool_name": parsed_name,
                    "arguments": parsed_args,
                    "observation": visible_result,
                    "error": result.get("error"),
                    "metadata": result.get("metadata") if isinstance(result, Mapping) else {},
                },
            )
            group.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id")),
                    "name": parsed_name,
                    "content": json.dumps(visible_result, ensure_ascii=False, sort_keys=True),
                }
            )
            if bool(result.get("terminated")):
                batch_terminal = True
                terminal_reached = True

        complete_groups.append(group)
        training_messages.extend(group)
        if environment_failed:
            break
        if action_attempts >= config.max_action_attempts and not terminal_reached:
            interaction_end_reason = "max_action_attempts"
            break
        if terminal_reached:
            interaction_end_reason = "terminal_tool"
            break
        if invalid_actions >= config.invalid_action_limit:
            recent_codes = {
                str(event.get("code"))
                for event in guard_events
                if int(event.get("turn_index", -1)) == turn_index and event.get("enforced")
            }
            interaction_end_reason = (
                "repeat_loop" if recent_codes == {"repeated_action"} else "invalid_action_limit"
            )
            break
        adapter_max_steps = getattr(adapter, "max_steps", None)
        if isinstance(adapter_max_steps, int) and environment_steps >= adapter_max_steps:
            interaction_end_reason = "max_environment_steps"
            break
    else:
        interaction_end_reason = "max_model_turns"

    try:
        terminal = adapter.evaluate()
    except Exception as exc:
        terminal = {
            "evaluation": {"error": type(exc).__name__ + ": " + str(exc)},
            "terminal_reward": None,
        }
        errors.append("evaluation_error: " + type(exc).__name__ + ": " + str(exc))
        interaction_end_reason = "evaluator_error"
        infrastructure_invalid = True

    evaluation = terminal.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("error"):
        interaction_end_reason = "evaluator_error"
        infrastructure_invalid = True
    evaluation_success = infer_evaluation_success(evaluation or {}) is True
    if not infrastructure_invalid and evaluation_success and terminal_reached:
        end_reason = "evaluator_success"
    elif not infrastructure_invalid and terminal_reached:
        end_reason = "evaluator_failure"
    else:
        end_reason = interaction_end_reason

    try:
        trajectory = adapter.export_trajectory()
    except Exception as exc:
        trajectory = None
        errors.append("trajectory_export_error: " + type(exc).__name__ + ": " + str(exc))
        infrastructure_invalid = True
        end_reason = "environment_error"

    conversation = [
        *base_messages,
        *(message for group in complete_groups for message in group),
    ]
    telemetry = _telemetry_delta(telemetry_before, _telemetry_snapshot(client))
    telemetry.setdefault("request_count", model_turns)

    record = {
        "schema_version": TEACHER_ROLLOUT_SCHEMA_VERSION,
        "task_id": task_id,
        "split": split,
        "rollout": {
            "end_reason": end_reason,
            "interaction_end_reason": interaction_end_reason,
            "model_turns": model_turns,
            "action_attempts": action_attempts,
            "environment_steps": environment_steps,
            "invalid_actions": invalid_actions,
            "dropped_tool_calls": dropped_tool_calls,
            "errors": errors,
        },
        "harness": {
            "version": HARNESS_VERSION,
            "config": config.to_dict(),
            "system_prompt_hash": stable_hash(system_prompt),
            "tool_schema_hash": stable_hash(AGENT_TOOL_SCHEMAS),
            "guard_events": guard_events,
            "context_events": context_events,
        },
        "provider": {
            "model": getattr(client, "model", None),
            "base_url": getattr(client, "base_url", None),
            "thinking": {
                "mode": "provider_default",
                "reasoning_content_history": True,
            },
            "telemetry": telemetry,
        },
        "conversation": conversation,
        "training_conversation": training_messages,
        "observations": observations,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "terminal_reward": terminal.get("terminal_reward"),
        "infrastructure_invalid": infrastructure_invalid,
    }
    return redact_sensitive_data(record)
