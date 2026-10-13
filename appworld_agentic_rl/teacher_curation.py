"""Deterministic fingerprinting and diversity-aware teacher trajectory curation."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonicalize_code(code: str) -> dict[str, str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        normalized = "\n".join(line.strip() for line in str(code).strip().splitlines())
        return {"format": "text", "value": normalized}
    return {
        "format": "python_ast",
        "value": ast.dump(tree, annotate_fields=True, include_attributes=False),
    }


def _canonicalize_argument(value: Any, *, key: str | None = None) -> Any:
    if key == "code" and isinstance(value, str):
        return _canonicalize_code(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _canonicalize_argument(item, key=str(item_key))
            for item_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_argument(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _trajectory_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def canonical_actions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return action-only tool calls, excluding thoughts and volatile observations."""

    actions: list[dict[str, Any]] = []
    for step in _trajectory_steps(record):
        arguments = step.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        actions.append(
            {
                "tool_name": str(step.get("tool_name") or "invalid_tool_call"),
                "arguments": _canonicalize_argument(arguments),
            }
        )
    return actions


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def extract_planned_api_calls(code: str) -> list[str]:
    """Extract ordered ``apis.<app>.<api>`` calls from Python without executing it."""

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return []
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path is None or len(path) != 3 or path[0] != "apis":
            continue
        calls.append(
            (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                ".".join(path),
            )
        )
    return [value for _, _, value in sorted(calls)]


def canonical_api_plan(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe the observed action sequence and planned AppWorld API calls."""

    plan: list[dict[str, Any]] = []
    for step in _trajectory_steps(record):
        tool_name = str(step.get("tool_name") or "invalid_tool_call")
        arguments = step.get("arguments")
        planned_calls: list[str] = []
        if isinstance(arguments, Mapping) and isinstance(arguments.get("code"), str):
            planned_calls = extract_planned_api_calls(str(arguments["code"]))
        elif tool_name == "complete_task":
            planned_calls = ["apis.supervisor.complete_task"]
        plan.append({"tool_name": tool_name, "planned_api_calls": planned_calls})
    return plan


def canonical_state_delta(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fingerprint auditable state-changing events without including observations."""

    events: list[dict[str, Any]] = []
    for step in _trajectory_steps(record):
        if not bool(step.get("state_change")):
            continue
        metadata = step.get("metadata")
        api_events: list[Any] = []
        state_delta: Any = None
        if isinstance(metadata, Mapping):
            if isinstance(metadata.get("api_events"), list):
                api_events = list(metadata["api_events"])
            state_delta = metadata.get("state_delta")
        events.append(
            {
                "tool_name": str(step.get("tool_name") or "invalid_tool_call"),
                "api_events": _canonicalize_argument(api_events),
                "state_delta": _canonicalize_argument(state_delta),
            }
        )
    return events


def process_quality(record: Mapping[str, Any]) -> tuple[float, dict[str, int]]:
    actions = canonical_actions(record)
    steps = _trajectory_steps(record)
    execution_errors = sum(bool(step.get("error")) for step in steps)
    immediate_repeats = sum(
        actions[index] == actions[index - 1] for index in range(1, len(actions))
    )
    step_count = len(steps)
    rollout = record.get("rollout")
    invalid_actions = (
        int(rollout.get("invalid_actions", 0)) if isinstance(rollout, Mapping) else 0
    )
    harness = record.get("harness")
    guard_events = harness.get("guard_events") if isinstance(harness, Mapping) else []
    guard_enforcements = (
        sum(bool(event.get("enforced")) for event in guard_events if isinstance(event, Mapping))
        if isinstance(guard_events, list)
        else 0
    )
    # Selection uses a strict lexicographic key, so an error cannot be hidden by brevity.
    score = max(
        0.0,
        1.0
        - 0.25 * execution_errors
        - 0.10 * immediate_repeats
        - 0.05 * invalid_actions
        - min(0.25, step_count / 200.0),
    )
    return round(score, 6), {
        "execution_errors": execution_errors,
        "immediate_repeats": immediate_repeats,
        "step_count": step_count,
        "invalid_actions": invalid_actions,
        "guard_enforcements": guard_enforcements,
    }


def enrich_rollout_fingerprints(record: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(record))
    actions = canonical_actions(output)
    plan = canonical_api_plan(output)
    state_delta = canonical_state_delta(output)
    score, metrics = process_quality(output)
    output["action_hash"] = _digest(actions)
    output["api_plan_hash"] = _digest(plan)
    output["state_delta_hash"] = _digest(state_delta)
    output["process_score"] = score
    output["process_metrics"] = metrics
    output.setdefault("duplicate_of", None)
    output.setdefault("selected_for_sft", False)
    return output


def _quality_key(record: Mapping[str, Any]) -> tuple[int, int, int, int, str, str]:
    metrics = record.get("process_metrics")
    if not isinstance(metrics, Mapping):
        _, metrics = process_quality(record)
    return (
        int(metrics.get("execution_errors", 0)),
        int(metrics.get("invalid_actions", 0)),
        int(metrics.get("immediate_repeats", 0)),
        int(metrics.get("step_count", 0)),
        str(record.get("action_hash") or ""),
        str(record.get("rollout_id") or ""),
    )


def curate_teacher_rollouts(
    records: Sequence[Mapping[str, Any]],
    *,
    max_curated_per_task: int = 2,
    require_clean_for_sft: bool = True,
) -> list[dict[str, Any]]:
    """Deduplicate verifier-passing rollouts and select diverse SFT representatives."""

    if max_curated_per_task < 1:
        raise ValueError("max_curated_per_task must be positive")
    enriched = [enrich_rollout_fingerprints(record) for record in records]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in enriched:
        record["duplicate_of"] = None
        record["selected_for_sft"] = False
        record.pop("curation_exclusion_reason", None)
        by_task[str(record.get("task_id") or "")].append(record)

    for task_records in by_task.values():
        accepted = [
            record
            for record in task_records
            if record.get("collection_status") == "accepted"
            and not bool(record.get("infrastructure_invalid"))
        ]
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in accepted:
            clusters[str(record["api_plan_hash"])].append(record)

        representatives: list[dict[str, Any]] = []
        for cluster in clusters.values():
            ordered = sorted(cluster, key=_quality_key)
            representative = ordered[0]
            representatives.append(representative)
            for duplicate in ordered[1:]:
                duplicate["duplicate_of"] = representative.get("rollout_id")
                duplicate["curation_exclusion_reason"] = "same_api_action_plan"

        ordered_representatives = sorted(representatives, key=_quality_key)
        eligible_representatives: list[dict[str, Any]] = []
        for record in ordered_representatives:
            metrics = record.get("process_metrics")
            execution_errors = (
                int(metrics.get("execution_errors", 0))
                if isinstance(metrics, Mapping)
                else 0
            )
            if require_clean_for_sft and execution_errors:
                record["curation_exclusion_reason"] = "execution_errors"
                continue
            eligible_representatives.append(record)

        for record in eligible_representatives[:max_curated_per_task]:
            record["selected_for_sft"] = True
        for record in eligible_representatives[max_curated_per_task:]:
            record["curation_exclusion_reason"] = "max_curated_per_task"
        for record in task_records:
            if bool(record.get("infrastructure_invalid")):
                record["curation_exclusion_reason"] = "infrastructure_invalid"
            elif record.get("collection_status") != "accepted":
                record["curation_exclusion_reason"] = "verifier_rejected"
    return enriched
