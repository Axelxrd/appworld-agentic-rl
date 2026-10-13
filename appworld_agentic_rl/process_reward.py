"""Deterministic lightweight process rewards for AppWorld trajectories.

The scorer deliberately uses only public trajectory evidence.  App-specific
task semantics remain the responsibility of AppWorld's terminal evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


_API_CALL_RE = re.compile(r"apis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
_READ_TOOLS = {"query_state", "run_code", "inspect_task", "search_api_docs", "inspect_api_schema"}
_WRITE_TOOLS = {"run_write_code"}


@dataclass(frozen=True, slots=True)
class ProcessEvent:
    """One auditable process-reward event aligned to an action step."""

    step_index: int
    delta: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"step_index": self.step_index, "delta": self.delta, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class ProcessRewardResult:
    process_score: float
    events: tuple[ProcessEvent, ...]
    metrics: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_score": self.process_score,
            "events": [event.to_dict() for event in self.events],
            "metrics": dict(self.metrics),
        }


def _stable_signature(tool_name: str, arguments: Any) -> str:
    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{tool_name}:{payload}".encode()).hexdigest()


def _api_keys(arguments: Any) -> set[tuple[str, str]]:
    if not isinstance(arguments, Mapping):
        return set()
    code = arguments.get("code")
    if not isinstance(code, str):
        return set()
    return {(app, api) for app, api in _API_CALL_RE.findall(code)}


def _search_result_keys(observation: Any) -> frozenset[tuple[str, str]]:
    """Return the API set exposed by one search result, ignoring descriptions."""

    if not isinstance(observation, Mapping):
        return frozenset()
    results = observation.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)):
        return frozenset()
    keys: set[tuple[str, str]] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        app, api = item.get("app"), item.get("api")
        if app and api:
            keys.add((str(app), str(api)))
    return frozenset(keys)


def _observation_has_new_fact(observation: Any, seen: set[str]) -> bool:
    """Recognize non-trivial read evidence without rewarding a generic ``ok``."""

    if not isinstance(observation, Mapping) or observation.get("error"):
        return False
    useful = {
        str(key): value
        for key, value in observation.items()
        if str(key) not in {"ok", "status", "message", "query", "results"}
    }
    if not useful:
        return False
    signature = json.dumps(useful, ensure_ascii=False, sort_keys=True, default=str)
    if signature in seen:
        return False
    seen.add(signature)
    return True


def score_trajectory(
    trajectory: Mapping[str, Any] | None,
    *,
    terminal_reward: float = 0.0,
    clip: tuple[float, float] = (-0.2, 0.2),
) -> ProcessRewardResult:
    """Score agent tool behavior without inspecting task ground truth.

    Positive events reward documented discovery, schema use, grounded reads,
    read-before-write, and recovery after an error.  Negative events penalize
    repeated identical calls, invalid/error steps, and writes without a prior
    read.  The terminal reward is used only for the small timely-completion
    adjustment; it is not allowed to dominate the process score.
    """

    raw_steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    steps: Sequence[Mapping[str, Any]] = [x for x in (raw_steps or []) if isinstance(x, Mapping)]
    events: list[ProcessEvent] = []
    seen_signatures: set[str] = set()
    seen_apis: set[tuple[str, str]] = set()
    discovered_apis: set[tuple[str, str]] = set()
    seen_search_sets: set[frozenset[tuple[str, str]]] = set()
    seen_read_apis: set[tuple[str, str]] = set()
    seen_read_facts: set[str] = set()
    grounded_write_events = 0
    discovery_rewarded = False
    schema_rewarded: set[tuple[str, str]] = set()
    read_since_write = False
    previous_error = False
    previous_error_signature: str | None = None
    metrics = {
        "discovery": 0,
        "schema_use": 0,
        "read_before_write": 0,
        "entity_grounding": 0,
        "error_recovery": 0,
        "repeated_action": 0,
        "invalid_action": 0,
        "unsafe_write": 0,
        "repeated_search_result_set": 0,
    }
    category_positive: dict[str, float] = {}
    category_negative: dict[str, float] = {}

    def add_capped(category: str, amount: float, cap: float) -> float:
        """Apply a category-local absolute budget to prevent reward farming."""

        if amount >= 0:
            used = category_positive.get(category, 0.0)
            applied = min(float(amount), max(0.0, cap - used))
            category_positive[category] = used + applied
            return applied
        used = category_negative.get(category, 0.0)
        applied = -min(float(-amount), max(0.0, cap - used))
        category_negative[category] = used - applied
        return applied
    for index, step in enumerate(steps):
        tool = str(step.get("tool_name") or "")
        arguments = step.get("arguments")
        reasons: list[str] = []
        delta = 0.0
        signature = _stable_signature(tool, arguments)
        if signature in seen_signatures and tool not in {"complete_task", "finish_without_completion"}:
            delta -= 0.12
            metrics["repeated_action"] += 1
            reasons.append("repeated_action")
        seen_signatures.add(signature)

        error = step.get("error")
        observation = step.get("observation")
        has_error = bool(error) or bool(isinstance(observation, Mapping) and observation.get("error"))
        if has_error:
            delta -= 0.10
            metrics["invalid_action"] += 1
            reasons.append("invalid_or_error")
        if (
            previous_error
            and not has_error
            and signature != previous_error_signature
            and tool not in {"complete_task", "finish_without_completion"}
        ):
            delta += add_capped("recovery", 0.10, 0.10)
            metrics["error_recovery"] += 1
            reasons.append("error_recovery")
        previous_error = has_error
        previous_error_signature = signature if has_error else None

        if tool == "search_api_docs":
            result_keys = _search_result_keys(observation)
            if result_keys and result_keys in seen_search_sets:
                delta += add_capped("search_duplicate", -0.02, 0.10)
                reasons.append("duplicate_search_result_set")
                metrics.setdefault("repeated_search_result_set", 0)
                metrics["repeated_search_result_set"] += 1
            elif not discovery_rewarded:
                delta += add_capped("discovery", 0.05 if result_keys else 0.03, 0.05)
                metrics["discovery"] += 1
                reasons.append("api_discovery")
                discovery_rewarded = True
                if result_keys:
                    seen_search_sets.add(result_keys)
            elif result_keys:
                new_keys = result_keys - discovered_apis
                if new_keys:
                    delta += add_capped("discovery", 0.01, 0.05)
                    reasons.append("new_api_discovery")
                else:
                    delta += add_capped("search_duplicate", -0.02, 0.10)
                    reasons.append("duplicate_search_result_set")
                seen_search_sets.add(result_keys)
            if result_keys:
                discovered_apis.update(result_keys)
        elif tool == "inspect_api_schema":
            schema_key = (
                str(arguments.get("app") or ""),
                str(arguments.get("api") or ""),
            ) if isinstance(arguments, Mapping) else ("", "")
            if schema_key not in schema_rewarded:
                delta += add_capped("grounded_read", 0.03, 0.06)
                metrics["schema_use"] += 1
                reasons.append("schema_use")
                schema_rewarded.add(schema_key)
        elif tool in _READ_TOOLS:
            api_keys = _api_keys(arguments)
            seen_read_apis.update(api_keys)
            has_new_fact = _observation_has_new_fact(observation, seen_read_facts)
            read_since_write = bool(seen_read_apis)
            if tool in {"query_state", "run_code"} and has_new_fact:
                delta += 0.03
                metrics["entity_grounding"] += 1
                reasons.append("grounded_read")
        elif tool in _WRITE_TOOLS:
            write_apis = _api_keys(arguments)
            metadata = step.get("metadata")
            has_provenance_contract = isinstance(metadata, Mapping) and "write_provenance" in metadata
            provenance = metadata.get("write_provenance", []) if has_provenance_contract else []
            provenance_rows = [item for item in provenance if isinstance(item, Mapping)]
            provenance_verified = bool(provenance_rows) and len(provenance_rows) == len(provenance) and all(
                bool(item.get("all_required_verified")) for item in provenance_rows
            )
            read_apps = {app for app, _ in seen_read_apis}
            legacy_grounded = (
                not has_provenance_contract
                and read_since_write
                and (not write_apis or any(app in read_apps for app, _ in write_apis))
            )
            write_is_grounded = bool(provenance_verified or legacy_grounded)
            if write_is_grounded:
                delta += add_capped("read_before_write", 0.06, 0.12)
                metrics["read_before_write"] += 1
                grounded_write_events += 1
                reasons.append("read_before_write")
            else:
                delta -= 0.12
                metrics["unsafe_write"] += 1
                reasons.append("write_without_read")
                if has_provenance_contract:
                    reasons.append("unverified_write_parameters")
                    metrics.setdefault("unverified_write_parameters", 0)
                    metrics["unverified_write_parameters"] += 1
            read_since_write = False

        api_keys = _api_keys(arguments)
        if api_keys & seen_apis:
            delta -= 0.03
            reasons.append("reused_entity_without_new_evidence")
        seen_apis.update(api_keys)
        events.append(ProcessEvent(index, delta, tuple(reasons)))

    # Reward finishing promptly after a non-error action, while keeping this
    # term small enough that evaluator success remains the primary signal.
    if steps and str(steps[-1].get("tool_name")) == "complete_task" and not previous_error:
        timely = add_capped("timely_completion", 0.05, 0.05) if terminal_reward > 0.9 else 0.0
        if timely:
            events[-1] = ProcessEvent(
                events[-1].step_index,
                events[-1].delta + timely,
                events[-1].reasons + ("timely_completion",),
            )
    # Mean aggregation prevents a policy from increasing reward merely by
    # emitting more low-value discovery actions.  Failure to terminate is a
    # trajectory-level process defect even when all intermediate calls worked.
    meaningful = [event for event in events if abs(event.delta) > 1e-12]
    raw_score = sum(event.delta for event in meaningful) / max(len(meaningful), 1)
    terminal_tool = str(steps[-1].get("tool_name")) if steps else ""
    if terminal_tool not in {"complete_task", "finish_without_completion"}:
        raw_score -= 0.10
        metrics["unfinished"] = 1
    else:
        metrics["unfinished"] = 0
    lo, hi = clip
    process_score = max(lo, min(hi, raw_score))
    metrics["event_count"] = len(events)
    metrics["grounded_write_events"] = grounded_write_events
    metrics["successful_terminal"] = int(terminal_reward > 0.9)
    return ProcessRewardResult(process_score, tuple(events), metrics)


__all__ = ["ProcessRewardResult", "ProcessEvent", "score_trajectory"]
