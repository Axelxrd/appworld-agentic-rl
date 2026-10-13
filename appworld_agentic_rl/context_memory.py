"""Deterministic context memory for long AppWorld rollouts.

This module intentionally does not call an LLM.  It extracts only facts that
are already present in actor-visible tool events and serializes them with a
stable template so rollout context can be reproduced during auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence


_ID_KEY = re.compile(r"(?:^|_)(?:id|uuid|token)$", re.IGNORECASE)
_AUTH_KEY = re.compile(r"(?:token|auth|session|credential|supervisor|user)$", re.IGNORECASE)
_SECRET_KEY = re.compile(
    r"(?:password|access[_ -]?token|api[_ -]?key|authorization|secret|credential|token)$",
    re.IGNORECASE,
)
_IMPORTANT_KEYS = {
    "amount",
    "count",
    "email",
    "name",
    "status",
    "state",
    "title",
    "username",
    "value",
}

_MAX_API_DISCOVERIES = 64
_MAX_API_DESCRIPTION_CHARS = 160
_MAX_REQUIRED_PARAMETERS = 12


@dataclass(slots=True)
class MemoryState:
    """Bounded, auditable facts extracted from completed tool events."""

    auth: dict[str, str] = field(default_factory=dict)
    entities: dict[str, str] = field(default_factory=dict)
    # Compact search results keyed by ``app.api``. Full schemas remain in
    # ``api_schemas`` and are only recorded after inspect_api_schema.
    api_discoveries: dict[str, dict[str, Any]] = field(default_factory=dict)
    api_schemas: dict[str, str] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    # Name/provenance metadata only; never store credential literal values here.
    variable_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    version: int = 0


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _flatten_facts(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Extract conservative scalar facts from nested actor-visible output."""

    facts: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, (str, int, float, bool)) or child is None:
                if _ID_KEY.search(key) or _AUTH_KEY.search(key) or key.lower() in _IMPORTANT_KEYS:
                    facts.append((path, str(child)))
            else:
                facts.extend(_flatten_facts(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value[:32]):
            facts.extend(_flatten_facts(child, f"{prefix}[{index}]"))
    return facts


def _observation_mapping(observation: Any) -> Mapping[str, Any] | None:
    if isinstance(observation, Mapping):
        return observation
    if isinstance(observation, str):
        try:
            parsed = json.loads(observation)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _required_parameter_names(parameters: Any) -> list[str]:
    """Extract only required argument names from a compact search result."""

    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)):
        return []
    names: list[str] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = str(parameter.get("name") or "").strip()
        if not name or name == "<MAX_DEPTH_REACHED>":
            continue
        required = parameter.get("required")
        if required is False:
            continue
        if name not in names:
            names.append(name)
        if len(names) >= _MAX_REQUIRED_PARAMETERS:
            break
    return names


def _discovery_priority(item: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Higher status and more recent observations should survive eviction."""

    status = str(item.get("status") or "discovered")
    status_rank = {"discovered": 0, "inspected": 1, "called": 2, "succeeded": 3}.get(status, 0)
    return (
        status_rank,
        int(item.get("last_seen_event") or 0),
        int(item.get("first_seen_event") or 0),
        str(item.get("key") or ""),
    )


def _record_api_discoveries(memory: MemoryState, observed: Mapping[str, Any], step_index: int) -> None:
    """Merge search results into bounded, deduplicated API discovery memory."""

    results = observed.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)):
        return
    for result in results:
        if not isinstance(result, Mapping):
            continue
        app = str(result.get("app") or "").strip()
        api = str(result.get("api") or "").strip()
        if not app or not api:
            continue
        key = f"{app}.{api}"
        description = " ".join(str(result.get("description") or "").split())
        description = description[:_MAX_API_DESCRIPTION_CHARS]
        required = _required_parameter_names(result.get("parameters"))
        existing = memory.api_discoveries.get(key)
        if existing is None:
            memory.api_discoveries[key] = {
                "key": key,
                "app": app,
                "api": api,
                "description": description,
                "required_parameters": required,
                "first_seen_event": int(step_index),
                "last_seen_event": int(step_index),
                "search_count": 1,
                "status": "inspected" if key in memory.api_schemas else "discovered",
            }
        else:
            existing["last_seen_event"] = int(step_index)
            existing["search_count"] = int(existing.get("search_count") or 0) + 1
            if description and not existing.get("description"):
                existing["description"] = description
            if required and not existing.get("required_parameters"):
                existing["required_parameters"] = required

    while len(memory.api_discoveries) > _MAX_API_DISCOVERIES:
        victim = min(memory.api_discoveries.values(), key=_discovery_priority)
        memory.api_discoveries.pop(str(victim.get("key") or ""), None)


def update_memory(memory: MemoryState, step: Mapping[str, Any]) -> MemoryState:
    """Update memory from one completed actor-visible trajectory step."""

    step_index = int(step.get("step_index", memory.version))
    tool_name = str(step.get("tool_name") or "")
    arguments = step.get("arguments")
    observation = step.get("observation")
    error = step.get("error")

    metadata = step.get("metadata")
    raw_bindings = metadata.get("variable_bindings") if isinstance(metadata, Mapping) else None
    if isinstance(raw_bindings, Sequence) and not isinstance(raw_bindings, (str, bytes, bytearray)):
        for raw_binding in raw_bindings:
            if not isinstance(raw_binding, Mapping) or not str(raw_binding.get("name") or "").strip():
                continue
            name = str(raw_binding["name"]).strip()
            # Copy a deliberately small allow-list. In particular, do not
            # accept arbitrary metadata that could contain a secret value.
            binding = {
                "name": name,
                "kind": str(raw_binding.get("kind") or "local"),
                "semantic": str(raw_binding.get("semantic") or "agent_state"),
                "app": str(raw_binding["app"]) if raw_binding.get("app") else None,
                "derived_from": str(raw_binding["derived_from"])
                if raw_binding.get("derived_from")
                else None,
                "source_event": int(raw_binding.get("source_event", step_index)),
                "available": bool(raw_binding.get("available", error is None)),
            }
            if binding["available"]:
                memory.variable_bindings[name] = binding
            else:
                # Preserve the fact that this name was attempted, while
                # making it explicit that the failed event did not define it.
                memory.variable_bindings[name] = binding

    observed = _observation_mapping(observation)
    fact_source: Any = observed if observed is not None else observation
    # Search results have their own compact representation below. Do not also
    # flatten every parameter field into entity memory.
    if tool_name == "search_api_docs" and isinstance(observed, Mapping):
        fact_source = {key: value for key, value in observed.items() if key != "results"}
    facts = _flatten_facts(fact_source)
    for key, value in facts:
        target = memory.auth if _AUTH_KEY.search(key.rsplit(".", 1)[-1]) else memory.entities
        if target is memory.auth and _SECRET_KEY.search(key.rsplit(".", 1)[-1]):
            value = "available"
        target[key] = value
        memory.sources[key] = step_index

    if tool_name == "inspect_api_schema":
        app = arguments.get("app") if isinstance(arguments, Mapping) else None
        api = arguments.get("api") if isinstance(arguments, Mapping) else None
        schema_key = ".".join(str(x) for x in (app, api) if x)
        if schema_key:
            required = []
            if isinstance(observed, Mapping):
                raw_required = observed.get("required") or observed.get("required_parameters")
                if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray)):
                    required = [str(x) for x in raw_required[:32]]
                else:
                    required = _required_parameter_names(observed.get("parameters"))
            memory.api_schemas[schema_key] = ", ".join(required) if required else "inspected"
            discovery = memory.api_discoveries.get(schema_key)
            if discovery is not None:
                discovery["status"] = "inspected"
                discovery["last_seen_event"] = int(step_index)
                if required:
                    discovery["required_parameters"] = required[:_MAX_REQUIRED_PARAMETERS]

    if tool_name == "search_api_docs" and observed is not None:
        _record_api_discoveries(memory, observed, step_index)

    if tool_name not in {"complete_task", "finish_without_completion"}:
        action = json.dumps(
            {"tool": tool_name, "arguments": _json_value(arguments)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if action not in memory.completed and not error:
            memory.completed.append(action)

    if error:
        normalized = f"{tool_name}: {str(error).strip()}"
        if normalized not in memory.errors:
            memory.errors.append(normalized)
        memory.errors = memory.errors[-4:]

    memory.version = max(memory.version + 1, step_index + 1)
    return memory


def serialize_memory(memory: MemoryState, *, max_items: int = 32) -> str:
    """Serialize memory using a deterministic, bounded text template."""

    lines = ["[APPWORLD MEMORY]"]
    lines.append("[AUTH]")
    for key, value in sorted(memory.auth.items())[:max_items]:
        display_value = "available" if _SECRET_KEY.search(key.rsplit(".", 1)[-1]) else value
        lines.append(f"- {key} = {display_value} [event {memory.sources.get(key, '?')}]")
    lines.append("[ENTITIES]")
    lines.extend(f"- {key} = {value} [event {memory.sources.get(key, '?')}]" for key, value in sorted(memory.entities.items())[:max_items])
    lines.append("[API DISCOVERIES]")
    discoveries = sorted(
        memory.api_discoveries.values(),
        key=lambda item: (-_discovery_priority(item)[0], -_discovery_priority(item)[1], str(item.get("key") or "")),
    )
    for item in discoveries[:max_items]:
        key = str(item.get("key") or "")
        description = " ".join(str(item.get("description") or "").split())
        required = item.get("required_parameters") or []
        required_text = ", ".join(str(value) for value in required[:_MAX_REQUIRED_PARAMETERS])
        suffix = f" | required: {required_text}" if required_text else ""
        status = str(item.get("status") or "discovered")
        lines.append(f"- {key} | {description} | {status}{suffix}")
    lines.append("[API SCHEMAS]")
    lines.extend(f"- {key}: {value}" for key, value in sorted(memory.api_schemas.items())[:max_items])
    lines.append("[VARIABLE BINDINGS]")
    for name, binding in sorted(memory.variable_bindings.items())[:max_items]:
        details = [
            str(binding.get("semantic") or "agent_state"),
            f"kind={binding.get('kind') or 'local'}",
        ]
        if binding.get("app"):
            details.append(f"app={binding['app']}")
        if binding.get("derived_from"):
            details.append(f"from={binding['derived_from']}")
        details.append(f"event={binding.get('source_event', '?')}")
        details.append("available" if binding.get("available") else "unavailable")
        lines.append(f"- {name}: {', '.join(details)}")
    lines.append("[COMPLETED]")
    lines.extend(f"- {item}" for item in memory.completed[-max_items:])
    lines.append("[ERRORS]")
    lines.extend(f"- {item}" for item in memory.errors[-4:])
    lines.append("[PENDING]")
    lines.extend(f"- {item}" for item in memory.pending[-max_items:])
    return "\n".join(lines)


def context_snapshot(
    *,
    turn: int,
    context_tokens_before: int,
    context_token_limit: int,
    tokens_removed: int,
    memory: MemoryState,
    recent_groups: int,
) -> dict[str, Any]:
    """Return a stable instrumentation record for one generation boundary."""

    return {
        "turn": int(turn),
        "context_tokens_before": int(context_tokens_before),
        "context_token_limit": int(context_token_limit),
        "context_tokens_after": max(0, int(context_tokens_before) - int(tokens_removed)),
        "truncated": bool(tokens_removed),
        "tokens_removed": int(tokens_removed),
        "preserved": {
            "system_prompt": True,
            "task_instruction": True,
            "authentication": bool(memory.auth),
            "discovered_apis": sorted(memory.api_discoveries)[:32],
            "entity_ids": sorted(key for key in memory.entities if _ID_KEY.search(key.rsplit(".", 1)[-1]))[:32],
            "variable_bindings": sorted(
                name
                for name, binding in memory.variable_bindings.items()
                if binding.get("available")
            )[:64],
            "recent_groups": int(recent_groups),
            "memory_version": int(memory.version),
        },
    }


__all__ = ["MemoryState", "context_snapshot", "serialize_memory", "update_memory"]
