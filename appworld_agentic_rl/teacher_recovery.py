"""Exact-slot recovery for teacher rollouts that ended in provider errors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from appworld_agentic_rl.teacher_client import TeacherModelClient
from appworld_agentic_rl.teacher_collector import collect_teacher_rollouts
from appworld_agentic_rl.harness import TeacherHarnessConfig
from appworld_agentic_rl.context_policy import DEFAULT_CONTEXT_POLICY


RECOVERY_SCHEMA_VERSION = "appworld-teacher-provider-recovery-v1"


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_raw(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_bytes = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return raw_bytes, rows


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"source run_config.json has no valid {key!r} object")
    return value


def teacher_recovery_settings(source_dir: str | Path) -> dict[str, Any]:
    """Load provider sampling settings from the immutable source run contract."""

    return _teacher_recovery_settings(Path(source_dir), seen=set())


def _teacher_recovery_settings(source: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved_source = source.resolve()
    if resolved_source in seen:
        raise ValueError("recovery provenance contains a source cycle")
    seen.add(resolved_source)

    config = _load_object(resolved_source / "run_config.json")
    teacher = _mapping(config, "teacher")
    raw_extra = config.get("extra")
    if raw_extra is None:
        extra: Mapping[str, Any] = {}
    elif isinstance(raw_extra, Mapping):
        extra = raw_extra
    else:
        raise ValueError("source run_config.json has no valid 'extra' object")
    model = str(teacher.get("model") or "")
    base_url = str(teacher.get("base_url") or "")
    if not model or not base_url:
        raise ValueError("source run_config.json has no valid teacher model/base_url")
    settings = {
        "model": model,
        "base_url": base_url,
        "temperature": float(extra.get("temperature", 0.0)),
        "max_tokens": int(extra.get("max_tokens", 1024)),
        "timeout": float(extra.get("timeout", 60.0)),
        "max_retries": int(extra.get("request_retries", 3)),
        "retry_backoff_seconds": float(extra.get("retry_backoff_seconds", 1.0)),
        "disable_provider_parallel_tool_calls": bool(
            teacher.get("disable_provider_parallel_tool_calls")
        ),
    }
    sampling_keys = {
        "temperature",
        "max_tokens",
        "timeout",
        "request_retries",
        "retry_backoff_seconds",
    }
    if config.get("exact_attempt_slots") is not None and not sampling_keys <= set(extra):
        provenance = _mapping(config, "provenance")
        parent_text = str(provenance.get("source_dir") or "")
        if not parent_text:
            raise ValueError("exact-slot recovery source has no parent source_dir")
        inherited = _teacher_recovery_settings(Path(parent_text), seen=seen)
        if inherited["model"] != model or inherited["base_url"] != base_url:
            raise ValueError("recovery source provider does not match its parent")
        inherited_parallel = inherited["disable_provider_parallel_tool_calls"]
        if inherited_parallel != settings["disable_provider_parallel_tool_calls"]:
            raise ValueError("recovery source parallel-tool setting does not match its parent")
        settings = inherited
    return settings


def _configured_attempt_slots(
    config: Mapping[str, Any],
    *,
    task_ids: list[str],
    attempts_per_task: int,
) -> set[tuple[str, int]]:
    raw_slots = config.get("exact_attempt_slots")
    if raw_slots is None:
        return {
            (task_id, attempt_index)
            for task_id in task_ids
            for attempt_index in range(1, attempts_per_task + 1)
        }
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("source run_config.json has invalid exact_attempt_slots")

    allowed_tasks = set(task_ids)
    slots: set[tuple[str, int]] = set()
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, Mapping):
            raise ValueError("source run_config.json has an invalid exact attempt slot")
        task_id = str(raw_slot.get("task_id") or "")
        attempt_index = int(raw_slot.get("attempt_index") or 0)
        slot = (task_id, attempt_index)
        if task_id not in allowed_tasks or not 1 <= attempt_index <= attempts_per_task:
            raise ValueError(f"source run_config.json has an invalid exact attempt slot: {slot}")
        if slot in slots:
            raise ValueError(f"source run_config.json has a duplicate exact attempt slot: {slot}")
        slots.add(slot)
    return slots


def _provider_error_slots(
    rows: list[dict[str, Any]],
    *,
    task_ids: list[str],
    attempts_per_task: int,
) -> list[tuple[str, int]]:
    task_order = {task_id: index for index, task_id in enumerate(task_ids)}
    seen: set[tuple[str, int]] = set()
    slots: list[tuple[str, int]] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        attempt_index = int(row.get("attempt_index") or 0)
        slot = (task_id, attempt_index)
        if task_id not in task_order or not 1 <= attempt_index <= attempts_per_task:
            raise ValueError(f"source raw.jsonl contains an invalid slot: {slot}")
        if slot in seen:
            raise ValueError(f"source raw.jsonl contains a duplicate slot: {slot}")
        seen.add(slot)
        expected_rollout_id = f"{task_id}:attempt-{attempt_index:03d}"
        if row.get("rollout_id") != expected_rollout_id:
            raise ValueError(
                f"source slot {slot} has unexpected rollout_id {row.get('rollout_id')!r}"
            )
        rollout = row.get("rollout")
        if isinstance(rollout, Mapping) and rollout.get("end_reason") == "provider_error":
            slots.append(slot)
    return sorted(slots, key=lambda slot: (task_order[slot[0]], slot[1]))


def recover_teacher_provider_errors(
    *,
    client: TeacherModelClient,
    adapter_factory: Callable[[], Any],
    source_dir: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Recollect only provider-error slots from a complete immutable source run."""

    source = Path(source_dir)
    config = _load_object(source / "run_config.json")
    summary = _load_object(source / "summary.json")
    raw_path = source / "raw.jsonl"
    source_bytes, rows = _load_raw(raw_path)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    raw_task_ids = config.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(
        isinstance(task_id, str) and task_id for task_id in raw_task_ids
    ):
        raise ValueError("source run_config.json has no valid task_ids list")
    task_ids = list(raw_task_ids)
    attempts_per_task = int(config.get("attempts_per_task") or 0)
    if attempts_per_task < 1:
        raise ValueError("source attempts_per_task must be positive")
    if summary.get("stopped_early") is True:
        raise ValueError("source collection stopped early; resume it before provider recovery")

    expected_slots = _configured_attempt_slots(
        config,
        task_ids=task_ids,
        attempts_per_task=attempts_per_task,
    )
    slots = _provider_error_slots(
        rows,
        task_ids=task_ids,
        attempts_per_task=attempts_per_task,
    )
    observed_slots = {
        (str(row.get("task_id") or ""), int(row.get("attempt_index") or 0)) for row in rows
    }
    expected_rollouts = len(expected_slots)
    if (
        observed_slots != expected_slots
        or len(rows) != expected_rollouts
        or int(summary.get("num_rollouts") or 0) != expected_rollouts
    ):
        raise ValueError(
            "source collection is incomplete; finish or resume it before freezing recovery slots"
        )
    if not slots:
        raise ValueError("source collection contains no provider_error slots")

    teacher = _mapping(config, "teacher")
    harness = _mapping(config, "harness")
    guard = _mapping(harness, "guard")
    context = _mapping(harness, "context")
    source_model = str(teacher.get("model") or "")
    client_model = getattr(client, "model", None)
    if client_model is not None and str(client_model) != source_model:
        raise ValueError(
            f"recovery model {client_model!r} does not match source model {source_model!r}"
        )
    source_parallel_disabled = bool(teacher.get("disable_provider_parallel_tool_calls"))
    client_parallel_disabled = getattr(client, "disable_provider_parallel_tool_calls", None)
    if (
        client_parallel_disabled is not None
        and bool(client_parallel_disabled) != source_parallel_disabled
    ):
        raise ValueError("recovery parallel-tool setting does not match the source run")

    affected = {task_id for task_id, _ in slots}
    affected_task_ids = [task_id for task_id in task_ids if task_id in affected]
    sampling_settings = {
        "temperature": float(getattr(client, "temperature", 0.0)),
        "max_tokens": int(getattr(client, "max_tokens", 1024)),
        "timeout": float(getattr(client, "timeout", 60.0)),
        "max_retries": int(getattr(client, "max_retries", 3)),
        "retry_backoff_seconds": float(
            getattr(client, "retry_backoff_seconds", 1.0)
        ),
    }
    provenance = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "source_dir": str(source.resolve()),
        "source_raw_sha256": source_sha256,
        "source_run_contract_hash": config.get("run_contract_hash"),
        "source_data_manifest_hash": harness.get("data_manifest_hash"),
        "selection": "rollout.end_reason == provider_error",
    }
    recovery_summary = collect_teacher_rollouts(
        client=client,
        adapter_factory=adapter_factory,
        output_dir=output_dir,
        model_name=source_model,
        task_ids=affected_task_ids,
        split=str(config.get("split") or ""),
        max_model_turns=int(config.get("max_model_turns") or 0),
        attempts_per_task=attempts_per_task,
        max_curated_per_task=int(config.get("max_curated_per_task") or 2),
        require_clean_for_sft=bool(config.get("require_clean_for_sft", True)),
        harness_config=TeacherHarnessConfig(
            bc_guard_mode=str(guard.get("mode") or "enforce"),
            max_action_attempts=int(harness.get("max_action_attempts") or 80),
            invalid_action_limit=int(guard.get("invalid_action_limit") or 6),
            max_identical_action_repeats=int(
                guard.get("max_identical_action_repeats") or 1
            ),
            context_budget_mode=str(context.get("context_budget_mode") or "chars"),
            max_context_chars=int(
                context.get("max_context_chars") or DEFAULT_CONTEXT_POLICY.trigger_chars
            ),
            max_observation_chars=int(context.get("max_observation_chars") or 12000),
            context_compaction_enable=bool(
                context.get("context_compaction_enable", True)
            ),
            context_compaction_trigger_chars=int(
                context.get("context_compaction_trigger_chars")
                or DEFAULT_CONTEXT_POLICY.trigger_chars
            ),
            context_compaction_target_chars=int(
                context.get("context_compaction_target_chars")
                or DEFAULT_CONTEXT_POLICY.target_chars
            ),
            max_context_tokens=int(context.get("max_context_tokens") or 155904),
            context_compaction_trigger_tokens=int(
                context.get("context_compaction_trigger_tokens")
                or DEFAULT_CONTEXT_POLICY.trigger_tokens
            ),
            context_compaction_target_tokens=int(
                context.get("context_compaction_target_tokens")
                or DEFAULT_CONTEXT_POLICY.target_tokens
            ),
            context_preserve_recent_groups=int(
                context.get("context_preserve_recent_groups")
                or DEFAULT_CONTEXT_POLICY.preserve_recent_groups
            ),
        ),
        resume=resume,
        run_config={
            "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
            "source_raw_sha256": source_sha256,
            "temperature": sampling_settings["temperature"],
            "max_tokens": sampling_settings["max_tokens"],
            "timeout": sampling_settings["timeout"],
            "request_retries": sampling_settings["max_retries"],
            "retry_backoff_seconds": sampling_settings["retry_backoff_seconds"],
        },
        task_id_loader=lambda split: task_ids,
        attempt_slots=slots,
        record_provenance=provenance,
    )
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("source raw.jsonl changed during provider recovery")
    recovery_summary["source_raw_sha256"] = source_sha256
    recovery_summary["provider_error_slots"] = len(slots)
    return recovery_summary
