"""Split-safe persistence and accounting for teacher rollout collection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any

from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS
from appworld_agentic_rl.reward import infer_evaluation_success
from appworld_agentic_rl.teacher_client import TeacherModelClient, TeacherQuotaError
from appworld_agentic_rl.teacher_curation import (
    curate_teacher_rollouts,
    enrich_rollout_fingerprints,
)
from appworld_agentic_rl.harness import HARNESS_VERSION, TeacherHarnessConfig, stable_hash
from appworld_agentic_rl.teacher_rollout import (
    SYSTEM_PROMPT,
    TEACHER_ROLLOUT_SCHEMA_VERSION,
    redact_sensitive_data,
    run_teacher_rollout,
)
from appworld_agentic_rl.trajectory import utc_now_iso


COLLECTION_SCHEMA_VERSION = "appworld-teacher-collection-v3"
TRAINING_COLLECTION_SPLITS = {"train", "appworld_rl_v1_teacher_sft"}


def _default_task_id_loader(split: str) -> list[str]:
    from appworld import load_task_ids

    return list(load_task_ids(split))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def _prepare_output_directory(output_dir: str | Path, *, resume: bool) -> tuple[Path, bool]:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"teacher rollout output is not a directory: {destination}")
        if not resume:
            raise FileExistsError(f"teacher rollout output already exists: {destination}")
        return destination, True
    destination.mkdir(parents=True)
    return destination, False


def _config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in config.items() if key != "created_at"}


def _validate_resume_config(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"cannot resume without run_config.json: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, Mapping):
        raise ValueError(f"invalid resume config: {path}")
    if _config_contract(existing) != _config_contract(expected):
        raise ValueError(
            "resume configuration does not match the existing teacher collection"
        )


def _completed_attempts(
    records: Iterable[Mapping[str, Any]],
    *,
    task_ids: set[str],
    attempts_per_task: int,
    allowed_attempts: set[tuple[str, int]] | None = None,
) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for record in records:
        task_id = str(record.get("task_id") or "")
        attempt_index = int(record.get("attempt_index") or 0)
        if task_id not in task_ids:
            raise ValueError(f"raw resume data contains an unexpected task: {task_id}")
        if not 1 <= attempt_index <= attempts_per_task:
            raise ValueError(
                f"raw resume data has invalid attempt index {attempt_index} for {task_id}"
            )
        key = (task_id, attempt_index)
        if allowed_attempts is not None and key not in allowed_attempts:
            raise ValueError(
                f"raw resume data contains an unrequested attempt: {task_id} #{attempt_index}"
            )
        if key in completed:
            raise ValueError(f"duplicate raw attempt in resume data: {task_id} #{attempt_index}")
        completed.add(key)
    return completed


def _resolve_task_ids(
    *,
    task_ids: Iterable[str] | None,
    limit: int | None,
    split: str,
    task_id_loader: Callable[[str], list[str]],
) -> list[str]:
    available = task_id_loader(split)
    available_set = set(available)
    requested = list(task_ids) if task_ids else available
    if not requested:
        raise ValueError(f"no task IDs are available in training dataset {split!r}")
    duplicates = sorted({task_id for task_id in requested if requested.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate task IDs are not allowed: {duplicates}")
    unknown = [task_id for task_id in requested if task_id not in available_set]
    if unknown:
        raise ValueError(f"task IDs are not in training dataset {split!r}: {unknown}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        requested = requested[:limit]
    return requested


def _collection_failure_record(
    task_id: str, *, split: str, error: Exception
) -> dict[str, Any]:
    error_text = type(error).__name__ + ": " + str(error)
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
            "errors": [error_text],
        },
        "conversation": [],
        "trajectory": None,
        "evaluation": {"error": error_text},
        "terminal_reward": None,
        "infrastructure_invalid": True,
    }


def _summary(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    num_tasks: int,
    collection_stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = sum(record["collection_status"] == "accepted" for record in records)
    clean_accepted = sum(
        record["collection_status"] == "accepted"
        and isinstance(record.get("process_metrics"), Mapping)
        and int(record["process_metrics"].get("execution_errors", 0)) == 0
        for record in records
    )
    curated = sum(bool(record.get("selected_for_sft")) for record in records)
    duplicates = sum(record.get("duplicate_of") is not None for record in records)
    step_counts: list[int] = []
    terminal_rewards: list[float] = []
    execution_errors = 0
    infrastructure_invalid = 0
    model_turn_counts: list[int] = []
    action_attempt_counts: list[int] = []
    environment_step_counts: list[int] = []
    end_reasons: Counter[str] = Counter()
    for record in records:
        rollout = record.get("rollout")
        if isinstance(rollout, Mapping):
            end_reasons[str(rollout.get("end_reason", "unknown"))] += 1
            model_turn_counts.append(int(rollout.get("model_turns") or 0))
            action_attempt_counts.append(int(rollout.get("action_attempts") or 0))
            environment_step_counts.append(int(rollout.get("environment_steps") or 0))
        infrastructure_invalid += bool(record.get("infrastructure_invalid"))
        trajectory = record.get("trajectory")
        if isinstance(trajectory, Mapping):
            steps = trajectory.get("steps")
            if isinstance(steps, list):
                step_counts.append(len(steps))
                execution_errors += sum(
                    1 for step in steps if isinstance(step, Mapping) and step.get("error")
                )
        reward = record.get("terminal_reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            terminal_rewards.append(float(reward))
    total = len(records)
    successful_tasks = len(
        {
            str(record["task_id"])
            for record in records
            if record["collection_status"] == "accepted"
        }
    )
    clean_successful_tasks = len(
        {
            str(record["task_id"])
            for record in records
            if record["collection_status"] == "accepted"
            and isinstance(record.get("process_metrics"), Mapping)
            and int(record["process_metrics"].get("execution_errors", 0)) == 0
        }
    )
    curated_execution_errors = sum(
        int(record["process_metrics"].get("execution_errors", 0))
        for record in records
        if bool(record.get("selected_for_sft"))
        and isinstance(record.get("process_metrics"), Mapping)
    )
    summary = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "num_tasks": num_tasks,
        "num_rollouts": total,
        "accepted": accepted,
        "clean_accepted": clean_accepted,
        "rejected": total - accepted,
        "curated_sft": curated,
        "duplicates": duplicates,
        "curation_cap_excluded": sum(
            record.get("curation_exclusion_reason") == "max_curated_per_task"
            for record in records
        ),
        "execution_error_excluded": sum(
            record.get("curation_exclusion_reason") == "execution_errors"
            for record in records
        ),
        "success_rate": accepted / total if total else 0.0,
        "task_success_rate": successful_tasks / num_tasks if num_tasks else 0.0,
        "clean_task_success_rate": (
            clean_successful_tasks / num_tasks if num_tasks else 0.0
        ),
        "average_step_count": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "average_terminal_reward": (
            sum(terminal_rewards) / len(terminal_rewards) if terminal_rewards else None
        ),
        "execution_error_count": execution_errors,
        "infrastructure_invalid_count": infrastructure_invalid,
        "average_model_turns": (
            sum(model_turn_counts) / len(model_turn_counts) if model_turn_counts else 0.0
        ),
        "average_action_attempts": (
            sum(action_attempt_counts) / len(action_attempt_counts)
            if action_attempt_counts
            else 0.0
        ),
        "average_environment_steps": (
            sum(environment_step_counts) / len(environment_step_counts)
            if environment_step_counts
            else 0.0
        ),
        "curated_execution_error_count": curated_execution_errors,
        "end_reason_counts": dict(sorted(end_reasons.items())),
    }
    if collection_stop is not None:
        summary["stopped_early"] = True
        summary["stop_reason"] = str(collection_stop["reason"])
        summary["pending_task_id"] = str(collection_stop["task_id"])
        summary["pending_attempt_index"] = int(collection_stop["attempt_index"])
        summary["stop_error"] = str(collection_stop["error"])
    else:
        summary["stopped_early"] = False
        summary["stop_reason"] = None
    return summary


def collect_teacher_rollouts(
    *,
    client: TeacherModelClient,
    adapter_factory: Callable[[], Any],
    output_dir: str | Path,
    model_name: str,
    task_ids: Iterable[str] | None = None,
    limit: int | None = None,
    split: str = "train",
    max_model_turns: int = 25,
    attempts_per_task: int = 1,
    max_curated_per_task: int = 2,
    require_clean_for_sft: bool = True,
    harness_config: TeacherHarnessConfig | None = None,
    resume: bool = False,
    run_config: Mapping[str, Any] | None = None,
    task_id_loader: Callable[[str], list[str]] | None = None,
    attempt_slots: Iterable[tuple[str, int]] | None = None,
    record_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect training rollouts, verifier-filter them, and curate diverse successes."""

    if split not in TRAINING_COLLECTION_SPLITS:
        raise ValueError(
            "teacher collection is restricted to the train split or the frozen teacher-SFT pool"
        )
    if attempts_per_task < 1:
        raise ValueError("attempts_per_task must be positive")
    if max_curated_per_task < 1:
        raise ValueError("max_curated_per_task must be positive")
    harness = harness_config or TeacherHarnessConfig()
    loader = task_id_loader or _default_task_id_loader
    resolved_task_ids = _resolve_task_ids(
        task_ids=task_ids,
        limit=limit,
        split=split,
        task_id_loader=loader,
    )
    selected_slots: set[tuple[str, int]] | None = None
    if attempt_slots is not None:
        slot_list = [(str(task_id), int(attempt_index)) for task_id, attempt_index in attempt_slots]
        selected_slots = set(slot_list)
        if len(selected_slots) != len(slot_list):
            raise ValueError("duplicate exact attempt slots are not allowed")
        invalid_slots = sorted(
            (task_id, attempt_index)
            for task_id, attempt_index in selected_slots
            if task_id not in set(resolved_task_ids)
            or not 1 <= attempt_index <= attempts_per_task
        )
        if invalid_slots:
            raise ValueError(f"invalid exact attempt slots: {invalid_slots}")
        if not selected_slots:
            raise ValueError("at least one exact attempt slot is required")
    destination, is_resume = _prepare_output_directory(output_dir, resume=resume)
    data_manifest_hash = stable_hash({"split": split, "task_ids": resolved_task_ids})
    system_prompt_hash = stable_hash(SYSTEM_PROMPT)
    tool_schema_hash = stable_hash(AGENT_TOOL_SCHEMAS)
    config = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "split": split,
        "task_ids": resolved_task_ids,
        "teacher": {
            "model": model_name,
            "base_url": getattr(client, "base_url", None),
            "user_agent": getattr(client, "user_agent", None),
            "disable_provider_parallel_tool_calls": getattr(
                client, "disable_provider_parallel_tool_calls", None
            ),
            "thinking": {
                "mode": "provider_default",
                "reasoning_content_history": True,
            },
        },
        "harness": {
            "version": HARNESS_VERSION,
            "system_prompt_hash": system_prompt_hash,
            "tool_schema_hash": tool_schema_hash,
            "data_manifest_hash": data_manifest_hash,
            "guard": {
                "mode": harness.bc_guard_mode,
                "invalid_action_limit": harness.invalid_action_limit,
                "max_identical_action_repeats": harness.max_identical_action_repeats,
            },
            "context": {
                "context_budget_mode": harness.context_budget_mode,
                "max_context_chars": harness.max_context_chars,
                "max_observation_chars": harness.max_observation_chars,
                "context_compaction_enable": harness.context_compaction_enable,
                "context_compaction_trigger_chars": harness.context_compaction_trigger_chars,
                "context_compaction_target_chars": harness.context_compaction_target_chars,
                "max_context_tokens": harness.max_context_tokens,
                "context_compaction_trigger_tokens": harness.context_compaction_trigger_tokens,
                "context_compaction_target_tokens": harness.context_compaction_target_tokens,
                "context_preserve_recent_groups": harness.context_preserve_recent_groups,
            },
            "max_action_attempts": harness.max_action_attempts,
        },
        "max_model_turns": max_model_turns,
        "attempts_per_task": attempts_per_task,
        "max_curated_per_task": max_curated_per_task,
        "require_clean_for_sft": require_clean_for_sft,
    }
    if selected_slots is not None:
        config["exact_attempt_slots"] = [
            {
                "task_id": task_id,
                "attempt_index": attempt_index,
                "rollout_id": f"{task_id}:attempt-{attempt_index:03d}",
            }
            for task_id, attempt_index in sorted(
                selected_slots,
                key=lambda slot: (resolved_task_ids.index(slot[0]), slot[1]),
            )
        ]
    if record_provenance:
        config["provenance"] = redact_sensitive_data(dict(record_provenance))
    if run_config:
        config["extra"] = redact_sensitive_data(dict(run_config))
    config["run_contract_hash"] = stable_hash(_config_contract(config))
    config_path = destination / "run_config.json"
    if is_resume:
        _validate_resume_config(config_path, config)
    else:
        _write_json(config_path, config)

    raw_path = destination / "raw.jsonl"
    accepted_path = destination / "accepted.jsonl"
    rejected_path = destination / "rejected.jsonl"
    duplicates_path = destination / "duplicates.jsonl"
    curated_path = destination / "curated_sft.jsonl"
    if not is_resume:
        for path in (raw_path, accepted_path, rejected_path, duplicates_path, curated_path):
            path.write_bytes(b"")
    records = _load_jsonl(raw_path)
    completed = _completed_attempts(
        records,
        task_ids=set(resolved_task_ids),
        attempts_per_task=attempts_per_task,
        allowed_attempts=selected_slots,
    )
    collection_stop: dict[str, Any] | None = None
    for task_id in resolved_task_ids:
        for attempt_index in range(1, attempts_per_task + 1):
            if selected_slots is not None and (task_id, attempt_index) not in selected_slots:
                continue
            if (task_id, attempt_index) in completed:
                continue
            adapter: Any | None = None
            release_error: Exception | None = None
            quota_error: TeacherQuotaError | None = None
            try:
                adapter = adapter_factory()
                record = run_teacher_rollout(
                    adapter=adapter,
                    client=client,
                    task_id=task_id,
                    split=split,
                    max_model_turns=max_model_turns,
                    harness_config=harness,
                )
            except TeacherQuotaError as exc:
                quota_error = exc
            except Exception as exc:
                record = _collection_failure_record(task_id, split=split, error=exc)
            finally:
                if adapter is not None:
                    try:
                        adapter.close()
                    except Exception as exc:
                        release_error = exc

            if quota_error is not None:
                collection_stop = {
                    "reason": "provider_quota_exhausted",
                    "task_id": task_id,
                    "attempt_index": attempt_index,
                    "error": redact_sensitive_data(
                        type(quota_error).__name__ + ": " + str(quota_error)
                    ),
                }
                break

            if release_error is not None:
                error_text = type(release_error).__name__ + ": " + str(release_error)
                rollout = record.setdefault("rollout", {})
                if isinstance(rollout, dict):
                    rollout["end_reason"] = "release_error"
                    rollout.setdefault("errors", []).append("release_error: " + error_text)
                record["infrastructure_invalid"] = True

            record = redact_sensitive_data(record)
            success = (
                infer_evaluation_success(record.get("evaluation") or {}) is True
                and not bool(record.get("infrastructure_invalid"))
                and isinstance(record.get("rollout"), Mapping)
                and record["rollout"].get("end_reason") == "evaluator_success"
            )
            record["collection_status"] = "accepted" if success else "rejected"
            record["rollout_id"] = f"{task_id}:attempt-{attempt_index:03d}"
            record["attempt_index"] = attempt_index
            record["teacher_model"] = model_name
            record["data_manifest_hash"] = data_manifest_hash
            record["run_contract_hash"] = config["run_contract_hash"]
            if record_provenance:
                record["provenance"] = redact_sensitive_data(dict(record_provenance))
            record = enrich_rollout_fingerprints(record)
            _append_jsonl(raw_path, record)
            records.append(record)
            completed.add((task_id, attempt_index))
        if collection_stop is not None:
            break

    task_order = {task_id: index for index, task_id in enumerate(resolved_task_ids)}
    records.sort(
        key=lambda record: (
            task_order[str(record["task_id"])],
            int(record["attempt_index"]),
        )
    )
    records = curate_teacher_rollouts(
        records,
        max_curated_per_task=max_curated_per_task,
        require_clean_for_sft=require_clean_for_sft,
    )
    _write_jsonl(raw_path, records)
    _write_jsonl(
        accepted_path,
        (record for record in records if record["collection_status"] == "accepted"),
    )
    _write_jsonl(
        rejected_path,
        (record for record in records if record["collection_status"] == "rejected"),
    )
    _write_jsonl(
        duplicates_path,
        (record for record in records if record.get("duplicate_of") is not None),
    )
    _write_jsonl(
        curated_path,
        (record for record in records if bool(record.get("selected_for_sft"))),
    )

    summary = _summary(
        records,
        destination,
        num_tasks=len(resolved_task_ids),
        collection_stop=collection_stop,
    )
    _write_json(destination / "summary.json", summary)
    return summary
