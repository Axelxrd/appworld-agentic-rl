"""Frozen AppWorld policy evaluation with resumable, auditable persistence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS
from appworld_agentic_rl.reward import infer_evaluation_success
from appworld_agentic_rl.teacher_client import TeacherModelClient, TeacherQuotaError
from appworld_agentic_rl.teacher_curation import enrich_rollout_fingerprints
from appworld_agentic_rl.harness import HARNESS_VERSION, TeacherHarnessConfig, stable_hash
from appworld_agentic_rl.teacher_rollout import (
    POLICY_SYSTEM_PROMPT,
    TEACHER_ROLLOUT_SCHEMA_VERSION,
    redact_sensitive_data,
    run_teacher_rollout,
)
from appworld_agentic_rl.trajectory import utc_now_iso


FROZEN_EVALUATION_SCHEMA_VERSION = "appworld-frozen-evaluation-v2-token-context"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
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
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _load_manifest(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    rows = _load_jsonl(path)
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError("evaluation manifest has an empty or missing task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("evaluation manifest contains duplicate task IDs")
    return raw, rows


def _config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in config.items() if key != "created_at"}


def _failure_record(task_id: str, *, split: str, error: Exception) -> dict[str, Any]:
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


def _validate_attempt_log(
    rows: list[dict[str, Any]], *, task_ids: set[str]
) -> tuple[dict[str, int], set[str]]:
    attempts: dict[str, int] = defaultdict(int)
    completed: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id not in task_ids:
            raise ValueError(f"raw evaluation data contains an unexpected task: {task_id}")
        attempts[task_id] += 1
        if int(row.get("evaluation_attempt_index") or 0) != attempts[task_id]:
            raise ValueError(f"raw evaluation attempts are not contiguous for {task_id}")
        if not bool(row.get("infrastructure_invalid")):
            if task_id in completed:
                raise ValueError(f"raw evaluation data has duplicate completed task: {task_id}")
            completed.add(task_id)
    return dict(attempts), completed


def _canonical_results(
    rows: list[dict[str, Any]], *, task_order: Mapping[str, int]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for task_id, attempts in grouped.items():
        valid = [row for row in attempts if not bool(row.get("infrastructure_invalid"))]
        selected.append(valid[-1] if valid else attempts[-1])
    return sorted(selected, key=lambda row: task_order[str(row["task_id"])])


def _has_collateral_damage(evaluation: Mapping[str, Any]) -> bool:
    """Accept tracker-level and nested evaluator state-integrity flags."""

    if any(
        bool(evaluation.get(key))
        for key in ("collateral_damage", "has_collateral_damage", "state_integrity_failed")
    ):
        return True
    data = evaluation.get("data")
    if isinstance(data, Mapping):
        return any(
            bool(data.get(key))
            for key in ("collateral_damage", "has_collateral_damage", "state_integrity_failed")
        )
    return False


def _contains_name_error(value: Any) -> bool:
    if isinstance(value, str):
        return "NameError" in value
    if isinstance(value, Mapping):
        return any(_contains_name_error(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_name_error(item) for item in value)
    return False


def _post_compaction_name_error_count(row: Mapping[str, Any]) -> int:
    """Count action observations containing NameError after the first compaction."""

    harness = row.get("harness")
    events = harness.get("context_events") if isinstance(harness, Mapping) else None
    compaction_turns: list[int] = []
    for event in events or []:
        if not isinstance(event, Mapping) or not bool(
            event.get("compacted") or event.get("truncated")
        ):
            continue
        raw_turn = event.get("turn_index", event.get("turn"))
        try:
            compaction_turns.append(int(raw_turn))
        except (TypeError, ValueError):
            continue
    if not compaction_turns:
        return 0
    first_compaction_turn = min(compaction_turns)

    observations = row.get("observations")
    if isinstance(observations, list):
        count = 0
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            try:
                turn = int(observation.get("turn_index", -1))
            except (TypeError, ValueError):
                continue
            if turn >= first_compaction_turn and _contains_name_error(
                observation.get("raw", observation.get("visible", observation))
            ):
                count += 1
        return count

    # Compatibility fallback for records that predate per-turn observations.
    trajectory = row.get("trajectory")
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    count = 0
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        metadata = step.get("metadata")
        raw_turn = (
            metadata.get("model_turn_index")
            if isinstance(metadata, Mapping) and "model_turn_index" in metadata
            else step.get("step_index", -1)
        )
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError):
            continue
        if turn >= first_compaction_turn and _contains_name_error(
            {"error": step.get("error"), "observation": step.get("observation")}
        ):
            count += 1
    return count


def _summary(
    *,
    attempt_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    output_dir: Path,
    collection_stop: Mapping[str, Any] | None,
) -> dict[str, Any]:
    completed = [row for row in results if not bool(row.get("infrastructure_invalid"))]

    def _execution_errors(row: Mapping[str, Any]) -> int:
        metrics = row.get("process_metrics")
        if isinstance(metrics, Mapping) and "execution_errors" in metrics:
            try:
                return max(0, int(metrics.get("execution_errors") or 0))
            except (TypeError, ValueError):
                pass
        trajectory = row.get("trajectory")
        steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
        return sum(
            bool(step.get("error"))
            for step in (steps or [])
            if isinstance(step, Mapping)
        )

    successes = [
        row for row in completed if infer_evaluation_success(row.get("evaluation") or {}) is True
    ]
    clean_successes = [
        row
        for row in successes
        if _execution_errors(row) == 0
    ]
    scenario_by_task = {
        str(row["task_id"]): str(row.get("scenario_id") or str(row["task_id"]).split("_")[0])
        for row in manifest_rows
    }
    scenario_tasks: dict[str, set[str]] = defaultdict(set)
    successful_task_ids = {str(row["task_id"]) for row in successes}
    for task_id, scenario_id in scenario_by_task.items():
        scenario_tasks[scenario_id].add(task_id)
    completed_scenarios = [
        scenario_id
        for scenario_id, task_ids in scenario_tasks.items()
        if task_ids <= {str(row["task_id"]) for row in completed}
    ]
    successful_scenarios = [
        scenario_id
        for scenario_id in completed_scenarios
        if scenario_tasks[scenario_id] <= successful_task_ids
    ]
    expected_scenarios = len(scenario_tasks)
    passed_requirements = 0
    total_requirements = 0
    for row in completed:
        data = (row.get("evaluation") or {}).get("data")
        if isinstance(data, Mapping):
            passes = data.get("passes")
            failures = data.get("failures")
            passed_requirements += len(passes) if isinstance(passes, list) else 0
            if isinstance(data.get("num_tests"), int):
                total_requirements += int(data["num_tests"])
            else:
                total_requirements += (
                    (len(passes) if isinstance(passes, list) else 0)
                    + (len(failures) if isinstance(failures, list) else 0)
                )
    end_reasons = Counter(
        str((row.get("rollout") or {}).get("end_reason") or "unknown") for row in results
    )
    infrastructure_errors = [
        row for row in results if bool(row.get("infrastructure_invalid"))
    ]
    execution_error_counts = [_execution_errors(row) for row in completed]
    collateral_damage_tasks = [
        row
        for row in completed
        if _has_collateral_damage(row.get("evaluation") or {})
    ]
    post_compaction_name_errors = [
        _post_compaction_name_error_count(row) for row in completed
    ]

    def _rollout_value(row: Mapping[str, Any], key: str) -> float:
        value = (row.get("rollout") or {}).get(key, 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _process_value(row: Mapping[str, Any], key: str) -> float:
        value = (row.get("process_metrics") or {}).get(key, 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    completed_count = len(completed)
    expected_count = len(manifest_rows)
    successful_count = len(successes)
    clean_successful_count = len(clean_successes)
    execution_error_total = sum(execution_error_counts)
    rollout_turns = [_rollout_value(row, "model_turns") for row in completed]
    action_attempts = [_rollout_value(row, "action_attempts") for row in completed]
    environment_steps = [_rollout_value(row, "environment_steps") for row in completed]
    process_steps = [_process_value(row, "step_count") for row in completed]
    compaction_counts = [
        float(
            sum(
                1
                for event in ((row.get("harness") or {}).get("context_events") or [])
                if isinstance(event, Mapping) and bool(event.get("compacted"))
            )
        )
        for row in completed
    ]
    payload: dict[str, Any] = {
        "schema_version": FROZEN_EVALUATION_SCHEMA_VERSION,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "expected_tasks": len(manifest_rows),
        "attempt_records": len(attempt_rows),
        "completed_tasks": len(completed),
        "infrastructure_pending_tasks": len(results) - len(completed),
        "successful_tasks": len(successes),
        "clean_successful_tasks": len(clean_successes),
        # Rates over completed tasks preserve the historical meaning.  The
        # expected-denominator variants expose infrastructure failures instead
        # of silently making a partially collected run look better.
        "task_success_rate": successful_count / completed_count if completed_count else 0.0,
        "task_success_rate_over_expected": successful_count / expected_count if expected_count else 0.0,
        "clean_task_success_rate": clean_successful_count / completed_count if completed_count else 0.0,
        "clean_task_success_rate_over_expected": clean_successful_count / expected_count if expected_count else 0.0,
        "infrastructure_error_tasks": len(infrastructure_errors),
        "infrastructure_error_rate": len(infrastructure_errors) / expected_count if expected_count else 0.0,
        "execution_error_total": execution_error_total,
        "execution_error_rate": execution_error_total / completed_count if completed_count else 0.0,
        "tasks_with_execution_errors": sum(value > 0 for value in execution_error_counts),
        "collateral_damage_tasks": len(collateral_damage_tasks),
        "collateral_damage_rate": len(collateral_damage_tasks) / completed_count if completed_count else 0.0,
        "post_compaction_name_error_total": sum(post_compaction_name_errors),
        "tasks_with_post_compaction_name_error": sum(
            value > 0 for value in post_compaction_name_errors
        ),
        "mean_model_turns": _mean(rollout_turns),
        "mean_action_attempts": _mean(action_attempts),
        "mean_environment_steps": _mean(environment_steps),
        "mean_process_steps": _mean(process_steps),
        "tasks_with_compaction": sum(value > 0 for value in compaction_counts),
        "compaction_rate": sum(value > 0 for value in compaction_counts) / completed_count if completed_count else 0.0,
        "total_compactions": int(sum(compaction_counts)),
        "mean_compactions": _mean(compaction_counts),
        "completed_scenarios": len(completed_scenarios),
        "expected_scenarios": expected_scenarios,
        "infrastructure_pending_scenarios": expected_scenarios - len(completed_scenarios),
        "successful_scenarios": len(successful_scenarios),
        "scenario_success_rate": (
            len(successful_scenarios) / len(completed_scenarios)
            if completed_scenarios
            else 0.0
        ),
        "scenario_success_rate_over_expected": (
            len(successful_scenarios) / expected_scenarios if expected_scenarios else 0.0
        ),
        "passed_requirements": passed_requirements,
        "total_requirements": total_requirements,
        "requirement_pass_rate": (
            passed_requirements / total_requirements if total_requirements else None
        ),
        "end_reason_counts": dict(sorted(end_reasons.items())),
        "stopped_early": collection_stop is not None or len(completed) < len(manifest_rows),
        "stop_reason": str(collection_stop["reason"]) if collection_stop else None,
        "metric_denominators": {
            "completed_task_rates": completed_count,
            "expected_task_rates": expected_count,
            "execution_and_collateral_rates": completed_count,
            "post_compaction_name_error_tasks": completed_count,
        },
    }
    if collection_stop:
        payload["pending_task_id"] = str(collection_stop["task_id"])
        payload["stop_error"] = str(collection_stop["error"])
    return payload


def evaluate_frozen_policy(
    *,
    client: TeacherModelClient,
    adapter_factory: Callable[[], Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    condition: str,
    split: str = "appworld_rl_v1_final",
    max_model_turns: int = 40,
    harness_config: TeacherHarnessConfig | None = None,
    resume: bool = False,
    run_config: Mapping[str, Any] | None = None,
    context_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen policy condition without producing training data."""

    manifest = Path(manifest_path).resolve()
    manifest_bytes, manifest_rows = _load_manifest(manifest)
    task_ids = [str(row["task_id"]) for row in manifest_rows]
    task_order = {task_id: index for index, task_id in enumerate(task_ids)}
    harness = harness_config or TeacherHarnessConfig()
    destination = Path(output_dir)
    if destination.exists() and not resume:
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "schema_version": FROZEN_EVALUATION_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "condition": condition,
        "split": split,
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "task_ids_sha256": stable_hash(task_ids),
        "expected_tasks": len(task_ids),
        "model": model_name,
        "provider": {
            "base_url": getattr(client, "base_url", None),
            "disable_parallel_tool_calls": getattr(
                client, "disable_provider_parallel_tool_calls", None
            ),
        },
        "system_prompt_sha256": stable_hash(POLICY_SYSTEM_PROMPT),
        "tool_schema_sha256": stable_hash(AGENT_TOOL_SCHEMAS),
        "harness_version": HARNESS_VERSION,
        "harness": harness.to_dict(),
        "max_model_turns": max_model_turns,
    }
    if run_config:
        config["extra"] = redact_sensitive_data(dict(run_config))
    config["run_contract_hash"] = stable_hash(_config_contract(config))
    config_path = destination / "run_config.json"
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"cannot resume without {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if _config_contract(existing) != _config_contract(config):
            raise ValueError("resume configuration does not match existing frozen evaluation")
    else:
        _write_json(config_path, config)

    raw_path = destination / "raw.jsonl"
    if not resume:
        raw_path.write_bytes(b"")
    attempt_rows = _load_jsonl(raw_path)
    attempt_counts, completed = _validate_attempt_log(attempt_rows, task_ids=set(task_ids))
    collection_stop: dict[str, Any] | None = None
    for task_id in task_ids:
        if task_id in completed:
            continue
        adapter: Any | None = None
        quota_error: TeacherQuotaError | None = None
        release_error: Exception | None = None
        try:
            adapter = adapter_factory()
            record = run_teacher_rollout(
                adapter=adapter,
                client=client,
                task_id=task_id,
                split=split,
                max_model_turns=max_model_turns,
                harness_config=harness,
                system_prompt=POLICY_SYSTEM_PROMPT,
                context_token_counter=context_token_counter,
            )
        except TeacherQuotaError as exc:
            quota_error = exc
        except Exception as exc:
            record = _failure_record(task_id, split=split, error=exc)
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
                "error": redact_sensitive_data(type(quota_error).__name__ + ": " + str(quota_error)),
            }
            break
        if release_error is not None:
            record["infrastructure_invalid"] = True
            rollout = record.setdefault("rollout", {})
            rollout["end_reason"] = "release_error"
            rollout.setdefault("errors", []).append(
                type(release_error).__name__ + ": " + str(release_error)
            )
        record = enrich_rollout_fingerprints(redact_sensitive_data(record))
        record["evaluation_id"] = f"{condition}:{task_id}"
        record["evaluation_attempt_index"] = attempt_counts.get(task_id, 0) + 1
        record["condition"] = condition
        record["model"] = model_name
        record["run_contract_hash"] = config["run_contract_hash"]
        success = (
            not bool(record.get("infrastructure_invalid"))
            and infer_evaluation_success(record.get("evaluation") or {}) is True
            and (record.get("rollout") or {}).get("end_reason") == "evaluator_success"
        )
        record["evaluation_status"] = (
            "infrastructure_error"
            if bool(record.get("infrastructure_invalid"))
            else "success" if success else "failure"
        )
        _append_jsonl(raw_path, record)
        attempt_rows.append(record)
        attempt_counts[task_id] = int(record["evaluation_attempt_index"])
        if bool(record.get("infrastructure_invalid")):
            collection_stop = {
                "reason": "infrastructure_error",
                "task_id": task_id,
                "error": str(((record.get("rollout") or {}).get("errors") or ["unknown"])[-1]),
            }
            break
        completed.add(task_id)

    results = _canonical_results(attempt_rows, task_order=task_order)
    _write_jsonl(destination / "results.jsonl", results)
    _write_jsonl(
        destination / "infrastructure_errors.jsonl",
        (row for row in attempt_rows if bool(row.get("infrastructure_invalid"))),
    )
    summary = _summary(
        attempt_rows=attempt_rows,
        results=results,
        manifest_rows=manifest_rows,
        output_dir=destination,
        collection_stop=collection_stop,
    )
    _write_json(destination / "summary.json", summary)
    return summary
