"""Offline replay and evaluator recovery for persisted teacher trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from appworld_agentic_rl.reward import infer_evaluation_success
from appworld_agentic_rl.harness import stable_hash
from appworld_agentic_rl.teacher_rollout import redact_sensitive_data
from appworld_agentic_rl.trajectory import utc_now_iso


REPLAY_SCHEMA_VERSION = "appworld-teacher-trajectory-replay-v1"
ReplaySelection = Literal[
    "terminal-evaluator-errors",
    "all-evaluator-errors",
    "all-trajectories",
]
REPLAY_SELECTIONS: tuple[ReplaySelection, ...] = (
    "terminal-evaluator-errors",
    "all-evaluator-errors",
    "all-trajectories",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
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


def _trajectory_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def has_terminal_tool(record: Mapping[str, Any]) -> bool:
    """Return whether a persisted trajectory reached a hard terminal agent tool."""

    return any(
        step.get("tool_name") in {"complete_task", "finish_without_completion"}
        for step in _trajectory_steps(record)
    )


def is_replay_candidate(record: Mapping[str, Any], selection: ReplaySelection) -> bool:
    """Apply the deterministic offline replay selection policy."""

    if selection not in REPLAY_SELECTIONS:
        raise ValueError(f"unknown replay selection: {selection}")
    if not _trajectory_steps(record):
        return False
    if selection == "all-trajectories":
        return True
    rollout = record.get("rollout")
    end_reason = rollout.get("end_reason") if isinstance(rollout, Mapping) else None
    if end_reason != "evaluator_error":
        return False
    if selection == "all-evaluator-errors":
        return True
    return has_terminal_tool(record)


def _source_identity(
    record: Mapping[str, Any], *, source_line_number: int
) -> dict[str, Any]:
    return {
        "task_id": str(record.get("task_id") or ""),
        "attempt_index": int(record.get("attempt_index") or 0),
        "rollout_id": str(record.get("rollout_id") or ""),
        "source_line_number": source_line_number,
        "source_record_hash": stable_hash(record),
        "action_hash": record.get("action_hash"),
        "api_plan_hash": record.get("api_plan_hash"),
        "state_delta_hash": record.get("state_delta_hash"),
        "data_manifest_hash": record.get("data_manifest_hash"),
        "run_contract_hash": record.get("run_contract_hash"),
    }


def replay_trajectory(
    record: Mapping[str, Any],
    *,
    adapter_factory: Callable[[], Any],
    source_line_number: int = 0,
) -> dict[str, Any]:
    """Replay one stored agent trajectory in a fresh environment without a model."""

    identity = _source_identity(record, source_line_number=source_line_number)
    task_id = identity["task_id"]
    split = record.get("split")
    steps = _trajectory_steps(record)
    started_at = utc_now_iso()
    adapter: Any | None = None
    step_audit: list[dict[str, Any]] = []
    replay_error: str | None = None
    release_error: str | None = None
    result: dict[str, Any] = {}

    try:
        if not task_id:
            raise ValueError("source record has no task_id")
        if not steps:
            raise ValueError("source record has no replayable trajectory steps")
        adapter = adapter_factory()
        adapter.reset(task_id, split=str(split) if split else None)
        for replay_index, source_step in enumerate(steps):
            tool_name = str(source_step.get("tool_name") or "")
            arguments = source_step.get("arguments")
            if not tool_name:
                raise ValueError(f"source trajectory step {replay_index} has no tool_name")
            if not isinstance(arguments, Mapping):
                raise TypeError(
                    f"source trajectory step {replay_index} arguments must be an object"
                )
            step_result = adapter.step(
                {"name": tool_name, "arguments": deepcopy(dict(arguments))}
            )
            if not isinstance(step_result, Mapping):
                raise TypeError(f"adapter step {replay_index} result must be an object")
            expected_error = bool(source_step.get("error"))
            actual_error = bool(step_result.get("error"))
            expected_terminated = bool(source_step.get("terminated"))
            actual_terminated = bool(step_result.get("terminated"))
            step_audit.append(
                {
                    "step_index": replay_index,
                    "tool_name": tool_name,
                    "expected_error": expected_error,
                    "actual_error": actual_error,
                    "error_match": expected_error == actual_error,
                    "expected_terminated": expected_terminated,
                    "actual_terminated": actual_terminated,
                    "terminated_match": expected_terminated == actual_terminated,
                }
            )

        evaluation_result = adapter.evaluate()
        if not isinstance(evaluation_result, Mapping):
            raise TypeError("adapter evaluation result must be an object")
        evaluation = evaluation_result.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("adapter evaluation has no evaluation object")
        evaluation = dict(evaluation)
        evaluation_error = evaluation.get("error")
        replay_status = "evaluator_error" if evaluation_error else "completed"
        result = {
            "replay_status": replay_status,
            "evaluation": evaluation,
            "evaluation_success": infer_evaluation_success(evaluation),
            "terminal_reward": evaluation_result.get("terminal_reward"),
            "trajectory": adapter.export_trajectory(),
        }
    except Exception as exc:
        replay_error = type(exc).__name__ + ": " + str(exc)
        result = {
            "replay_status": "replay_error",
            "evaluation": None,
            "evaluation_success": None,
            "terminal_reward": None,
            "trajectory": None,
        }
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                release_error = type(exc).__name__ + ": " + str(exc)

    if release_error is not None:
        result["replay_status"] = "release_error"
    output = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source": identity,
        "task_id": task_id,
        "attempt_index": identity["attempt_index"],
        "rollout_id": identity["rollout_id"],
        "split": split,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "source_step_count": len(steps),
        "replayed_step_count": len(step_audit),
        "step_audit": step_audit,
        "step_outcome_mismatches": sum(
            not item["error_match"] or not item["terminated_match"]
            for item in step_audit
        ),
        "replay_error": replay_error,
        "release_error": release_error,
        **result,
    }
    return redact_sensitive_data(output)


def _summary(
    rows: list[Mapping[str, Any]],
    *,
    source_records: int,
    eligible_records: int,
    selected_records: int,
    source_raw_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    statuses = Counter(str(row.get("replay_status") or "unknown") for row in rows)
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "source_records": source_records,
        "eligible_records": eligible_records,
        "selected_records": selected_records,
        "completed_records": len(rows),
        "pending_records": max(0, selected_records - len(rows)),
        "evaluation_successes": sum(row.get("evaluation_success") is True for row in rows),
        "evaluation_failures": sum(row.get("evaluation_success") is False for row in rows),
        "step_outcome_mismatches": sum(
            int(row.get("step_outcome_mismatches") or 0) for row in rows
        ),
        "replay_status_counts": dict(sorted(statuses.items())),
        "source_raw_sha256": source_raw_sha256,
        "source_unchanged": True,
    }


def _config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in config.items() if key != "created_at"}


def replay_teacher_trajectories(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    adapter_factory: Callable[[], Any],
    selection: ReplaySelection = "terminal-evaluator-errors",
    task_ids: Iterable[str] | None = None,
    rollout_ids: Iterable[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    replay_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay selected teacher trajectories while treating source raw data as immutable."""

    if selection not in REPLAY_SELECTIONS:
        raise ValueError(f"unknown replay selection: {selection}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source = Path(source_dir).resolve()
    destination = Path(output_dir).resolve()
    if source == destination:
        raise ValueError("replay output directory must differ from the source directory")
    raw_path = source / "raw.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(f"teacher source raw.jsonl does not exist: {raw_path}")

    source_hash = _sha256_file(raw_path)
    source_rows = _load_jsonl(raw_path)
    requested_task_ids = list(dict.fromkeys(task_ids or []))
    requested_rollout_ids = list(dict.fromkeys(rollout_ids or []))
    eligible: list[tuple[int, dict[str, Any]]] = []
    for source_line_number, record in enumerate(source_rows, 1):
        if requested_task_ids and str(record.get("task_id") or "") not in requested_task_ids:
            continue
        if (
            requested_rollout_ids
            and str(record.get("rollout_id") or "") not in requested_rollout_ids
        ):
            continue
        if is_replay_candidate(record, selection):
            eligible.append((source_line_number, record))
    selected = eligible[:limit] if limit is not None else eligible

    config: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "source_dir": str(source),
        "source_raw_path": str(raw_path),
        "source_raw_sha256": source_hash,
        "selection": selection,
        "task_ids": requested_task_ids,
        "limit": limit,
        "external_model_calls": False,
        "replay": dict(replay_config or {}),
    }
    if requested_rollout_ids:
        config["rollout_ids"] = requested_rollout_ids
    config["replay_contract_hash"] = stable_hash(_config_contract(config))
    config_path = destination / "run_config.json"
    replay_path = destination / "replay.jsonl"

    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"replay output is not a directory: {destination}")
        if not resume:
            raise FileExistsError(f"replay output already exists: {destination}")
        if not config_path.is_file():
            raise FileNotFoundError(f"cannot resume without run_config.json: {config_path}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if _config_contract(existing_config) != _config_contract(config):
            raise ValueError("resume configuration does not match the existing replay")
    else:
        destination.mkdir(parents=True)
        replay_path.write_bytes(b"")
        _write_json(config_path, config)

    replay_rows = _load_jsonl(replay_path)
    completed: set[str] = set()
    for row in replay_rows:
        rollout_id = str(row.get("rollout_id") or "")
        if not rollout_id:
            raise ValueError("replay resume data has a row without rollout_id")
        if rollout_id in completed:
            raise ValueError(f"duplicate replay result for rollout_id: {rollout_id}")
        completed.add(rollout_id)

    for source_line_number, record in selected:
        rollout_id = str(record.get("rollout_id") or "")
        if not rollout_id:
            raise ValueError(f"source record at line {source_line_number} has no rollout_id")
        if rollout_id in completed:
            continue
        replay_row = replay_trajectory(
            record,
            adapter_factory=adapter_factory,
            source_line_number=source_line_number,
        )
        _append_jsonl(replay_path, replay_row)
        replay_rows.append(replay_row)
        completed.add(rollout_id)
        interim_summary = _summary(
            replay_rows,
            source_records=len(source_rows),
            eligible_records=len(eligible),
            selected_records=len(selected),
            source_raw_sha256=source_hash,
            output_dir=destination,
        )
        _write_json(destination / "summary.json", interim_summary)

    final_hash = _sha256_file(raw_path)
    if final_hash != source_hash:
        raise RuntimeError("source raw.jsonl changed while replay was running")
    summary = _summary(
        replay_rows,
        source_records=len(source_rows),
        eligible_records=len(eligible),
        selected_records=len(selected),
        source_raw_sha256=source_hash,
        output_dir=destination,
    )
    _write_json(destination / "summary.json", summary)
    return summary
