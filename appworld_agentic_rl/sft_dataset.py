"""Build trainer-ready action-only SFT data from teacher rollout runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS
from appworld_agentic_rl.teacher_rollout import POLICY_SYSTEM_PROMPT


SFT_DATASET_SCHEMA_VERSION = "appworld-action-sft-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL source does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        rows.append(row)
    return rows


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _slot(record: Mapping[str, Any]) -> tuple[str, int]:
    task_id = str(record.get("task_id") or "")
    attempt_index = int(record.get("attempt_index") or 0)
    if not task_id or attempt_index < 1:
        raise ValueError("teacher record has no valid task_id/attempt_index")
    return task_id, attempt_index


def _overlay_sources(
    source_dirs: Sequence[Path],
) -> tuple[list[tuple[dict[str, Any], int]], list[dict[str, Any]]]:
    if not source_dirs:
        raise ValueError("at least one teacher source directory is required")
    effective: dict[tuple[str, int], tuple[dict[str, Any], int]] = {}
    source_metadata: list[dict[str, Any]] = []
    for source_index, source_dir in enumerate(source_dirs):
        raw_path = source_dir / "raw.jsonl"
        rows = _load_jsonl(raw_path)
        source_slots: set[tuple[str, int]] = set()
        for record in rows:
            slot = _slot(record)
            if slot in source_slots:
                raise ValueError(f"duplicate attempt slot in {raw_path}: {slot}")
            if source_index and slot not in effective:
                raise ValueError(
                    f"recovery source contains a slot absent from earlier sources: {slot}"
                )
            source_slots.add(slot)
            effective[slot] = (record, source_index)
        source_metadata.append(
            {
                "path": str(source_dir.resolve()),
                "raw_path": str(raw_path.resolve()),
                "raw_rows": len(rows),
                "raw_sha256": _sha256(raw_path),
            }
        )
    ordered = [effective[slot] for slot in sorted(effective)]
    return ordered, source_metadata


def _is_clean_accepted(record: Mapping[str, Any]) -> bool:
    if record.get("collection_status") != "accepted":
        return False
    if bool(record.get("infrastructure_invalid")):
        return False
    metrics = record.get("process_metrics")
    if not isinstance(metrics, Mapping) or int(metrics.get("execution_errors", 0)) != 0:
        return False
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return False
    steps = trajectory.get("steps")
    return bool(steps) and isinstance(steps, list) and all(
        isinstance(step, Mapping) and not step.get("error") for step in steps
    )


def _context_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    # New teacher runs persist the exact post-compaction stream used for SFT.
    # Legacy rows fall back to their original conversation for compatibility.
    conversation = record.get("training_conversation") or record.get("conversation")
    if not isinstance(conversation, list):
        raise ValueError(f"record {_slot(record)} has no conversation")
    messages: list[dict[str, Any]] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role not in {"system", "user"}:
            break
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"record {_slot(record)} has invalid {role} content")
        messages.append({"role": str(role), "content": content})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"record {_slot(record)} has no initial user message")
    return messages


def _tool_content(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    return json.dumps(observation, ensure_ascii=False, sort_keys=True)


def build_action_sft_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild one row strictly from actions that the environment executed."""

    if not _is_clean_accepted(record):
        raise ValueError(f"record {_slot(record)} is not a clean accepted rollout")
    messages = _context_messages(record)
    # Teacher traces are collected with a teacher-role prompt. The SFT policy
    # must see the same neutral prompt used during policy evaluation instead.
    for message in messages:
        if message["role"] == "system":
            message["content"] = POLICY_SYSTEM_PROMPT
            break
    trajectory = record["trajectory"]
    assert isinstance(trajectory, Mapping)
    steps = trajectory["steps"]
    assert isinstance(steps, list)
    for index, step in enumerate(steps, 1):
        assert isinstance(step, Mapping)
        tool_name = str(step.get("tool_name") or "")
        arguments = step.get("arguments")
        if not tool_name or not isinstance(arguments, Mapping):
            raise ValueError(f"record {_slot(record)} has an invalid executed step")
        call_id = f"call_{index:04d}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                dict(arguments),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": call_id,
                "content": _tool_content(step.get("observation")),
            }
        )
    task_id, attempt_index = _slot(record)
    return {
        "schema_version": SFT_DATASET_SCHEMA_VERSION,
        "context_policy": (
            "structured_memory" if isinstance(record.get("training_conversation"), list)
            else "legacy_full_replay"
        ),
        "trajectory_id": str(record.get("rollout_id") or f"{task_id}:attempt-{attempt_index:03d}"),
        "task_id": task_id,
        "attempt_index": attempt_index,
        "messages": messages,
        "tools": deepcopy(AGENT_TOOL_SCHEMAS),
    }


def _validation_task_ids(
    task_ids: Sequence[str], *, validation_ratio: float, seed: int
) -> set[str]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be at least zero and less than one")
    if validation_ratio == 0.0:
        return set()
    unique = sorted(set(task_ids))
    if len(unique) < 2:
        raise ValueError("at least two tasks are required for a task-disjoint split")
    ranked = sorted(
        unique,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, math.floor(len(ranked) * validation_ratio + 0.5)))
    return set(ranked[:count])


def build_action_sft_dataset(
    *,
    source_dirs: Sequence[str | Path],
    output_dir: str | Path,
    validation_ratio: float = 0.0,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Overlay recovery runs and persist all clean verifier-passing SFT rows."""

    sources = [Path(path) for path in source_dirs]
    overlaid, source_metadata = _overlay_sources(sources)
    selected = [
        (record, source_index)
        for record, source_index in overlaid
        if _is_clean_accepted(record)
    ]
    rows = [build_action_sft_row(record) for record, _ in selected]
    validation_tasks = _validation_task_ids(
        [row["task_id"] for row in rows],
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_rows = [row for row in rows if row["task_id"] not in validation_tasks]
    validation_rows = [row for row in rows if row["task_id"] in validation_tasks]
    source_selected = Counter(source_index for _, source_index in selected)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft": destination / "sft.jsonl",
        "train": destination / "train.jsonl",
        "validation": destination / "validation.jsonl",
    }
    _atomic_write_jsonl(paths["sft"], rows)
    _atomic_write_jsonl(paths["train"], train_rows)
    _atomic_write_jsonl(paths["validation"], validation_rows)

    metadata: dict[str, Any] = {
        "schema_version": SFT_DATASET_SCHEMA_VERSION,
        "selection": {
            "policy": "latest exact-slot overlay; all clean verifier-passing rollouts",
            "effective_slots": len(overlaid),
            "rows": len(rows),
            "tasks": len({row["task_id"] for row in rows}),
            "action_targets": sum(
                len(record["trajectory"]["steps"]) for record, _ in selected
            ),
            "source_selected_rows": {
                str(sources[index].resolve()): source_selected.get(index, 0)
                for index in range(len(sources))
            },
        },
        "split": {
            "strategy": (
                "external AppWorld dev evaluation; all teacher rows train"
                if not validation_tasks
                else "task-disjoint hash ranking"
            ),
            "seed": int(seed),
            "validation_ratio": float(validation_ratio),
            "train_rows": len(train_rows),
            "train_tasks": len({row["task_id"] for row in train_rows}),
            "validation_rows": len(validation_rows),
            "validation_tasks": len(validation_tasks),
            "validation_task_ids": sorted(validation_tasks),
        },
        "sources": source_metadata,
        "files": {
            name: {
                "path": str(path.resolve()),
                "rows": len(
                    rows if name == "sft" else train_rows if name == "train" else validation_rows
                ),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }
    _atomic_write_json(destination / "metadata.json", metadata)
    return metadata
