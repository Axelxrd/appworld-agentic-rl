from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from appworld_agentic_rl.sft_dataset import build_action_sft_dataset


def _record(
    task_id: str,
    attempt_index: int,
    *,
    accepted: bool,
    step_error: str | None = None,
) -> dict[str, Any]:
    steps = [
        {
            "tool_name": "inspect_task",
            "arguments": {},
            "observation": {"task_id": task_id},
            "error": step_error,
            "state_change": False,
        },
        {
            "tool_name": "complete_task",
            "arguments": {"summary": "done"},
            "observation": "Execution successful.",
            "error": None,
            "state_change": True,
        },
    ]
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "rollout_id": f"{task_id}:attempt-{attempt_index:03d}",
        "collection_status": "accepted" if accepted else "rejected",
        "infrastructure_invalid": False,
        "process_metrics": {"execution_errors": int(step_error is not None)},
        "conversation": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "unexecuted",
                        "type": "function",
                        "function": {"name": "run_write_code", "arguments": "{}"},
                    }
                ],
            },
        ],
        "trajectory": {"steps": steps},
    }


def _write_raw(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True)
    (directory / "raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_sft_data_overlays_recovery_and_keeps_all_clean_rows(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    output = tmp_path / "output"
    _write_raw(
        base,
        [
            _record("task_a", 1, accepted=False),
            _record("task_b", 1, accepted=True),
            _record("task_c", 1, accepted=True, step_error="bad call"),
        ],
    )
    _write_raw(recovery, [_record("task_a", 1, accepted=True)])

    metadata = build_action_sft_dataset(
        source_dirs=[base, recovery],
        output_dir=output,
        validation_ratio=0.5,
        seed=7,
    )

    rows = [json.loads(line) for line in (output / "sft.jsonl").read_text().splitlines()]
    assert metadata["selection"]["effective_slots"] == 3
    assert metadata["selection"]["rows"] == 2
    assert metadata["selection"]["action_targets"] == 4
    assert {row["task_id"] for row in rows} == {"task_a", "task_b"}
    assert rows[0]["messages"][0]["content"].startswith(
        "You are an AppWorld agent solving one task"
    )
    assert all(len(row["messages"]) == 6 for row in rows)
    assert all(
        call["function"]["name"] != "run_write_code"
        for row in rows
        for message in row["messages"]
        for call in message.get("tool_calls", [])
    )
    train_tasks = {
        json.loads(line)["task_id"]
        for line in (output / "train.jsonl").read_text().splitlines()
    }
    validation_tasks = {
        json.loads(line)["task_id"]
        for line in (output / "validation.jsonl").read_text().splitlines()
    }
    assert train_tasks
    assert validation_tasks
    assert train_tasks.isdisjoint(validation_tasks)


def test_build_sft_data_rejects_unknown_recovery_slot(tmp_path: Path) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    _write_raw(base, [_record("task_a", 1, accepted=True)])
    _write_raw(recovery, [_record("task_b", 1, accepted=True)])
    with pytest.raises(ValueError, match="absent from earlier sources"):
        build_action_sft_dataset(
            source_dirs=[base, recovery],
            output_dir=tmp_path / "output",
            validation_ratio=0.5,
        )


def test_default_split_uses_every_clean_row_for_training(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_raw(
        source,
        [
            _record("task_a", 1, accepted=True),
            _record("task_b", 1, accepted=True),
        ],
    )

    metadata = build_action_sft_dataset(source_dirs=[source], output_dir=output)

    assert metadata["split"]["train_rows"] == 2
    assert metadata["split"]["validation_rows"] == 0
    assert (output / "train.jsonl").read_bytes() == (output / "sft.jsonl").read_bytes()
    assert (output / "validation.jsonl").read_bytes() == b""
