from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from appworld_agentic_rl.teacher_client import TeacherProviderError, TeacherQuotaError
from appworld_agentic_rl.teacher_collector import collect_teacher_rollouts
from appworld_agentic_rl.teacher_recovery import (
    recover_teacher_provider_errors,
    teacher_recovery_settings,
)


def _tool_call() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "complete",
                "type": "function",
                "function": {"name": "complete_task", "arguments": "{}"},
            }
        ],
    }


class SequenceClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests = 0
        self.model = "fake-teacher"
        self.base_url = "https://provider.invalid/v1"
        self.temperature = 0.0
        self.max_tokens = 1024
        self.timeout = 60.0
        self.max_retries = 3
        self.retry_backoff_seconds = 1.0
        self.disable_provider_parallel_tool_calls = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.requests += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reset(self, task_id: str, *, split: str) -> dict[str, Any]:
        self.task_id = task_id
        return {
            "task_id": task_id,
            "instruction": "Complete the task.",
            "current_datetime": "2026-01-01T00:00:00",
            "allowed_apps": ["calendar"],
            "supervisor": {},
        }

    def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tool_call)
        return {
            "tool_name": "complete_task",
            "observation": {"ok": True},
            "error": None,
            "state_change": True,
            "terminated": True,
        }

    def evaluate(self) -> dict[str, Any]:
        return {"evaluation": {"data": {"success": True}}, "terminal_reward": 1.0}

    def export_trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [{"tool_name": "complete_task", "error": None}],
        }

    def close(self) -> None:
        return None


def _create_complete_source(root: Path) -> Path:
    source = root / "source"
    collect_teacher_rollouts(
        client=SequenceClient(
            [
                TeacherProviderError("network down"),
                _tool_call(),
                TeacherProviderError("network down again"),
            ]
        ),
        adapter_factory=FakeAdapter,
        output_dir=source,
        model_name="fake-teacher",
        task_ids=["task_a"],
        attempts_per_task=3,
        task_id_loader=lambda split: ["task_a"],
    )
    return source


def test_recovery_collects_only_exact_provider_error_slots_and_keeps_source_immutable() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        root = Path(temporary_directory)
        source = _create_complete_source(root)
        source_before = (source / "raw.jsonl").read_bytes()
        output = root / "recovery"

        summary = recover_teacher_provider_errors(
            client=SequenceClient([_tool_call(), _tool_call()]),
            adapter_factory=FakeAdapter,
            source_dir=source,
            output_dir=output,
        )

        rows = [
            json.loads(line)
            for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [row["rollout_id"] for row in rows] == [
            "task_a:attempt-001",
            "task_a:attempt-003",
        ]
        assert all(row["collection_status"] == "accepted" for row in rows)
        assert summary["num_rollouts"] == 2
        assert summary["provider_error_slots"] == 2
        assert (source / "raw.jsonl").read_bytes() == source_before


def test_recovery_quota_stop_does_not_consume_slot_and_resume_has_no_duplicates() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        root = Path(temporary_directory)
        source = _create_complete_source(root)
        output = root / "recovery"

        stopped = recover_teacher_provider_errors(
            client=SequenceClient([TeacherQuotaError("quota exhausted")]),
            adapter_factory=FakeAdapter,
            source_dir=source,
            output_dir=output,
        )
        assert stopped["stopped_early"] is True
        assert (output / "raw.jsonl").read_bytes() == b""

        resumed = recover_teacher_provider_errors(
            client=SequenceClient([_tool_call(), _tool_call()]),
            adapter_factory=FakeAdapter,
            source_dir=source,
            output_dir=output,
            resume=True,
        )
        rows = (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 2
        assert len({json.loads(row)["rollout_id"] for row in rows}) == 2
        assert resumed["stopped_early"] is False


def test_recovery_refuses_an_incomplete_source() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        root = Path(temporary_directory)
        source = _create_complete_source(root)
        raw_path = source / "raw.jsonl"
        raw_path.write_text(raw_path.read_text(encoding="utf-8").splitlines()[0] + "\n")

        with pytest.raises(ValueError, match="source collection is incomplete"):
            recover_teacher_provider_errors(
                client=SequenceClient([]),
                adapter_factory=FakeAdapter,
                source_dir=source,
                output_dir=root / "recovery",
            )


def test_recovery_can_recursively_recover_exact_slots() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        root = Path(temporary_directory)
        source = _create_complete_source(root)
        first_recovery = root / "recovery_v1"
        recover_teacher_provider_errors(
            client=SequenceClient(
                [TeacherProviderError("network still down"), _tool_call()]
            ),
            adapter_factory=FakeAdapter,
            source_dir=source,
            output_dir=first_recovery,
        )

        second_recovery = root / "recovery_v2"
        summary = recover_teacher_provider_errors(
            client=SequenceClient([_tool_call()]),
            adapter_factory=FakeAdapter,
            source_dir=first_recovery,
            output_dir=second_recovery,
        )

        rows = [
            json.loads(line)
            for line in (second_recovery / "raw.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["rollout_id"] for row in rows] == ["task_a:attempt-001"]
        assert rows[0]["collection_status"] == "accepted"
        assert summary["num_rollouts"] == 1
        assert summary["provider_error_slots"] == 1


def test_recovery_settings_follow_legacy_recovery_provenance() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        root = Path(temporary_directory)
        source = _create_complete_source(root)
        first_recovery = root / "recovery_v1"
        recover_teacher_provider_errors(
            client=SequenceClient([_tool_call(), _tool_call()]),
            adapter_factory=FakeAdapter,
            source_dir=source,
            output_dir=first_recovery,
        )
        config_path = first_recovery / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["extra"] = {
            "recovery_schema_version": config["extra"]["recovery_schema_version"],
            "source_raw_sha256": config["extra"]["source_raw_sha256"],
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        settings = teacher_recovery_settings(first_recovery)
        assert settings["model"] == "fake-teacher"
        assert settings["temperature"] == 0.0
        assert settings["max_tokens"] == 1024
