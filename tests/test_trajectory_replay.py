from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from appworld_agentic_rl.trajectory_replay import (
    is_replay_candidate,
    replay_teacher_trajectories,
    replay_trajectory,
)


def _source_record(
    *,
    rollout_id: str = "task_a:attempt-001",
    end_reason: str = "evaluator_error",
    terminal: bool = True,
) -> dict[str, Any]:
    steps = [
        {
            "step_index": 0,
            "tool_name": "query_state",
            "arguments": {"code": "value = 1\nprint(value)"},
            "observation": "1\n",
            "error": None,
            "state_change": False,
            "terminated": False,
        }
    ]
    if terminal:
        steps.append(
            {
                "step_index": 1,
                "tool_name": "complete_task",
                "arguments": {"status": "success"},
                "observation": {"message": "done"},
                "error": None,
                "state_change": True,
                "terminated": True,
            }
        )
    task_id, attempt_text = rollout_id.split(":attempt-")
    return {
        "schema_version": "appworld-teacher-rollout-v2",
        "task_id": task_id,
        "split": "appworld_rl_v1_teacher_sft",
        "rollout_id": rollout_id,
        "attempt_index": int(attempt_text),
        "rollout": {"end_reason": end_reason},
        "trajectory": {"steps": steps},
        "evaluation": {"error": "RecursionError: maximum recursion depth exceeded"},
        "terminal_reward": -0.7,
        "action_hash": "action-hash",
        "api_plan_hash": "api-plan-hash",
        "state_delta_hash": "state-delta-hash",
        "data_manifest_hash": "manifest-hash",
        "run_contract_hash": "contract-hash",
    }


class FakeReplayAdapter:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls
        self.task_id = ""
        self.split: str | None = None
        self.steps: list[dict[str, Any]] = []

    def reset(self, task_id: str, *, split: str | None = None) -> dict[str, Any]:
        self.task_id = task_id
        self.split = split
        self.calls.append(("reset", (task_id, split)))
        return {"instruction": "test"}

    def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("step", deepcopy(tool_call)))
        terminated = tool_call["name"] == "complete_task"
        result = {
            "tool_name": tool_call["name"],
            "observation": {"ok": True},
            "error": None,
            "state_change": terminated,
            "terminated": terminated,
        }
        self.steps.append(
            {
                "step_index": len(self.steps),
                "tool_name": tool_call["name"],
                "arguments": deepcopy(tool_call["arguments"]),
                **{key: result[key] for key in ("observation", "error", "state_change", "terminated")},
            }
        )
        return result

    def evaluate(self) -> dict[str, Any]:
        self.calls.append(("evaluate", None))
        return {"evaluation": {"data": {"success": True}}, "terminal_reward": 1.0}

    def export_trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "steps": deepcopy(self.steps),
            "evaluation": {"data": {"success": True}},
            "terminal_reward": 1.0,
        }

    def close(self) -> None:
        self.calls.append(("close", None))


@pytest.fixture
def replay_tmp_path() -> Any:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        yield Path(temporary_directory)


def test_default_replay_selection_requires_evaluator_error_and_terminal_tool() -> None:
    terminal_error = _source_record()
    nonterminal_error = _source_record(terminal=False)
    valid_terminal = _source_record(end_reason="evaluator_success")

    assert is_replay_candidate(terminal_error, "terminal-evaluator-errors") is True
    assert is_replay_candidate(nonterminal_error, "terminal-evaluator-errors") is False
    assert is_replay_candidate(nonterminal_error, "all-evaluator-errors") is True
    assert is_replay_candidate(valid_terminal, "all-trajectories") is True


def test_replay_trajectory_preserves_identity_and_replays_stored_actions() -> None:
    calls: list[tuple[str, Any]] = []
    record = _source_record()

    result = replay_trajectory(
        record,
        adapter_factory=lambda: FakeReplayAdapter(calls),
        source_line_number=17,
    )

    assert result["replay_status"] == "completed"
    assert result["evaluation_success"] is True
    assert result["terminal_reward"] == 1.0
    assert result["rollout_id"] == record["rollout_id"]
    assert result["source"]["source_line_number"] == 17
    assert result["source"]["action_hash"] == "action-hash"
    assert result["replayed_step_count"] == 2
    assert result["step_outcome_mismatches"] == 0
    assert [name for name, _ in calls] == ["reset", "step", "step", "evaluate", "close"]
    assert calls[1][1] == {
        "name": "query_state",
        "arguments": {"code": "value = 1\nprint(value)"},
    }


def test_collection_keeps_source_raw_immutable_and_resume_skips_completed(
    replay_tmp_path: Path,
) -> None:
    tmp_path = replay_tmp_path
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_rows = [
        _source_record(),
        _source_record(rollout_id="task_b:attempt-002", terminal=False),
        _source_record(
            rollout_id="task_c:attempt-003",
            end_reason="provider_error",
        ),
    ]
    raw_path = source_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    raw_before = raw_path.read_bytes()
    calls: list[tuple[str, Any]] = []
    output_dir = tmp_path / "replay"

    summary = replay_teacher_trajectories(
        source_dir=source_dir,
        output_dir=output_dir,
        adapter_factory=lambda: FakeReplayAdapter(calls),
        rollout_ids=["task_a:attempt-001"],
        replay_config={"experiment_name": "unit-replay"},
    )

    assert raw_path.read_bytes() == raw_before
    assert summary["source_records"] == 3
    assert summary["eligible_records"] == 1
    assert summary["selected_records"] == 1
    assert summary["completed_records"] == 1
    assert summary["evaluation_successes"] == 1
    replay_rows = [
        json.loads(line)
        for line in (output_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["rollout_id"] for row in replay_rows] == ["task_a:attempt-001"]

    calls_before_resume = len(calls)
    resumed = replay_teacher_trajectories(
        source_dir=source_dir,
        output_dir=output_dir,
        adapter_factory=lambda: FakeReplayAdapter(calls),
        rollout_ids=["task_a:attempt-001"],
        resume=True,
        replay_config={"experiment_name": "unit-replay"},
    )
    assert len(calls) == calls_before_resume
    assert resumed["completed_records"] == 1
    assert raw_path.read_bytes() == raw_before


def test_replay_error_is_recorded_without_aborting_the_collection(
    replay_tmp_path: Path,
) -> None:
    tmp_path = replay_tmp_path
    class BrokenAdapter:
        def reset(self, task_id: str, *, split: str | None = None) -> None:
            raise RuntimeError("fresh environment unavailable")

        def close(self) -> None:
            return None

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "raw.jsonl").write_text(
        json.dumps(_source_record()) + "\n",
        encoding="utf-8",
    )

    summary = replay_teacher_trajectories(
        source_dir=source_dir,
        output_dir=tmp_path / "replay",
        adapter_factory=BrokenAdapter,
    )

    assert summary["replay_status_counts"] == {"replay_error": 1}
    row = json.loads(
        (tmp_path / "replay" / "replay.jsonl").read_text(encoding="utf-8")
    )
    assert row["replay_error"] == "RuntimeError: fresh environment unavailable"
    assert row["replayed_step_count"] == 0
