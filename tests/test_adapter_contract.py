from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from appworld_agentic_rl.adapter import (
    AppWorldAdapter,
    extract_variable_bindings,
    AGENT_TOOL_SCHEMAS,
    execution_error_from_observation,
    looks_state_changing,
    parse_tool_call,
    extract_api_call_provenance,
)
from appworld_agentic_rl.reward import compute_terminal_reward
from appworld_agentic_rl.trajectory import TrajectoryRecord, TrajectoryStep


def test_agent_tool_names_are_the_expected_mvp_contract() -> None:
    names = [tool["function"]["name"] for tool in AGENT_TOOL_SCHEMAS]
    assert names == [
        "inspect_task",
        "search_api_docs",
        "inspect_api_schema",
        "query_state",
        "run_code",
        "run_write_code",
        "complete_task",
        "finish_without_completion",
    ]
    query_state = next(
        tool["function"]
        for tool in AGENT_TOOL_SCHEMAS
        if tool["function"]["name"] == "query_state"
    )
    assert "apis.<app>.<api>" in query_state["description"]


def test_parse_tool_call_accepts_openai_and_compact_forms() -> None:
    assert parse_tool_call({"name": "inspect_task", "arguments": {}}) == ("inspect_task", {})
    assert parse_tool_call(
        {"function": {"name": "search_api_docs", "arguments": "{\"query\": \"calendar\"}"}}
    ) == ("search_api_docs", {"query": "calendar"})


def test_extract_variable_bindings_tracks_api_and_derived_names_without_values() -> None:
    bindings = extract_variable_bindings(
        "auth = apis.venmo.login(username=user, password=supervisor_password)\n"
        "venmo_token = auth['access_token']\n"
        "user_id = auth['user_id']"
    )
    by_name = {item["name"]: item for item in bindings}
    assert by_name["auth"]["kind"] == "api_result"
    assert by_name["auth"]["app"] == "venmo"
    assert by_name["venmo_token"]["semantic"] == "authentication"
    assert by_name["venmo_token"]["derived_from"] == "auth[access_token]"
    assert by_name["user_id"]["kind"] == "derived_field"
    assert all("password" not in str(item).lower() for item in bindings)


def test_extract_api_call_provenance_verifies_observed_and_bound_arguments() -> None:
    docs = {
        ("calendar", "create_event"): {
            "method": "POST",
            "parameters": [
                {"name": "title", "required": True},
                {"name": "calendar_id", "required": True},
            ],
        }
    }
    records = extract_api_call_provenance(
        "apis.calendar.create_event(title=event_title, calendar_id=calendar_id)",
        docs=docs,
        observed_values={"team sync", "cal-123"},
        initial_text="",
        bindings={
            "event_title": {"available": True, "kind": "derived_field"},
            "calendar_id": {"available": False, "kind": "derived_field"},
        },
    )
    assert len(records) == 1
    assert records[0]["all_required_verified"] is False
    by_name = {item["name"]: item for item in records[0]["parameters"]}
    assert by_name["title"]["source"] == "variable_binding"
    assert by_name["title"]["verified"] is True
    assert by_name["calendar_id"]["source"] == "unknown_variable"
    assert by_name["calendar_id"]["verified"] is False
    assert "cal-123" not in str(records)


def test_extract_api_call_provenance_accepts_task_instruction_literal() -> None:
    records = extract_api_call_provenance(
        "apis.calendar.create_event(title='team sync')",
        docs={("calendar", "create_event"): {"method": "POST", "parameters": [{"name": "title", "required": True}]}},
        observed_values=set(),
        initial_text="The event title is team sync.",
    )
    assert records[0]["all_required_verified"] is True
    assert records[0]["parameters"][0]["source"] == "task_instruction"


def test_state_change_heuristic_separates_read_and_write_calls() -> None:
    assert looks_state_changing("apis.calendar.create_event(title='x')")
    assert looks_state_changing("apis.supervisor.complete_task()")
    assert not looks_state_changing("print(apis.calendar.search_events(query='x'))")


def test_complete_task_schema_accepts_optional_answer_and_status() -> None:
    complete = next(
        tool["function"] for tool in AGENT_TOOL_SCHEMAS if tool["function"]["name"] == "complete_task"
    )
    properties = complete["parameters"]["properties"]
    assert "answer" in properties
    assert properties["status"]["enum"] == ["success", "fail"]


def test_execution_error_is_extracted_from_appworld_failure_text() -> None:
    failure = "Execution failed. Traceback:\nException: unauthorized"
    assert execution_error_from_observation(failure) == failure
    assert execution_error_from_observation("Execution successful.") is None


def test_read_only_tools_reject_runtime_introspection_pattern() -> None:
    from appworld_agentic_rl.adapter import QUERY_EXPLORATION_PATTERN

    assert QUERY_EXPLORATION_PATTERN.search("print(dir(apis.spotify))")
    assert QUERY_EXPLORATION_PATTERN.search("import appworld")
    assert not QUERY_EXPLORATION_PATTERN.search("print(apis.spotify.show_song(song_id=1))")


def test_trajectory_record_serializes_steps() -> None:
    record = TrajectoryRecord(
        task_id="task_1",
        split="train",
        experiment_name="unit",
        initial_observation={"instruction": "do it"},
    )
    record.append(
        TrajectoryStep(
            step_index=0,
            tool_name="inspect_task",
            arguments={},
            observation={"ok": True},
        )
    )
    payload = record.to_dict()
    assert payload["schema_version"] == "appworld-agentic-rl-trajectory-v1"
    assert payload["steps"][0]["tool_name"] == "inspect_task"


def test_api_event_metadata_keeps_routes_but_drops_request_credentials() -> None:
    adapter = AppWorldAdapter()
    adapter.world = SimpleNamespace(
        requester=SimpleNamespace(
            requests=[
                {
                    "method": "get",
                    "url": "http://appworld.local/spotify/library/playlists",
                    "data": {"access_token": "must-not-leak"},
                }
            ]
        )
    )
    assert adapter._api_events_since(0) == [
        {
            "method": "GET",
            "route": "/spotify/library/playlists",
            "app": "spotify",
            "endpoint": "library/playlists",
        }
    ]


def test_appworld_guard_checks_api_schema_and_read_write_method() -> None:
    adapter = AppWorldAdapter()
    adapter.world = SimpleNamespace(
        task=SimpleNamespace(
            allowed_apps=["calendar"],
            api_docs={
                "calendar": {
                    "search_events": {
                        "method": "GET",
                        "parameters": [
                            {"name": "query", "required": True},
                            {"name": "page", "required": False},
                        ],
                    },
                    "create_event": {
                        "method": "POST",
                        "parameters": [{"name": "title", "required": True}],
                    },
                }
            },
        )
    )
    missing = adapter.validate_tool_call(
        {
            "name": "query_state",
            "arguments": {"code": "print(apis.calendar.search_events(page=1))"},
        }
    )
    assert {item["code"] for item in missing} == {"missing_api_parameters"}

    wrong_tool = adapter.validate_tool_call(
        {
            "name": "query_state",
            "arguments": {"code": "apis.calendar.create_event(title='x')"},
        }
    )
    assert "write_api_in_read_tool" in {item["code"] for item in wrong_tool}

    unknown = adapter.validate_tool_call(
        {
            "name": "query_state",
            "arguments": {"code": "print(apis.calendar.guessed(query='x'))"},
        }
    )
    assert "api_not_found" in {item["code"] for item in unknown}


def test_core_safety_blocks_imports_even_in_write_tool() -> None:
    adapter = AppWorldAdapter()
    adapter.world = SimpleNamespace(execute=lambda code: code)
    with pytest.raises(ValueError, match="imports are not allowed"):
        adapter._execute_guarded("import requests", require_read_only=False)


def test_local_appworld_evaluation_uses_isolated_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = False
    captured: dict[str, object] = {}

    def save() -> None:
        nonlocal saved
        saved = True

    def fake_subprocess_evaluation(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"data": {"success": True}}

    adapter = AppWorldAdapter(experiment_name="isolated_evaluator")
    adapter.world = SimpleNamespace(
        remote_environment_url=None,
        save=save,
        evaluate=lambda **_: pytest.fail("local evaluator ran in the rollout process"),
    )
    adapter.trajectory = TrajectoryRecord(
        task_id="task_1",
        split="train",
        experiment_name="isolated_evaluator",
        initial_observation={},
    )
    monkeypatch.setattr(
        "appworld_agentic_rl.adapter._evaluate_appworld_subprocess",
        fake_subprocess_evaluation,
    )

    result = adapter.evaluate()

    assert saved is True
    assert captured == {
        "task_id": "task_1",
        "experiment_name": "isolated_evaluator",
        "appworld_root": None,
        "suppress_errors": True,
    }
    assert result["evaluation"]["data"]["success"] is True
    assert result["terminal_reward"] == 1.0


def test_evaluator_subprocess_forces_utf8_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        captured.update(kwargs)
        output_index = command.index("--output") + 1
        Path(command[output_index]).write_text(
            '{"evaluation": {"data": {"success": true}}}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("appworld_agentic_rl.adapter.subprocess.run", fake_run)

    from appworld_agentic_rl.adapter import _evaluate_appworld_subprocess

    result = _evaluate_appworld_subprocess(
        task_id="task_1",
        experiment_name="utf8_evaluator",
        appworld_root=None,
        suppress_errors=True,
    )

    assert result == {"data": {"success": True}}
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["env"]["PYTHONUTF8"] == "1"


def test_appworld_environment_log_accepts_unicode_on_windows(
) -> None:
    AppWorld = pytest.importorskip("appworld.environment").AppWorld

    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        world = SimpleNamespace(
            output_logs_directory=temporary_directory,
            environment_io=[
                {"number": "1", "input": "print('ok')", "output": "🎧 New Headphones"}
            ],
            _maybe_raise_remote_environment_error=lambda _: None,
        )

        AppWorld._save_environment_io_log(world)

        content = (Path(temporary_directory) / "environment_io.md").read_text(
            encoding="utf-8"
        )
        assert "🎧 New Headphones" in content


@pytest.mark.parametrize(
    ("evaluation", "reward"),
    [
        ({"success": True}, 1.0),
        ({"data": {"success": True}}, 1.0),
        ({"data": {"success": True}, "report": "Num Failed Tests : 0"}, 1.0),
        ({"success": True, "collateral_damage": True}, 0.7),
        ({"success": False}, -0.3),
        ({"success": False, "collateral_damage": True}, -0.7),
    ],
)
def test_terminal_reward_basics(evaluation: dict, reward: float) -> None:
    assert compute_terminal_reward(evaluation) == reward
