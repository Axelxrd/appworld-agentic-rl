from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.error import HTTPError

import pytest

from appworld_agentic_rl.teacher_collector import collect_teacher_rollouts
from appworld_agentic_rl.teacher_client import (
    OpenAICompatibleTeacherClient,
    TeacherModelError,
    TeacherProviderError,
    TeacherQuotaError,
    _safe_response_detail,
    normalize_assistant_message,
)
from appworld_agentic_rl.teacher_curation import (
    curate_teacher_rollouts,
    extract_planned_api_calls,
)
from appworld_agentic_rl.teacher_rollout import (
    redact_sensitive_data,
    run_teacher_rollout,
    sanitize_assistant_message,
)
from appworld_agentic_rl.harness import (
    TeacherHarnessConfig,
    build_bounded_context,
    build_memory_context,
    build_memory_token_context,
    stable_hash,
)
from appworld_agentic_rl.context_memory import MemoryState


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.requests.append(deepcopy(messages))
        assert len(tools) == 8
        return self.responses.pop(0)


class FakeAdapter:
    def __init__(self, *, success: bool) -> None:
        self.success = success
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def reset(self, task_id: str, *, split: str) -> dict[str, Any]:
        self.task_id = task_id
        assert split == "train"
        return {
            "task_id": task_id,
            "instruction": "Solve this safely.",
            "current_datetime": "2026-01-01T00:00:00",
            "allowed_apps": ["calendar"],
            "supervisor": {"password": "must-not-leak"},
        }

    def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tool_call)
        name = tool_call["function"]["name"]
        return {
            "tool_name": name,
            "observation": {"password": "must-not-leak", "value": len(self.calls)},
            "error": None,
            "state_change": name == "complete_task",
            "terminated": name == "complete_task",
        }

    def evaluate(self) -> dict[str, Any]:
        return {
            "evaluation": {"data": {"success": self.success}},
            "terminal_reward": 1.0 if self.success else -0.3,
        }

    def export_trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [
                {"tool_name": call["function"]["name"], "error": None}
                for call in self.calls
            ],
        }

    def close(self) -> None:
        self.closed = True


def tool_call(name: str, arguments: str = "{}") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def test_rollout_returns_tool_observation_to_the_next_model_turn() -> None:
    client = FakeClient(
        [
            tool_call("inspect_task"),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    adapter = FakeAdapter(success=True)

    record = run_teacher_rollout(adapter=adapter, client=client, task_id="task_a")

    assert [call["function"]["name"] for call in adapter.calls] == ["inspect_task", "complete_task"]
    second_request = client.requests[1]
    assert "apis.<app>.<api>" in second_request[1]["content"]
    assert second_request[-1]["role"] == "tool"
    assert "must-not-leak" not in second_request[-1]["content"]
    assert record["rollout"]["end_reason"] == "evaluator_success"
    assert record["rollout"]["interaction_end_reason"] == "terminal_tool"
    assert record["evaluation"]["data"]["success"] is True


def test_rollout_accepts_a_policy_system_prompt() -> None:
    client = FakeClient([tool_call("complete_task", '{"answer": "done"}')])

    record = run_teacher_rollout(
        adapter=FakeAdapter(success=True),
        client=client,
        task_id="task_a",
        system_prompt="You are an AppWorld policy.",
    )

    assert client.requests[0][0]["content"] == "You are an AppWorld policy."
    assert record["harness"]["system_prompt_hash"] == stable_hash("You are an AppWorld policy.")


def test_rollout_reminds_model_of_api_namespace_after_tool_error() -> None:
    class ErrorAdapter(FakeAdapter):
        def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
            result = super().step(tool_call)
            if tool_call["function"]["name"] == "query_state":
                result["error"] = "NameError: client is not defined"
                result["observation"] = {"error": result["error"]}
            return result

    client = FakeClient(
        [
            tool_call("query_state", '{"code": "client.spotify.login()"}'),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    run_teacher_rollout(
        adapter=ErrorAdapter(success=True), client=client, task_id="task_a"
    )

    assert "apis.<app>.<api>" in client.requests[1][-1]["content"]


def test_rollout_executes_only_the_first_tool_call_per_turn() -> None:
    response = tool_call("inspect_task")
    response["tool_calls"].append(
        {"id": "ignored", "type": "function", "function": {"name": "query_state", "arguments": "{}"}}
    )
    client = FakeClient(
        [response, tool_call("complete_task", '{"answer": "done"}')]
    )
    adapter = FakeAdapter(success=False)
    record = run_teacher_rollout(adapter=adapter, client=client, task_id="task_a")

    assert [call["function"]["name"] for call in adapter.calls] == [
        "inspect_task",
        "complete_task",
    ]
    assert record["rollout"]["dropped_tool_calls"] == [
        {"turn_index": 0, "count": 1, "tool_names": ["query_state"]}
    ]
    second_request = client.requests[1]
    historical_assistant = second_request[-3]
    historical_tools = second_request[-2:]
    assert len(historical_assistant["tool_calls"]) == 2
    assert [message["tool_call_id"] for message in historical_tools] == [
        "call_inspect_task",
        "ignored",
    ]
    assert json.loads(historical_tools[1]["content"])["executed"] is False


def test_rollout_preserves_reasoning_content_in_tool_call_history() -> None:
    first = tool_call("inspect_task")
    first["reasoning_content"] = "I need the task instruction before acting."
    client = FakeClient([first, tool_call("complete_task", '{"answer": "done"}')])

    run_teacher_rollout(adapter=FakeAdapter(success=True), client=client, task_id="task_a")

    historical_assistant = client.requests[1][-2]
    assert historical_assistant["reasoning_content"] == first["reasoning_content"]


def test_rollout_warns_on_immediate_duplicate_tool_call() -> None:
    client = FakeClient(
        [
            tool_call("inspect_task"),
            tool_call("inspect_task"),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    adapter = FakeAdapter(success=True)
    run_teacher_rollout(adapter=adapter, client=client, task_id="task_a")

    correction = json.loads(client.requests[2][-1]["content"])
    assert correction["executed"] is False
    assert correction["observation"]["harness_correction"] is True
    assert [call["function"]["name"] for call in adapter.calls] == [
        "inspect_task",
        "complete_task",
    ]


def test_collector_writes_raw_and_verifier_filtered_partitions() -> None:
    outcomes = iter([True, False])

    def adapter_factory() -> FakeAdapter:
        return FakeAdapter(success=next(outcomes))

    client = FakeClient(
        [
            tool_call("complete_task", '{"answer": "done"}'),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "teacher_run"
        summary = collect_teacher_rollouts(
            client=client,
            adapter_factory=adapter_factory,
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a", "task_b"],
            task_id_loader=lambda split: ["task_a", "task_b"],
        )

        raw_rows = [
            json.loads(line)
            for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        accepted_rows = (output_dir / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
        rejected_rows = (output_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(raw_rows) == 2
        assert len(accepted_rows) == 1
        assert len(rejected_rows) == 1
        assert (output_dir / "duplicates.jsonl").is_file()
        assert len((output_dir / "curated_sft.jsonl").read_text().splitlines()) == 1
        assert [row["collection_status"] for row in raw_rows] == ["accepted", "rejected"]
        assert summary["accepted"] == 1
        assert summary["rejected"] == 1
        assert "must-not-leak" not in (output_dir / "raw.jsonl").read_text(encoding="utf-8")


def test_collector_rejects_non_train_collection() -> None:
    with pytest.raises(ValueError, match="train split"):
        collect_teacher_rollouts(
            client=FakeClient([]),
            adapter_factory=lambda: FakeAdapter(success=False),
            output_dir=Path(".analysis") / "invalid",
            model_name="fake-teacher",
            task_ids=["task_a"],
            split="dev",
            task_id_loader=lambda split: ["task_a"],
        )


def _curation_record(
    rollout_id: str,
    steps: list[dict[str, Any]],
    *,
    status: str = "accepted",
) -> dict[str, Any]:
    return {
        "task_id": "task_a",
        "rollout_id": rollout_id,
        "attempt_index": int(rollout_id[-1]),
        "collection_status": status,
        "trajectory": {"steps": steps},
    }


def test_same_plan_successes_deduplicate_and_best_process_record_wins() -> None:
    errored_steps = [
        {
            "tool_name": "query_state",
            "arguments": {"code": "print(apis.spotify.search_songs(query='x'))"},
            "error": "temporary failure",
            "state_change": False,
        },
        {
            "tool_name": "complete_task",
            "arguments": {"answer": "x"},
            "error": None,
            "state_change": True,
        },
    ]
    clean_steps = deepcopy(errored_steps)
    clean_steps[0]["error"] = None
    curated = curate_teacher_rollouts(
        [
            _curation_record("rollout-1", errored_steps),
            _curation_record("rollout-2", clean_steps),
        ]
    )
    first, second = curated
    assert first["api_plan_hash"] == second["api_plan_hash"]
    assert first["selected_for_sft"] is False
    assert first["duplicate_of"] == "rollout-2"
    assert second["selected_for_sft"] is True
    assert second["process_score"] > first["process_score"]


def test_genuinely_different_api_plans_are_both_curated() -> None:
    search_plan = [
        {
            "tool_name": "query_state",
            "arguments": {"code": "print(apis.spotify.search_songs(query='x'))"},
            "error": None,
            "state_change": False,
        }
    ]
    playlist_plan = [
        {
            "tool_name": "query_state",
            "arguments": {"code": "print(apis.spotify.show_playlist(playlist_id=1))"},
            "error": None,
            "state_change": False,
        }
    ]
    curated = curate_teacher_rollouts(
        [
            _curation_record("rollout-1", search_plan),
            _curation_record("rollout-2", playlist_plan),
        ],
        max_curated_per_task=2,
    )
    assert curated[0]["api_plan_hash"] != curated[1]["api_plan_hash"]
    assert all(record["selected_for_sft"] for record in curated)
    assert all(record["duplicate_of"] is None for record in curated)


def test_unique_success_with_execution_error_is_not_selected_for_sft() -> None:
    record = _curation_record(
        "rollout-1",
        [
            {
                "tool_name": "query_state",
                "arguments": {"code": "print(apis.spotify.search_songs(query='x'))"},
                "error": "temporary failure",
                "state_change": False,
            }
        ],
    )
    curated = curate_teacher_rollouts([record])
    assert curated[0]["collection_status"] == "accepted"
    assert curated[0]["selected_for_sft"] is False
    assert curated[0]["curation_exclusion_reason"] == "execution_errors"

    allowed = curate_teacher_rollouts([record], require_clean_for_sft=False)
    assert allowed[0]["selected_for_sft"] is True


def test_api_extraction_uses_ast_and_preserves_call_order() -> None:
    code = """
token = apis.spotify.login(username='u', password=password)
rows = apis.spotify.search_songs(access_token=token, query='x')
"""
    assert extract_planned_api_calls(code) == [
        "apis.spotify.login",
        "apis.spotify.search_songs",
    ]


def test_collector_attempt_accounting_and_duplicate_artifact() -> None:
    client = FakeClient(
        [
            tool_call("complete_task", '{"answer": "done"}'),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "multi_attempt"
        summary = collect_teacher_rollouts(
            client=client,
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=2,
            max_curated_per_task=2,
            task_id_loader=lambda split: ["task_a"],
        )
        assert summary["num_tasks"] == 1
        assert summary["num_rollouts"] == 2
        assert summary["accepted"] == 2
        assert summary["curated_sft"] == 1
        assert summary["duplicates"] == 1
        assert len((output_dir / "raw.jsonl").read_text().splitlines()) == 2
        assert len((output_dir / "accepted.jsonl").read_text().splitlines()) == 2
        assert len((output_dir / "duplicates.jsonl").read_text().splitlines()) == 1
        assert len((output_dir / "curated_sft.jsonl").read_text().splitlines()) == 1


def test_collector_resumes_only_missing_attempts_after_interrupt() -> None:
    calls = 0

    def interrupting_factory() -> FakeAdapter:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption")
        return FakeAdapter(success=True)

    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "resumable"
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            collect_teacher_rollouts(
                client=FakeClient([tool_call("complete_task")]),
                adapter_factory=interrupting_factory,
                output_dir=output_dir,
                model_name="fake-teacher",
                task_ids=["task_a"],
                attempts_per_task=2,
                task_id_loader=lambda split: ["task_a"],
            )
        partial_rows = (output_dir / "raw.jsonl").read_text().splitlines()
        assert len(partial_rows) == 1

        resumed_factory_calls = 0

        def resumed_factory() -> FakeAdapter:
            nonlocal resumed_factory_calls
            resumed_factory_calls += 1
            return FakeAdapter(success=True)

        summary = collect_teacher_rollouts(
            client=FakeClient([tool_call("complete_task")]),
            adapter_factory=resumed_factory,
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=2,
            resume=True,
            task_id_loader=lambda split: ["task_a"],
        )
        assert resumed_factory_calls == 1
        assert summary["num_rollouts"] == 2
        assert summary["accepted"] == 2
        raw_rows = [
            json.loads(line) for line in (output_dir / "raw.jsonl").read_text().splitlines()
        ]
        assert [row["attempt_index"] for row in raw_rows] == [1, 2]
        assert len({row["rollout_id"] for row in raw_rows}) == 2


def test_resume_rejects_changed_collection_contract() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "config_mismatch"
        collect_teacher_rollouts(
            client=FakeClient([tool_call("complete_task")]),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=1,
            task_id_loader=lambda split: ["task_a"],
        )
        with pytest.raises(ValueError, match="resume configuration"):
            collect_teacher_rollouts(
                client=FakeClient([]),
                adapter_factory=lambda: FakeAdapter(success=True),
                output_dir=output_dir,
                model_name="fake-teacher",
                task_ids=["task_a"],
                attempts_per_task=2,
                resume=True,
                task_id_loader=lambda split: ["task_a"],
            )


def test_redaction_keeps_variable_references_but_removes_literals() -> None:
    payload = {
        "password": "raw-secret",
        "code": "password=spotify_password\naccess_token='raw-token'",
        "observation": "password: raw-secret\nBearer raw-token",
    }
    clean = redact_sensitive_data(payload)
    assert clean["password"] == "<REDACTED>"
    assert "password=spotify_password" in clean["code"]
    assert "raw-token" not in clean["code"]
    assert "raw-secret" not in clean["observation"]


def test_assistant_message_is_sanitized_before_the_next_model_turn() -> None:
    client = FakeClient(
        [
            tool_call(
                "query_state",
                json.dumps({"code": "access_token='raw-token'\nprint(access_token)"}),
            ),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    adapter = FakeAdapter(success=True)

    run_teacher_rollout(adapter=adapter, client=client, task_id="task_a")

    assistant_message = client.requests[1][-2]
    assert assistant_message["role"] == "assistant"
    assert "raw-token" not in json.dumps(assistant_message)
    assert "raw-token" in adapter.calls[0]["function"]["arguments"]


def test_sanitize_assistant_message_preserves_code_variable_references() -> None:
    message = tool_call(
        "query_state",
        json.dumps({"code": "password=spotify_password\naccess_token='raw-token'"}),
    )
    clean = sanitize_assistant_message(message)
    arguments = json.loads(clean["tool_calls"][0]["function"]["arguments"])
    assert "password=spotify_password" in arguments["code"]
    assert "raw-token" not in arguments["code"]


def test_provider_error_detail_extracts_a_message() -> None:
    response = HTTPError(
        "https://provider/v1/chat/completions",
        400,
        "Bad Request",
        None,
        __import__("io").BytesIO(b'{"error": {"message": "unsupported model"}}'),
    )
    assert _safe_response_detail(response) == "unsupported model"


def test_normalize_assistant_message_preserves_reasoning_content() -> None:
    message = normalize_assistant_message(
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need to inspect the task first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "inspect_task", "arguments": "{}"},
                }
            ],
        }
    )
    assert message["reasoning_content"] == "Need to inspect the task first."


def test_normalize_assistant_message_rejects_invalid_reasoning_content() -> None:
    with pytest.raises(TeacherModelError, match="reasoning_content"):
        normalize_assistant_message({"role": "assistant", "content": None, "reasoning_content": {}})


def test_teacher_client_retries_transient_503(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": []}}]}'

    attempts = 0

    def fake_urlopen(request: Any, timeout: float) -> Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HTTPError(
                request.full_url,
                503,
                "Service unavailable",
                None,
                __import__("io").BytesIO(b'{"error": {"message": "try again"}}'),
            )
        return Response()

    monkeypatch.setattr("appworld_agentic_rl.teacher_client.urlopen", fake_urlopen)
    monkeypatch.setattr("appworld_agentic_rl.teacher_client.time.sleep", lambda _: None)
    client = OpenAICompatibleTeacherClient(
        model="fake",
        base_url="https://provider/v1",
        api_key="test-key",
        max_retries=2,
        retry_backoff_seconds=0,
    )
    assert client.generate([], []) == {"role": "assistant", "content": None}
    assert attempts == 3


def test_teacher_client_raises_quota_signal_after_final_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> Any:
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            None,
            __import__("io").BytesIO(b'{"error": {"message": "quota exhausted"}}'),
        )

    monkeypatch.setattr("appworld_agentic_rl.teacher_client.urlopen", fake_urlopen)
    client = OpenAICompatibleTeacherClient(
        model="fake",
        base_url="https://provider/v1",
        api_key="test-key",
        max_retries=0,
    )

    with pytest.raises(TeacherQuotaError, match="HTTP 429: quota exhausted"):
        client.generate([], [])


def test_teacher_client_parallel_parameter_is_provider_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    captured_headers: dict[str, str | None] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": []}}]}'

    def fake_urlopen(request: Any, timeout: float) -> Response:
        captured.update(json.loads(request.data.decode("utf-8")))
        captured_headers.update(
            {
                "accept": request.get_header("Accept"),
                "user_agent": request.get_header("User-agent"),
            }
        )
        return Response()

    monkeypatch.setattr("appworld_agentic_rl.teacher_client.urlopen", fake_urlopen)
    client = OpenAICompatibleTeacherClient(
        model="fake",
        base_url="https://provider/v1",
        api_key="test-key",
        max_retries=0,
    )

    client.generate([], [])

    assert "parallel_tool_calls" not in captured
    assert "chat_template_kwargs" not in captured
    captured.clear()
    opted_in_client = OpenAICompatibleTeacherClient(
        model="fake",
        base_url="https://provider/v1",
        api_key="test-key",
        max_retries=0,
        disable_provider_parallel_tool_calls=True,
    )
    opted_in_client.generate([], [])
    assert captured["parallel_tool_calls"] is False
    captured.clear()
    no_thinking_client = OpenAICompatibleTeacherClient(
        model="fake",
        base_url="https://provider/v1",
        api_key="test-key",
        max_retries=0,
        disable_thinking=True,
    )
    no_thinking_client.generate([], [])
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured_headers == {
        "accept": "application/json",
        "user_agent": "appworld-agentic-rl/0.1",
    }


def test_shadow_mode_executes_parallel_calls_serially() -> None:
    response = tool_call("inspect_task")
    response["tool_calls"].append(
        {
            "id": "call_query",
            "type": "function",
            "function": {"name": "query_state", "arguments": "{}"},
        }
    )
    client = FakeClient([response, tool_call("complete_task", '{"answer": "done"}')])
    adapter = FakeAdapter(success=True)

    record = run_teacher_rollout(
        adapter=adapter,
        client=client,
        task_id="task_a",
        harness_config=TeacherHarnessConfig(bc_guard_mode="shadow"),
    )

    assert [call["function"]["name"] for call in adapter.calls] == [
        "inspect_task",
        "query_state",
        "complete_task",
    ]
    assert record["rollout"]["action_attempts"] == 3
    assert record["rollout"]["environment_steps"] == 3
    assert record["rollout"]["dropped_tool_calls"] == []
    parallel_events = [
        event
        for event in record["harness"]["guard_events"]
        if event["code"] == "parallel_tool_calls"
    ]
    assert len(parallel_events) == 1
    assert parallel_events[0]["enforced"] is False


def test_enforce_mode_blocks_appworld_guard_violation() -> None:
    class GuardedAdapter(FakeAdapter):
        def validate_tool_call(self, call: dict[str, Any]) -> list[dict[str, Any]]:
            if call["function"]["name"] != "query_state":
                return []
            return [
                {
                    "code": "api_not_found",
                    "message": "API calendar.guessed does not exist.",
                    "component": "appworld_guard",
                    "always_enforced": False,
                }
            ]

    client = FakeClient(
        [
            tool_call("query_state", '{"code": "print(apis.calendar.guessed())"}'),
            tool_call("complete_task", '{"answer": "done"}'),
        ]
    )
    adapter = GuardedAdapter(success=True)
    record = run_teacher_rollout(adapter=adapter, client=client, task_id="task_a")

    assert [call["function"]["name"] for call in adapter.calls] == ["complete_task"]
    assert record["rollout"]["action_attempts"] == 2
    assert record["rollout"]["environment_steps"] == 1
    assert record["rollout"]["invalid_actions"] == 1


def test_context_budget_drops_only_complete_oldest_groups() -> None:
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    groups = [
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "x" * 80},
        ],
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "b"}]},
            {"role": "tool", "tool_call_id": "b", "content": "latest"},
        ],
    ]
    messages, dropped, _, exhausted = build_bounded_context(
        base_messages=base,
        complete_groups=groups,
        max_chars=230,
    )
    assert dropped == 1
    assert exhausted is False
    assert all(message.get("tool_call_id") != "a" for message in messages)
    assert any(message.get("tool_call_id") == "b" for message in messages)


def test_memory_context_replaces_old_history_after_trigger() -> None:
    base = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}]
    groups = [
        [{"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
         {"role": "tool", "tool_call_id": "a", "content": "old" * 100}],
        [{"role": "assistant", "content": None, "tool_calls": [{"id": "b"}]},
         {"role": "tool", "tool_call_id": "b", "content": "latest"}],
    ]
    memory = MemoryState(variable_bindings={"venmo_token": {"available": True, "semantic": "authentication", "kind": "derived_field", "source_event": 1}})
    messages, compacted, _, exhausted = build_memory_context(
        base_messages=base,
        complete_groups=groups,
        memory=memory,
        max_chars=5000,
        trigger_chars=100,
        target_chars=500,
        preserve_recent_groups=1,
    )
    assert compacted is True
    assert exhausted is False
    assert "APPWORLD MEMORY" in messages[0]["content"]
    assert "venmo_token" in messages[0]["content"]
    assert all(message.get("tool_call_id") != "a" for message in messages)
    assert any(message.get("tool_call_id") == "b" for message in messages)


def test_memory_context_can_use_exact_token_counter() -> None:
    base = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}]
    groups = [
        [{"role": "assistant", "content": "old"}, {"role": "tool", "content": "x" * 80}],
        [{"role": "assistant", "content": "new"}, {"role": "tool", "content": "latest"}],
    ]

    def token_counter(messages: list[dict[str, Any]]) -> int:
        return sum(len(str(message.get("content") or "")) for message in messages)

    messages, compacted, token_count, exhausted = build_memory_token_context(
        base_messages=base,
        complete_groups=groups,
        memory=MemoryState(),
        token_counter=token_counter,
        max_tokens=1000,
        trigger_tokens=40,
        target_tokens=200,
        preserve_recent_groups=1,
    )

    assert compacted is True
    assert exhausted is False
    assert token_count == token_counter(messages)
    assert "APPWORLD MEMORY" in messages[0]["content"]
    assert all(message.get("content") != "old" for message in messages)


def test_rollout_records_exact_token_budget_telemetry() -> None:
    record = run_teacher_rollout(
        adapter=FakeAdapter(success=True),
        client=FakeClient([tool_call("complete_task", '{"answer": "done"}')]),
        task_id="task_a",
        harness_config=TeacherHarnessConfig(
            context_budget_mode="tokens",
            max_context_tokens=100,
            context_compaction_trigger_tokens=50,
            context_compaction_target_tokens=25,
        ),
        context_token_counter=lambda messages: 10,
    )

    event = record["harness"]["context_events"][0]
    assert event["budget_mode"] == "tokens"
    assert event["input_tokens"] == 10
    assert event["token_count_is_exact"] is True


def test_token_budget_requires_a_token_counter() -> None:
    with pytest.raises(ValueError, match="context_token_counter"):
        run_teacher_rollout(
            adapter=FakeAdapter(success=True),
            client=FakeClient([]),
            task_id="task_a",
            harness_config=TeacherHarnessConfig(context_budget_mode="tokens"),
        )


def test_provider_failure_is_infrastructure_invalid_and_not_accepted() -> None:
    class FailingClient:
        def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            raise TeacherProviderError("temporary outage")

    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "provider_failure"
        summary = collect_teacher_rollouts(
            client=FailingClient(),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            task_id_loader=lambda split: ["task_a"],
        )
        row = json.loads((output_dir / "raw.jsonl").read_text().splitlines()[0])
        assert row["infrastructure_invalid"] is True
        assert row["collection_status"] == "rejected"
        assert summary["infrastructure_invalid_count"] == 1


def test_quota_circuit_breaker_stops_without_consuming_attempt_or_rewriting_raw() -> None:
    class QuotaAfterOneClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
        ) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return tool_call("complete_task")
            raise TeacherQuotaError("teacher endpoint returned HTTP 429: quota exhausted")

    class QuotaClient:
        def generate(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
        ) -> dict[str, Any]:
            raise TeacherQuotaError("teacher endpoint returned HTTP 429: quota exhausted")

    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "quota_breaker"
        first_summary = collect_teacher_rollouts(
            client=QuotaAfterOneClient(),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=3,
            task_id_loader=lambda split: ["task_a"],
        )
        raw_before = (output_dir / "raw.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw_before.decode().splitlines()]
        assert [row["attempt_index"] for row in rows] == [1]
        assert first_summary["stopped_early"] is True
        assert first_summary["stop_reason"] == "provider_quota_exhausted"
        assert first_summary["pending_attempt_index"] == 2

        second_summary = collect_teacher_rollouts(
            client=QuotaClient(),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=3,
            resume=True,
            task_id_loader=lambda split: ["task_a"],
        )
        assert (output_dir / "raw.jsonl").read_bytes() == raw_before
        assert second_summary["pending_attempt_index"] == 2

        resumed = collect_teacher_rollouts(
            client=FakeClient([tool_call("complete_task"), tool_call("complete_task")]),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            attempts_per_task=3,
            resume=True,
            task_id_loader=lambda split: ["task_a"],
        )
        final_rows = [
            json.loads(line)
            for line in (output_dir / "raw.jsonl").read_text().splitlines()
        ]
        assert [row["attempt_index"] for row in final_rows] == [1, 2, 3]
        assert resumed["stopped_early"] is False


def test_evaluator_pass_without_terminal_tool_is_not_accepted() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        output_dir = Path(temporary_directory) / "missing_terminal"
        summary = collect_teacher_rollouts(
            client=FakeClient([{"role": "assistant", "content": "done"}]),
            adapter_factory=lambda: FakeAdapter(success=True),
            output_dir=output_dir,
            model_name="fake-teacher",
            task_ids=["task_a"],
            task_id_loader=lambda split: ["task_a"],
            harness_config=TeacherHarnessConfig(bc_guard_mode="shadow"),
        )
        row = json.loads((output_dir / "raw.jsonl").read_text().splitlines()[0])
        assert row["rollout"]["end_reason"] == "model_stopped_without_tool"
        assert row["infrastructure_invalid"] is False
        assert row["collection_status"] == "rejected"
        assert summary["accepted"] == 0
