from appworld_agentic_rl.event_credit import compute_event_credit, group_normalize
from appworld_agentic_rl.online_rl import build_online_group
from appworld_agentic_rl.process_reward import score_trajectory


def _step(tool, arguments=None, observation=None, error=None):
    return {
        "tool_name": tool,
        "arguments": arguments or {},
        "observation": observation or {"ok": True},
        "error": error,
    }


def test_prm_rewards_discovery_read_write_and_completion():
    result = score_trajectory(
        {
            "steps": [
                _step("inspect_task"),
                _step("search_api_docs", {"query": "calendar events"}),
                _step("query_state", {"code": "x = apis.calendar.list_events()"}),
                _step("run_write_code", {"code": "apis.calendar.create_event(title='x')"}),
                _step("complete_task"),
            ]
        },
        terminal_reward=1.0,
    )
    assert result.process_score > 0
    assert result.metrics["read_before_write"] == 1
    assert any("timely_completion" in event.reasons for event in result.events)


def test_prm_penalizes_repeated_and_unsafe_write():
    result = score_trajectory(
        {
            "steps": [
                _step("run_write_code", {"code": "apis.todoist.add_task(content='x')"}),
                _step("run_write_code", {"code": "apis.todoist.add_task(content='x')"}),
            ]
        }
    )
    assert result.process_score < 0
    assert result.metrics["unsafe_write"] == 2
    assert result.metrics["repeated_action"] == 1


def test_prm_does_not_reward_endless_documentation():
    result = score_trajectory(
        {"steps": [_step("search_api_docs", {"query": str(i)}) for i in range(8)]}
    )
    assert result.process_score < 0
    assert result.metrics["discovery"] == 1
    assert result.metrics["unfinished"] == 1


def test_prm_penalizes_duplicate_search_result_sets_not_different_queries():
    observation = {
        "results": [
            {"app": "calendar", "api": "list_events", "description": "List events."},
        ]
    }
    result = score_trajectory(
        {
            "steps": [
                _step("search_api_docs", {"query": "events"}, observation),
                _step("search_api_docs", {"query": "meetings"}, observation),
                _step("finish_without_completion"),
            ]
        }
    )
    assert result.metrics["repeated_search_result_set"] == 1
    assert any("duplicate_search_result_set" in event.reasons for event in result.events)


def test_prm_does_not_reward_generic_read_observation():
    result = score_trajectory(
        {"steps": [_step("query_state", {"code": "apis.calendar.list_events()"})]}
    )
    assert result.metrics["entity_grounding"] == 0


def test_prm_requires_parameter_provenance_for_grounded_write():
    result = score_trajectory(
        {
            "steps": [
                _step("query_state", {"code": "apis.calendar.list_events()"}, {"items": [{"id": "1"}]}),
                {
                    **_step(
                        "run_write_code",
                        {"code": "apis.calendar.create_event(title=guessed_title)"},
                        {"ok": True},
                    ),
                    "metadata": {
                        "write_provenance": [
                            {"all_required_verified": False, "app": "calendar", "api": "create_event"}
                        ]
                    },
                },
            ]
        }
    )
    assert result.metrics["grounded_write_events"] == 0
    assert result.metrics["unverified_write_parameters"] == 1


def test_prm_accepts_verified_parameter_provenance():
    result = score_trajectory(
        {
            "steps": [
                _step("query_state", {"code": "apis.calendar.list_events()"}, {"items": [{"id": "1"}]}),
                {
                    **_step("run_write_code", {"code": "apis.calendar.create_event(title=event_title)"}),
                    "metadata": {
                        "write_provenance": [
                            {"all_required_verified": True, "app": "calendar", "api": "create_event"}
                        ]
                    },
                },
            ]
        }
    )
    assert result.metrics["grounded_write_events"] == 1


def test_lata_group_and_local_event_alignment():
    normalized = group_normalize([0.0, 1.0])
    assert normalized[0] < 0 < normalized[1]
    result = compute_event_credit(
        terminal_reward=1.0,
        process_deltas=[0.2, -0.1],
        token_counts=[2, 3],
        normalized_advantage=1.0,
    )
    assert len(result.token_advantages) == 5
    assert result.turn_advantages[0] > result.turn_advantages[1]
    assert result.token_advantages[0] == result.turn_advantages[0]


def test_online_group_excludes_infrastructure_invalid_rows(monkeypatch):
    import appworld_agentic_rl.online_rl as online_rl

    rows = iter(
        [
            {
                "terminal_reward": 1.0,
                "infrastructure_invalid": False,
                "trajectory": {"steps": [_step("complete_task")]},
            },
            {
                "terminal_reward": 1.0,
                "infrastructure_invalid": True,
                "trajectory": {"steps": [_step("complete_task")]},
            },
        ]
    )

    def fake_rollout(**_kwargs):
        return next(rows)

    class Adapter:
        def close(self):
            pass

    monkeypatch.setattr(online_rl, "run_teacher_rollout", fake_rollout)
    group = build_online_group(
        task_id="task",
        client=object(),
        adapter_factory=Adapter,
        group_size=2,
    )
    assert group["eligible_count"] == 1
    assert group["excluded_count"] == 1
    assert group["trainable"] is True
    assert group["records"][0]["group_advantage"] == 0.0
    assert group["records"][0]["training_eligible"] is True
    assert group["records"][1]["group_advantage"] is None
    assert group["records"][1]["exclusion_reason"] == "infrastructure_invalid"


def test_online_group_with_only_invalid_rows_is_not_trainable(monkeypatch):
    import appworld_agentic_rl.online_rl as online_rl

    monkeypatch.setattr(
        online_rl,
        "run_teacher_rollout",
        lambda **_kwargs: {
            "terminal_reward": -1.0,
            "infrastructure_invalid": True,
            "trajectory": {"steps": []},
        },
    )

    class Adapter:
        def close(self):
            pass

    group = build_online_group(
        task_id="task",
        client=object(),
        adapter_factory=Adapter,
        group_size=2,
    )
    assert group["eligible_count"] == 0
    assert group["trainable"] is False
    assert all(row["group_advantage"] is None for row in group["records"])
