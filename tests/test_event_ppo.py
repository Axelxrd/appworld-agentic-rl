from types import SimpleNamespace

import pytest


def _output(uid: str, group: str, score: float, records: list[dict], events: list[dict]):
    return SimpleNamespace(
        reward_score=score,
        extra_fields={
            "appworld_rollout_uid": uid,
            "appworld_group_uid": group,
            "appworld_event_records": records,
            "appworld_reward": {
                "process_score": sum(float(item["delta"]) for item in events),
                "events": events,
            },
        },
    )


def test_event_advantages_keep_events_before_and_after_compaction():
    from appworld_agentic_rl.event_ppo import _event_advantages

    records = [
        {"step_index": 0, "context_ids": [1, 2, 3], "action_ids": [4, 5], "action_log_probs": [-.1, -.2]},
        {"step_index": 1, "context_ids": [9, 10], "action_ids": [11], "action_log_probs": [-.3]},
    ]
    outputs = [
        _output("a", "task", 1.0, records, [{"step_index": 0, "delta": .1}, {"step_index": 1, "delta": 0.0}]),
        _output("b", "task", -1.0, records, [{"step_index": 0, "delta": 0.0}, {"step_index": 1, "delta": 0.0}]),
    ]
    values = _event_advantages(outputs, process_reward_lambda=0.3)
    assert set(values) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert values[(0, 0)] != 0.0
    assert values[(0, 1)] != 0.0


def test_event_advantages_are_group_relative_and_singleton_is_zero():
    from appworld_agentic_rl.event_ppo import _event_advantages

    record = [{"step_index": 0, "context_ids": [1], "action_ids": [2], "action_log_probs": [-.1]}]
    outputs = [_output("a", "singleton", 1.0, record, [{"step_index": 0, "delta": 0.0}])]
    values = _event_advantages(outputs, process_reward_lambda=0.3)
    assert values[(0, 0)] == 0.0


def test_event_advantages_fallback_to_process_variance_for_homogeneous_outcomes():
    from appworld_agentic_rl.event_ppo import _event_advantages

    record = [{"step_index": 0, "context_ids": [1], "action_ids": [2], "action_log_probs": [-.1]}]
    outputs = [
        _output("a", "task", -0.7, record, [{"step_index": 0, "delta": .1}]),
        _output("b", "task", -0.7, record, [{"step_index": 0, "delta": -.1}]),
    ]
    values = _event_advantages(outputs, process_reward_lambda=0.3)
    assert values[(0, 0)] > 0
    assert values[(1, 0)] < 0


def test_event_dataproto_rejects_missing_action_logprobs():
    pytest.importorskip("torch")
    pytest.importorskip("verl")
    from appworld_agentic_rl.event_ppo import build_event_dataproto

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                prompt_length=16,
                response_length=16,
                agent=SimpleNamespace(process_reward_lambda=0.3),
            )
        ),
        algorithm=SimpleNamespace(turn_discount={"alpha": 1.05}),
    )
    output = _output(
        "a",
        "task",
        1.0,
        [{"step_index": 0, "context_ids": [1], "action_ids": [2], "action_log_probs": []}],
        [{"step_index": 0, "delta": 0.0}],
    )
    with pytest.raises(ValueError, match="one rollout log-prob"):
        build_event_dataproto([output], config)


def test_event_replay_falls_back_to_process_variance_for_homogeneous_group():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.event_ppo import event_replay_advantage

    rewards = torch.tensor([
        [-0.7, 0.2, 0.0, 0.0],
        [-0.7, -0.2, 0.0, 0.0],
    ])
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ])
    config = SimpleNamespace(turn_discount={"alpha": 1.05})
    advantages, _ = event_replay_advantage(
        rewards,
        mask,
        index=["group|traj-a|0", "group|traj-b|0"],
        config=config,
    )
    assert advantages[0, 0] > 0
    assert advantages[1, 0] < 0


def test_event_replay_returns_zero_when_terminal_and_process_are_homogeneous():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.event_ppo import event_replay_advantage

    rewards = torch.tensor([[-0.7, 0.0], [-0.7, 0.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = event_replay_advantage(
        rewards,
        mask,
        index=["group|traj-a|0", "group|traj-b|0"],
        config=SimpleNamespace(turn_discount={"alpha": 1.05}),
    )
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_shared_context_policy_is_recorded_by_harness():
    from appworld_agentic_rl.context_policy import (
        DEFAULT_CONTEXT_POLICY,
        TEACHER_CONTEXT_TRIGGER_CHARS,
    )
    from appworld_agentic_rl.harness import TeacherHarnessConfig

    config = TeacherHarnessConfig()
    recorded = config.to_dict()["context_policy"]
    assert recorded["version"] == "appworld-structured-memory-token-v1"
    assert recorded["trigger_tokens"] == DEFAULT_CONTEXT_POLICY.trigger_tokens
    assert recorded["target_tokens"] == DEFAULT_CONTEXT_POLICY.target_tokens
    assert recorded["evaluation_budget_mode"] == "chars"
    assert recorded["effective_trigger_chars"] == TEACHER_CONTEXT_TRIGGER_CHARS
