import pytest


def test_event_aligned_process_credit_stays_in_its_action_span():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.verl_compat import _event_aligned_credit

    # Two action spans separated by tool-observation tokens.  There is no
    # terminal marker here; the final active token is zero, so process credit
    # can be checked independently of outcome credit.
    rewards = torch.tensor(
        [[0.2, 0.0, 0.0, -0.2, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0]], dtype=torch.bool)
    advantages, _ = _event_aligned_credit(rewards, mask, ["task", "task"])

    assert torch.allclose(advantages[0, :2], torch.tensor([0.1, 0.1]))
    assert torch.allclose(advantages[0, 3:5], torch.tensor([-0.1, -0.1]))
    assert torch.all(advantages[:, [2, 5]] == 0)


def test_event_aligned_terminal_outcome_reaches_all_events_with_early_discount():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.verl_compat import _event_aligned_credit

    rewards = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0]], dtype=torch.bool)
    advantages, _ = _event_aligned_credit(rewards, mask, [0, 0])

    # Both events receive terminal credit, with the earlier event weighted
    # more strongly when alpha > 1.
    assert advantages[0, 0] > advantages[0, 3] > 0
    assert torch.allclose(advantages[0, :2], advantages[0, 1].expand(2))
    assert torch.allclose(advantages[0, 3:5], advantages[0, 3].expand(2))
    assert torch.all(advantages[0, [2, 5]] == 0)
    assert torch.allclose(advantages[0], -advantages[1], atol=1e-5)


def test_event_aligned_handles_singletons_and_empty_responses_without_nan():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.verl_compat import _event_aligned_credit

    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    advantages, returns = _event_aligned_credit(rewards, mask, [0, 1])

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    assert torch.all(advantages[1] == 0)
    # A singleton group has no relative signal and must not update the policy.
    assert torch.all(advantages[0] == 0)


def test_position_credit_estimator_matches_reference_shape():
    torch = pytest.importorskip("torch")
    from appworld_agentic_rl.verl_compat import install_appworld_advantage_estimators

    pytest.importorskip("verl")
    install_appworld_advantage_estimators()
    from verl.trainer.ppo import core_algos

    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.ones_like(rewards)
    fn = core_algos.ADV_ESTIMATOR_REGISTRY["grpo_position_credit"]
    values, returns = fn(rewards, mask, [0, 0, 1, 1])
    assert values.shape == rewards.shape
    assert returns.shape == rewards.shape
