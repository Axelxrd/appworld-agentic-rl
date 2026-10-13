"""veRL advantage estimators used by the AppWorld online RL pipeline.

The AppWorld agent loop puts process reward on the first token of each action
span and terminal reward on the last active token. ``grpo_event_credit`` uses
the response mask to recover those spans without transferring Python metadata.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch


def _active_event_spans(response_mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return contiguous active (assistant action) spans for one response."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    values = response_mask.detach().to(device="cpu").tolist()
    for position, active in enumerate(values):
        if int(active) != 0 and start is None:
            start = position
        elif int(active) == 0 and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(values)))
    return spans


def _group_normalize(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    epsilon: float,
    normalize_std: bool,
) -> torch.Tensor:
    """Compute stable GRPO group-relative trajectory scores."""

    valid = response_mask.to(dtype=torch.bool).any(dim=-1)
    grouped: dict[Any, list[int]] = defaultdict(list)
    for row, group_id in enumerate(index):
        if bool(valid[row]):
            grouped[group_id].append(row)

    normalized = torch.zeros_like(scores, dtype=torch.float32)
    with torch.no_grad():
        for rows in grouped.values():
            values = scores[rows].to(dtype=torch.float32)
            mean = values.mean()
            centered = values - mean
            if len(rows) == 1:
                normalized[rows] = 0.0
            elif normalize_std:
                std = values.std(unbiased=False)
                normalized[rows] = 0.0 if float(std) <= epsilon else centered / (std + epsilon)
            else:
                normalized[rows] = centered
    return normalized


def _turn_discount(alpha: float, event_count: int, *, device: torch.device) -> torch.Tensor:
    """Return normalized early-to-late event weights."""

    if alpha <= 0:
        raise ValueError("turn_discount.alpha must be positive")
    exponents = torch.arange(event_count - 1, -1, -1, dtype=torch.float32, device=device)
    weights = torch.exp(exponents * math.log(alpha))
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def _event_aligned_credit(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute lightweight process-reward event-aligned advantages.

    Process reward remains local to its action span. The terminal marker is
    removed from the final span and represented by the trajectory-level GRPO
    outcome advantage. Event outcome credit is length-normalized inside each
    span and uses the existing sqrt(active-token) length-aware scale.
    """

    if token_level_rewards.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("token_level_rewards and response_mask must be rank-2 tensors")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must have the same shape")

    mask = response_mask.to(dtype=torch.float32)
    scores = token_level_rewards.to(dtype=torch.float32).sum(dim=-1)
    outcome = _group_normalize(scores, response_mask, index, epsilon, norm_adv_by_std_in_grpo)

    alpha = 1.05
    if config is not None and getattr(config, "turn_discount", None) is not None:
        turn_discount = getattr(config, "turn_discount")
        alpha = float(
            turn_discount.get("alpha", alpha)
            if hasattr(turn_discount, "get")
            else getattr(turn_discount, "alpha", alpha)
        )

    advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
    for row in range(scores.shape[0]):
        spans = _active_event_spans(response_mask[row])
        if not spans:
            continue
        event_weights = _turn_discount(alpha, len(spans), device=scores.device)
        active_length = int(response_mask[row].to(dtype=torch.bool).sum().item())
        outcome_total = outcome[row] * math.sqrt(max(1, active_length))
        row_rewards = token_level_rewards[row].to(dtype=torch.float32)
        terminal_marker = row_rewards[spans[-1][1] - 1]
        for event, (start, end) in enumerate(spans):
            length = max(1, end - start)
            local_process = row_rewards[start:end].sum()
            if event == len(spans) - 1:
                # A one-token final span necessarily combines process and
                # terminal markers; that combined value is treated as terminal.
                local_process = local_process - terminal_marker
            event_outcome_total = outcome_total * event_weights[event]
            advantages[row, start:end] = (event_outcome_total + local_process) / float(length)

    advantages = advantages * mask
    return advantages, advantages.clone()


def _legacy_grpo_position_credit(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the previous token-position estimator for explicit ablations."""

    scores = token_level_rewards.to(dtype=torch.float32).sum(dim=-1)
    normalized = _group_normalize(scores, response_mask, index, epsilon, norm_adv_by_std_in_grpo)
    mask = response_mask.to(dtype=torch.float32)
    lengths = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    alpha = 1.05
    if config is not None and getattr(config, "turn_discount", None) is not None:
        td = getattr(config, "turn_discount")
        alpha = float(td.get("alpha", alpha) if hasattr(td, "get") else getattr(td, "alpha", alpha))
    positions = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32).unsqueeze(0)
    exponents = lengths - 1.0 - positions
    log_weights = exponents * math.log(alpha)
    log_weights = log_weights.masked_fill(mask == 0, -float("inf"))
    max_weight = log_weights.max(dim=-1, keepdim=True).values
    weights = torch.exp(log_weights - max_weight) * mask
    weights = weights * lengths / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    advantages = normalized.unsqueeze(-1) * weights / torch.sqrt(lengths)
    advantages = advantages * mask
    return advantages, advantages.clone()


def install_appworld_advantage_estimators() -> None:
    """Register AppWorld estimators once in the installed veRL runtime."""

    from verl.trainer.ppo import core_algos
    from appworld_agentic_rl.event_ppo import event_replay_advantage

    registry = getattr(core_algos, "ADV_ESTIMATOR_REGISTRY", {})
    if "grpo_position_credit" not in registry:
        core_algos.register_adv_est("grpo_position_credit")(_legacy_grpo_position_credit)
    if "grpo_event_credit" not in registry:
        core_algos.register_adv_est("grpo_event_credit")(_event_aligned_credit)
    if "event_replay" not in registry:
        core_algos.register_adv_est("event_replay")(event_replay_advantage)


def install_appworld_grpo_runtime() -> None:
    """Entry point for veRL runtime setup hooks."""

    install_appworld_advantage_estimators()


__all__ = [
    "_active_event_spans",
    "_event_aligned_credit",
    "install_appworld_advantage_estimators",
    "install_appworld_grpo_runtime",
]
