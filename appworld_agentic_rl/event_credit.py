"""Length-aware event-level credit-allocation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EventCreditResult:
    total_reward: float
    normalized_advantage: float
    turn_advantages: tuple[float, ...]
    token_advantages: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_reward": self.total_reward,
            "normalized_advantage": self.normalized_advantage,
            "turn_advantages": list(self.turn_advantages),
            "token_advantages": list(self.token_advantages),
        }


def group_normalize(rewards: Sequence[float], *, eps: float = 1e-6) -> tuple[float, ...]:
    """Compute GRPO-style group-relative rewards with a stable zero-variance path."""

    if not rewards:
        return ()
    mean = sum(rewards) / len(rewards)
    variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < eps:
        return tuple(0.0 for _ in rewards)
    return tuple((x - mean) / (std + eps) for x in rewards)


def compute_event_credit(
    *,
    terminal_reward: float,
    process_deltas: Sequence[float],
    token_counts: Sequence[int],
    normalized_advantage: float,
    turn_discount: float = 0.95,
) -> EventCreditResult:
    """Align local process deltas and outcome advantage to action tokens.

    The outcome term is discounted by turn and normalized by sqrt(total action
    length).  A process delta is added to the turn that generated it, so it is
    not broadcast over unrelated observations.
    """

    n = max(len(process_deltas), len(token_counts))
    if n == 0:
        return EventCreditResult(terminal_reward, normalized_advantage, (), ())
    counts = [max(0, int(token_counts[i])) if i < len(token_counts) else 0 for i in range(n)]
    deltas = [float(process_deltas[i]) if i < len(process_deltas) else 0.0 for i in range(n)]
    total_tokens = max(1, sum(counts))
    denom = math.sqrt(total_tokens)
    weights = [turn_discount**i for i in range(n)]
    weight_sum = sum(weights) or 1.0
    turns = tuple(normalized_advantage * w / weight_sum / denom + delta for w, delta in zip(weights, deltas))
    tokens: list[float] = []
    for advantage, count in zip(turns, counts):
        tokens.extend([advantage] * count)
    return EventCreditResult(
        total_reward=terminal_reward,
        normalized_advantage=normalized_advantage,
        turn_advantages=turns,
        token_advantages=tuple(tokens),
    )


__all__ = ["EventCreditResult", "compute_event_credit", "group_normalize"]
