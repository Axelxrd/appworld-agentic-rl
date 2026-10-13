"""Single context-budget contract shared by RL and frozen evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CONTEXT_POLICY_VERSION = "appworld-structured-memory-token-v1"
CHARS_PER_TOKEN_ESTIMATE = 4
# Remote teachers do not always expose their tokenizer.  Use a deliberately
# conservative character budget for that path instead of presenting it as an
# exact projection of the RL token budget.
TEACHER_CONTEXT_TRIGGER_CHARS = 131_072
TEACHER_CONTEXT_TARGET_CHARS = 98_304
TEACHER_CHARS_PER_TOKEN_ESTIMATE = 2


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    # Compact before the model reaches the hard context limit, and retain a
    # materially smaller memory window so repeated compactions do not churn.
    trigger_tokens: int = 65_536
    target_tokens: int = 49_152
    min_generation_tokens: int = 4_096
    preserve_recent_groups: int = 4
    max_observation_chars: int = 12_000

    def __post_init__(self) -> None:
        if self.trigger_tokens <= self.target_tokens:
            raise ValueError("context trigger must exceed target")
        if self.target_tokens <= 0 or self.min_generation_tokens <= 0:
            raise ValueError("context token budgets must be positive")
        if self.preserve_recent_groups < 1 or self.max_observation_chars < 1:
            raise ValueError("context policy limits must be positive")

    @property
    def trigger_chars(self) -> int:
        return self.trigger_tokens * CHARS_PER_TOKEN_ESTIMATE

    @property
    def target_chars(self) -> int:
        return self.target_tokens * CHARS_PER_TOKEN_ESTIMATE

    def to_dict(self) -> dict[str, int | str]:
        return {
            "version": CONTEXT_POLICY_VERSION,
            **asdict(self),
            "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
            "trigger_chars": self.trigger_chars,
            "target_chars": self.target_chars,
        }


DEFAULT_CONTEXT_POLICY = ContextPolicy()


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "CONTEXT_POLICY_VERSION",
    "ContextPolicy",
    "DEFAULT_CONTEXT_POLICY",
    "TEACHER_CONTEXT_TARGET_CHARS",
    "TEACHER_CONTEXT_TRIGGER_CHARS",
    "TEACHER_CHARS_PER_TOKEN_ESTIMATE",
]
