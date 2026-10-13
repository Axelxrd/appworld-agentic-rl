"""Tokenizer-exact chat-context accounting for frozen policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class HuggingFaceChatTokenCounter:
    """Count the exact chat-template tokens sent to a local vLLM model."""

    def __init__(
        self,
        tokenizer_path: str | Path,
        *,
        tools: Sequence[Mapping[str, Any]],
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific dependency
            raise RuntimeError(
                "token-exact evaluation requires transformers; install requirements-eval.txt"
            ) from exc

        self.tokenizer_path = str(tokenizer_path)
        self.tools = [dict(tool) for tool in tools]
        self.enable_thinking = bool(enable_thinking)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tools=self.tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if encoded is None:
            raise RuntimeError("tokenizer chat template returned no input_ids")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if (
            isinstance(encoded, Sequence)
            and encoded
            and isinstance(encoded[0], Sequence)
        ):
            if len(encoded) != 1:
                raise RuntimeError("tokenizer unexpectedly returned a batched context")
            encoded = encoded[0]
        if not isinstance(encoded, Sequence):
            raise RuntimeError("tokenizer chat template returned an unsupported value")
        return len(encoded)


__all__ = ["HuggingFaceChatTokenCounter"]
