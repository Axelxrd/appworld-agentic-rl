from __future__ import annotations

from typing import Any

from appworld_agentic_rl.token_counting import HuggingFaceChatTokenCounter


def test_chat_token_counter_uses_tools_and_generation_prompt() -> None:
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> list[int]:
            captured["messages"] = messages
            captured.update(kwargs)
            return [1, 2, 3, 4]

    counter = object.__new__(HuggingFaceChatTokenCounter)
    counter.tokenizer_path = "model"
    counter.tools = [{"type": "function", "function": {"name": "inspect_task"}}]
    counter.enable_thinking = False
    counter.tokenizer = FakeTokenizer()

    count = counter([{"role": "user", "content": "task"}])

    assert count == 4
    assert captured["tools"] == counter.tools
    assert captured["tokenize"] is True
    assert captured["add_generation_prompt"] is True
    assert captured["enable_thinking"] is False
