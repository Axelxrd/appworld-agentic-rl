"""verl SFT dataset adapter for AppWorld tool-call trajectories.

The canonical AppWorld JSONL keeps OpenAI-compatible tool-call arguments as a
JSON string.  Qwen3.5's chat template expects the same field as a mapping, and
parquet round-tripping turns nested lists into numpy arrays.  Normalize those
representation details here while delegating tokenization and loss masking to
verl's ``MultiTurnSFTDataset``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def _to_python(value: Any) -> Any:
    """Convert parquet-produced numpy containers into ordinary Python values."""

    if isinstance(value, Mapping):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_to_python(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_to_python(item) for item in value]
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _normalize_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse canonical JSON-string arguments for Qwen3.5's Jinja template."""

    normalized = _to_python(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid tool-call arguments for {function.get('name')}: {exc}"
                    ) from exc
                if not isinstance(parsed, Mapping):
                    raise ValueError(
                        f"tool-call arguments for {function.get('name')} must decode to an object"
                    )
                function["arguments"] = dict(parsed)
    return normalized


def _decode_json_column(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON-encoded SFT column") from exc
    return value


class AppWorldMultiTurnSFTDataset(MultiTurnSFTDataset):
    """Multi-turn verl dataset with AppWorld/Qwen3.5 representation fixes."""

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        self.messages = [
            _normalize_tool_arguments(_decode_json_column(messages))
            for messages in self.messages
        ]
        if self.tools is not None:
            self.tools = [_to_python(_decode_json_column(tools)) for tools in self.tools]
        if self.enable_thinking is None:
            self.enable_thinking = [False] * len(self.messages)

    @staticmethod
    def _assistant_spans(rendered: str) -> list[tuple[int, int]]:
        """Return character spans containing assistant-generated action text."""

        spans: list[tuple[int, int]] = []
        header = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        cursor = 0
        while True:
            start = rendered.find(header, cursor)
            if start < 0:
                break
            content_start = start + len(header)
            end = rendered.find(end_marker, content_start)
            if end < 0:
                raise ValueError("assistant message is missing an im_end marker")
            content = rendered[content_start:end]
            # Qwen3.5 inserts an empty thinking block before later turns. It is
            # a template prefix, so leave it outside the loss mask.
            if content.startswith("<think>"):
                thinking_end = content.find("</think>")
                if thinking_end < 0:
                    raise ValueError("assistant thinking block is not closed")
                content_start += thinking_end + len("</think>")
                while content_start < end and rendered[content_start] in "\n ":
                    content_start += 1
            elif content and content[0].isspace():
                content_start += len(content) - len(content.lstrip())
            if content_start < end:
                # Train the assistant turn terminator as part of the action;
                # the following newline remains template formatting.
                spans.append((content_start, end + len(end_marker)))
            cursor = end + len(end_marker)
        return spans

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else False
        )
        template_kwargs = dict(self.apply_chat_template_kwargs)
        template_kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **template_kwargs,
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        assistant_spans = self._assistant_spans(rendered)
        loss_mask = [
            int(any(start < span_end and end > span_start for span_start, span_end in assistant_spans))
            for start, end in offsets
        ]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={len(input_ids)} is larger than max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(f"unknown truncation method: {self.truncation}")

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_tensor = torch.tensor(loss_mask, dtype=torch.long)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
        if self.pad_mode == "right" and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_tensor = torch.cat((input_tensor, torch.full((pad_length,), pad_id, dtype=torch.long)))
            loss_tensor = torch.cat((loss_tensor, torch.zeros(pad_length, dtype=torch.long)))
            attention_tensor = torch.cat((attention_tensor, torch.zeros(pad_length, dtype=torch.long)))
        elif self.pad_mode not in {"no_padding", "right"}:
            raise ValueError(f"unknown pad mode: {self.pad_mode}")

        position_ids = torch.arange(len(input_tensor), dtype=torch.long)
        position_ids[attention_tensor == 0] = 0
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_tensor,
            "position_ids": position_ids,
            "loss_mask": loss_tensor,
        }


__all__ = ["AppWorldMultiTurnSFTDataset"]
