from __future__ import annotations

import sys

from appworld_agentic_rl.cli import parse_args
from appworld_agentic_rl.harness import TeacherHarnessConfig


def test_frozen_evaluation_exposes_structured_memory_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "appworld-agentic-rl",
            "evaluate-frozen",
            "--model",
            "model",
            "--model-revision",
            "revision",
            "--base-url",
            "http://localhost:8000/v1",
            "--condition",
            "structured_memory",
            "--manifest",
            "dev.jsonl",
            "--output-dir",
            "output",
            "--experiment-name",
            "comparison",
            "--bc-guard-mode",
            "enforce",
            "--tokenizer-path",
            "model",
            "--context-compaction-enable",
            "--context-preserve-recent-groups",
            "4",
        ],
    )

    args = parse_args()

    assert args.context_budget_mode == "tokens"
    assert str(args.tokenizer_path) == "model"
    assert args.context_compaction_enable is True
    assert args.context_compaction_trigger_tokens == 65536
    assert args.context_compaction_target_tokens == 49152
    assert args.context_preserve_recent_groups == 4


def test_teacher_collection_uses_conservative_character_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "appworld-agentic-rl",
            "collect-teacher",
            "--model",
            "teacher",
            "--output-dir",
            "output",
        ],
    )

    args = parse_args()

    assert args.max_context_chars == 131072
    assert args.context_compaction_enable is True
    assert args.context_compaction_trigger_chars == 131072
    assert args.context_compaction_target_chars == 98304
    assert args.context_preserve_recent_groups == 4


def test_compaction_rejects_hard_limit_below_trigger() -> None:
    try:
        TeacherHarnessConfig(
            max_context_chars=100000,
            context_compaction_trigger_chars=131072,
            context_compaction_target_chars=98304,
        )
    except ValueError as error:
        assert "max_context_chars" in str(error)
    else:
        raise AssertionError("invalid context contract was accepted")
