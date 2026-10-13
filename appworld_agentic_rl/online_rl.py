"""Small online lightweight-process-reward and event-credit batch builder.

This module intentionally stops at a trainer-ready batch.  It is usable with
the existing OpenAI-compatible actor during smoke tests and keeps veRL-specific
weight-update code behind a narrow boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

from appworld_agentic_rl.event_credit import compute_event_credit, group_normalize
from appworld_agentic_rl.process_reward import score_trajectory
from appworld_agentic_rl.harness import TeacherHarnessConfig
from appworld_agentic_rl.teacher_rollout import POLICY_SYSTEM_PROMPT, run_teacher_rollout


def _action_token_counts(row: Mapping[str, Any]) -> list[int]:
    """Estimate action-token lengths for smoke; trainer integration replaces this with offsets."""

    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else []
    counts: list[int] = []
    for step in steps or []:
        payload = json.dumps(
            {"tool_name": step.get("tool_name"), "arguments": step.get("arguments")},
            ensure_ascii=False,
            sort_keys=True,
        )
        counts.append(max(1, (len(payload) + 3) // 4))
    return counts


def build_online_group(
    *,
    task_id: str,
    client: Any,
    adapter_factory: Callable[[], Any],
    split: str = "train",
    group_size: int = 2,
    max_model_turns: int = 8,
    harness_config: TeacherHarnessConfig | None = None,
    process_lambda: float = 0.3,
) -> dict[str, Any]:
    """Sample one task group and return auditable process-reward training records."""

    rows: list[dict[str, Any]] = []
    for _ in range(group_size):
        adapter = adapter_factory()
        try:
            row = run_teacher_rollout(
                adapter=adapter,
                client=client,
                task_id=task_id,
                split=split,
                max_model_turns=max_model_turns,
                harness_config=harness_config,
                system_prompt=POLICY_SYSTEM_PROMPT,
            )
            rows.append(row)
        finally:
            adapter.close()
    scored: list[dict[str, Any]] = []
    for row in rows:
        terminal = float(row.get("terminal_reward") or 0.0)
        process_reward = score_trajectory(row.get("trajectory"), terminal_reward=terminal)
        total = terminal + process_lambda * process_reward.process_score
        infrastructure_invalid = bool(row.get("infrastructure_invalid"))
        scored.append(
            {
                "row": row,
                "terminal_reward": terminal,
                "process_reward": process_reward,
                "total_reward": total,
                "infrastructure_invalid": infrastructure_invalid,
            }
        )

    # Infrastructure failures are retained for audit, but must not contribute
    # to a GRPO group mean/std or receive a trainable advantage.
    eligible = [item for item in scored if not item["infrastructure_invalid"]]
    normalized = group_normalize([item["total_reward"] for item in eligible])
    eligible_advantages = iter(normalized)
    records: list[dict[str, Any]] = []
    for item in scored:
        process_reward = item["process_reward"]
        training_eligible = not item["infrastructure_invalid"]
        advantage = next(eligible_advantages) if training_eligible else None
        deltas = [event.delta for event in process_reward.events]
        event_credit = compute_event_credit(
            terminal_reward=item["terminal_reward"],
            process_deltas=deltas,
            token_counts=_action_token_counts(item["row"]),
            normalized_advantage=float(advantage or 0.0),
        )
        records.append(
            {
                "task_id": task_id,
                "terminal_reward": item["terminal_reward"],
                "process_score": process_reward.process_score,
                "total_reward": item["total_reward"],
                "group_advantage": advantage,
                "training_eligible": training_eligible,
                "exclusion_reason": "infrastructure_invalid" if not training_eligible else None,
                "process_reward": process_reward.to_dict(),
                "event_credit": event_credit.to_dict(),
                "rollout": item["row"],
            }
        )
    return {
        "schema_version": "appworld-online-rl-smoke-v1",
        "task_id": task_id,
        "group_size": group_size,
        "process_lambda": process_lambda,
        "eligible_count": len(eligible),
        "excluded_count": len(scored) - len(eligible),
        "trainable": bool(eligible),
        "records": records,
    }


def write_group(path: str | Path, group: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(group), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["build_online_group", "write_group"]
