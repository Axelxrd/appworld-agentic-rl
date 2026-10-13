"""Deterministic terminal reward helpers for AppWorld rollouts."""

from __future__ import annotations

from typing import Any


def _contains_failure(value: Any) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in ("fail", "error", "traceback", "exception"))


def _contains_success(value: Any) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in ("pass", "success", "all tests passed", "complete"))


def infer_evaluation_success(evaluation: dict[str, Any]) -> bool | None:
    """Infer pass/fail across AppWorld tracker versions without binding internals."""

    for key in ("success", "passed", "pass", "all_passed", "task_success"):
        value = evaluation.get(key)
        if isinstance(value, bool):
            return value

    metrics = evaluation.get("metrics")
    if isinstance(metrics, dict):
        for key in ("success", "passed", "task_success"):
            value = metrics.get(key)
            if isinstance(value, bool):
                return value

    data = evaluation.get("data")
    if isinstance(data, dict):
        nested_success = infer_evaluation_success(data)
        if nested_success is not None:
            return nested_success
        if _contains_failure(data):
            return False
        if _contains_success(data):
            return True

    report = evaluation.get("report")
    if report is not None:
        if _contains_failure(report):
            return False
        if _contains_success(report):
            return True
    return None


def compute_terminal_reward(
    evaluation: dict[str, Any] | None,
    *,
    had_execution_error: bool = False,
    finished_without_completion: bool = False,
) -> float:
    """First-pass terminal reward aligned with AppWorld programmatic evaluation."""

    if finished_without_completion:
        return -0.3 if not had_execution_error else -0.7
    if evaluation is None:
        return -1.0 if had_execution_error else -0.3

    success = infer_evaluation_success(evaluation)
    collateral_damage = bool(
        evaluation.get("collateral_damage")
        or evaluation.get("has_collateral_damage")
        or evaluation.get("state_integrity_failed")
    )

    if success is True and not collateral_damage:
        return 1.0
    if success is True and collateral_damage:
        return 0.7
    if collateral_damage:
        return -0.7
    if had_execution_error:
        return -0.7
    return -0.3 if success is False else 0.0
