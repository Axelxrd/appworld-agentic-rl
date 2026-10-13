"""Isolated AppWorld evaluator entry point.

AppWorld's evaluator installs its own freezegun context. Running it in a fresh
interpreter prevents evaluator cleanup from recursing through the rollout
process's already-patched datetime state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _serialize_tracker(tracker: Any) -> dict[str, Any]:
    evaluation: dict[str, Any] = {}
    if hasattr(tracker, "to_dict"):
        try:
            evaluation["data"] = _jsonable(tracker.to_dict())
        except TypeError:
            pass
    if hasattr(tracker, "report"):
        try:
            evaluation["report"] = str(tracker.report())
        except TypeError:
            pass
    if not evaluation:
        evaluation["data"] = _jsonable(tracker)
    return evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appworld-root", type=Path)
    parser.add_argument("--suppress-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from appworld import evaluate_task, update_root

        if args.appworld_root is not None:
            update_root(str(args.appworld_root.resolve()))
        tracker = evaluate_task(
            task_id=args.task_id,
            experiment_name=args.experiment_name,
            suppress_errors=args.suppress_errors,
        )
        payload: dict[str, Any] = {"evaluation": _serialize_tracker(tracker)}
    except BaseException as exc:
        payload = {"error": type(exc).__name__ + ": " + str(exc)}
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
