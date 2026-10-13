"""Aggregate frozen-evaluation summaries into reproducible comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_SPECS = (
    ("base", "Qwen3.5-4B Base"),
    ("sft", "Qwen3.5-4B SFT-final"),
    ("rl", "Qwen3.5-4B RL-best"),
)


def _number(summary: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return float(value)
    return float(default)


def _comparison_row(key: str, label: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = int(summary.get("expected_tasks") or 0)
    return {
        "model_key": key,
        "model": label,
        "expected_tasks": expected,
        "successful_tasks": int(summary.get("successful_tasks") or 0),
        "task_success_rate": _number(
            summary, "task_success_rate_over_expected", "task_success_rate"
        ),
        "clean_successful_tasks": int(summary.get("clean_successful_tasks") or 0),
        "clean_success_rate": _number(
            summary,
            "clean_task_success_rate_over_expected",
            "clean_task_success_rate",
        ),
        "passed_requirements": int(summary.get("passed_requirements") or 0),
        "total_requirements": int(summary.get("total_requirements") or 0),
        "requirement_pass_rate": _number(summary, "requirement_pass_rate"),
        "collateral_damage_rate": _number(summary, "collateral_damage_rate"),
        "execution_error_rate": _number(summary, "execution_error_rate"),
        "infrastructure_error_rate": _number(summary, "infrastructure_error_rate"),
    }


def _count_rate(count: int, denominator: int, rate: float) -> str:
    if denominator > 0:
        return f"{count} / {denominator} ({rate * 100:.2f}%)"
    return f"{rate * 100:.2f}%"


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Frozen Final Model Comparison",
        "",
        "| Model | Task Success | Clean Success | Requirement Pass | Collateral Damage | Execution Error | Infrastructure Error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        requirement_denominator = int(row["total_requirements"])
        lines.append(
            "| {model} | {task} | {clean} | {requirement} | {collateral:.2%} | "
            "{execution:.2%} | {infrastructure:.2%} |".format(
                model=row["model"],
                task=_count_rate(
                    int(row["successful_tasks"]),
                    int(row["expected_tasks"]),
                    float(row["task_success_rate"]),
                ),
                clean=_count_rate(
                    int(row["clean_successful_tasks"]),
                    int(row["expected_tasks"]),
                    float(row["clean_success_rate"]),
                ),
                requirement=_count_rate(
                    int(row["passed_requirements"]),
                    requirement_denominator,
                    float(row["requirement_pass_rate"]),
                ),
                collateral=float(row["collateral_damage_rate"]),
                execution=float(row["execution_error_rate"]),
                infrastructure=float(row["infrastructure_error_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_final_model_comparison(
    summaries: Mapping[str, str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Load Base/SFT/RL summaries and write JSON, CSV, and Markdown reports."""

    expected_keys = {key for key, _ in MODEL_SPECS}
    if set(summaries) != expected_keys:
        raise ValueError(f"summaries must contain exactly {sorted(expected_keys)}")
    rows: list[dict[str, Any]] = []
    source_summaries: dict[str, str] = {}
    expected_counts: set[int] = set()
    for key, label in MODEL_SPECS:
        path = Path(summaries[key]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping):
            raise ValueError(f"evaluation summary must be an object: {path}")
        row = _comparison_row(key, label, summary)
        if row["expected_tasks"] <= 0:
            raise ValueError(f"evaluation summary has no expected tasks: {path}")
        expected_counts.add(int(row["expected_tasks"]))
        rows.append(row)
        source_summaries[key] = str(path)
    if len(expected_counts) != 1:
        raise ValueError(f"model summaries use different task counts: {sorted(expected_counts)}")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "appworld-final-model-comparison-v1",
        "expected_tasks": next(iter(expected_counts)),
        "sources": source_summaries,
        "models": rows,
    }
    (destination / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = list(rows[0])
    with (destination / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "comparison.md").write_text(_markdown(rows), encoding="utf-8")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Base/SFT/RL frozen-final table.")
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--rl-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    build_final_model_comparison(
        {"base": args.base_summary, "sft": args.sft_summary, "rl": args.rl_summary},
        args.output_dir,
    )
    print((args.output_dir.resolve() / "comparison.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_final_model_comparison"]
