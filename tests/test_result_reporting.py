import csv
import json
from pathlib import Path

import pytest

from appworld_agentic_rl.result_reporting import build_final_model_comparison


def _summary(path: Path, *, successes: int, clean: int) -> Path:
    payload = {
        "expected_tasks": 180,
        "successful_tasks": successes,
        "task_success_rate_over_expected": successes / 180,
        "clean_successful_tasks": clean,
        "clean_task_success_rate_over_expected": clean / 180,
        "passed_requirements": successes + 10,
        "total_requirements": 200,
        "requirement_pass_rate": (successes + 10) / 200,
        "collateral_damage_rate": 0.05,
        "execution_error_rate": 0.1,
        "infrastructure_error_rate": 0.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_final_model_comparison_writes_all_formats(tmp_path: Path) -> None:
    summaries = {
        "base": _summary(tmp_path / "base.json", successes=0, clean=0),
        "sft": _summary(tmp_path / "sft.json", successes=96, clean=89),
        "rl": _summary(tmp_path / "rl.json", successes=117, clean=110),
    }
    output = tmp_path / "comparison"

    payload = build_final_model_comparison(summaries, output)

    assert payload["expected_tasks"] == 180
    assert payload["models"][2]["successful_tasks"] == 117
    assert (output / "comparison.json").is_file()
    assert "117 / 180 (65.00%)" in (output / "comparison.md").read_text(encoding="utf-8")
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model_key"] for row in rows] == ["base", "sft", "rl"]


def test_build_final_model_comparison_rejects_mismatched_manifests(tmp_path: Path) -> None:
    paths = {
        "base": _summary(tmp_path / "base.json", successes=0, clean=0),
        "sft": _summary(tmp_path / "sft.json", successes=96, clean=89),
        "rl": _summary(tmp_path / "rl.json", successes=117, clean=110),
    }
    data = json.loads(paths["rl"].read_text(encoding="utf-8"))
    data["expected_tasks"] = 179
    paths["rl"].write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="different task counts"):
        build_final_model_comparison(paths, tmp_path / "comparison")
