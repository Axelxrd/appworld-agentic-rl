from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from appworld_agentic_rl.training_pools import (
    POOL_NAMES,
    build_training_pools,
    write_training_pool_manifests,
)


def _rows(
    scenario_id: str,
    count: int,
    *,
    source_split: str = "train",
    difficulty: int = 1,
    num_apps: int = 1,
) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"{scenario_id}_{index}",
            "scenario_id": scenario_id,
            "source_split": source_split,
            "difficulty": difficulty,
            "num_apps": num_apps,
            "generated": index > 3,
        }
        for index in range(1, count + 1)
    ]


def _training_rows() -> list[dict[str, object]]:
    return [
        *_rows("aaa0001", 3, difficulty=1),
        *_rows("aaa0002", 4, difficulty=1),
        *_rows("aaa0003", 20, difficulty=2, num_apps=2),
        *_rows("aaa0004", 3, source_split="dev", difficulty=2),
        *_rows("aaa0005", 5, source_split="test_normal", difficulty=3),
        *_rows("aaa0006", 3, source_split="test_challenge", difficulty=3, num_apps=2),
    ]


def test_training_pool_partition_is_deterministic_complete_and_leak_free() -> None:
    rows = _training_rows()
    holdout = _rows("hold001", 3, source_split="test_challenge", difficulty=3)
    first = build_training_pools(
        rows,
        holdout_rows=holdout,
        rl_novel_scenarios=2,
        rl_active_cap=3,
        seed=17,
    )
    second = build_training_pools(
        rows,
        holdout_rows=holdout,
        rl_novel_scenarios=2,
        rl_active_cap=3,
        seed=17,
    )
    assert first == second
    assert set(first) == set(POOL_NAMES)

    all_rows = [row for name in POOL_NAMES for row in first[name]]
    assert len(all_rows) == len(rows)
    assert {row["task_id"] for row in all_rows} == {row["task_id"] for row in rows}
    assert len({row["task_id"] for row in all_rows}) == len(rows)
    assert not ({row["task_id"] for row in all_rows} & {row["task_id"] for row in holdout})

    sft_scenarios = {row["scenario_id"] for row in first["teacher_sft"]}
    novel_scenarios = {
        row["scenario_id"]
        for name in ("online_rl_active_novel", "online_rl_reserve")
        for row in first[name]
        if row["rl_novel_scenario"]
    }
    assert len(novel_scenarios) == 2
    assert sft_scenarios.isdisjoint(novel_scenarios)


def test_active_cap_routes_excess_tasks_to_reserve() -> None:
    pools = build_training_pools(
        _training_rows(),
        rl_novel_scenarios=1,
        rl_active_cap=2,
        seed=23,
    )
    active_counts: dict[str, int] = {}
    for name in ("online_rl_active_shared", "online_rl_active_novel"):
        for row in pools[name]:
            scenario_id = str(row["scenario_id"])
            active_counts[scenario_id] = active_counts.get(scenario_id, 0) + 1
    assert max(active_counts.values()) <= 2
    assert pools["online_rl_reserve"]
    assert all(
        row["sampling_policy"] == "uniform_scenario_then_uniform_task"
        for name in POOL_NAMES
        for row in pools[name]
    )


def test_written_manifests_and_dataset_hashes_are_stable() -> None:
    Path(".analysis").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".analysis") as temporary_directory:
        temp_path = Path(temporary_directory)
        rows = _training_rows()
        pools = build_training_pools(
            rows, rl_novel_scenarios=2, rl_active_cap=3, seed=31
        )
        source = temp_path / "train_expanded.jsonl"
        source.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        holdout = temp_path / "validation.jsonl"
        holdout.write_text("", encoding="utf-8")
        first_output = temp_path / "first"
        second_output = temp_path / "second"
        first = write_training_pool_manifests(
            pools,
            output_dir=first_output,
            datasets_dir=first_output / "datasets",
            source_manifest=source,
            holdout_manifests=[holdout],
            seed=31,
            rl_novel_scenarios=2,
            rl_active_cap=3,
        )
        second = write_training_pool_manifests(
            pools,
            output_dir=second_output,
            datasets_dir=second_output / "datasets",
            source_manifest=source,
            holdout_manifests=[holdout],
            seed=31,
            rl_novel_scenarios=2,
            rl_active_cap=3,
        )
        assert first["manifest_sha256"] == second["manifest_sha256"]
        assert first["appworld_dataset_sha256"] == second["appworld_dataset_sha256"]
        assert first["invariants"]["all_source_tasks_partitioned_exactly_once"] is True
        assert (first_output / "teacher_sft.jsonl").is_file()
        assert (first_output / "datasets" / "appworld_rl_v1_teacher_sft.txt").is_file()
