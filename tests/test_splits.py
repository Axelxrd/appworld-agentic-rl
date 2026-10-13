from __future__ import annotations

import json
from pathlib import Path

from appworld_agentic_rl.splits import (
    ScenarioRecord,
    _largest_remainder_quotas,
    build_custom_split,
    write_split_manifests,
)
from appworld_agentic_rl.task_generation import (
    _generated_scenario_is_complete,
    _known_searchable_setup_error,
    prepare_isolated_root,
)


def _scenario(
    source: str, number: int, difficulty: int, num_apps: int
) -> ScenarioRecord:
    prefix = {
        "train": "tr",
        "dev": "dv",
        "test_normal": "tn",
        "test_challenge": "tc",
    }[source]
    scenario_id = f"{prefix}{number:05d}"
    return ScenarioRecord(
        scenario_id=scenario_id,
        source_split=source,
        task_ids=tuple(f"{scenario_id}_{index}" for index in range(1, 4)),
        difficulty=difficulty,
        num_apps=num_apps,
    )


def test_largest_remainder_quotas_preserve_total_and_capacity() -> None:
    populations = {(1, 1): 24, (2, 1): 50, (3, 2): 65}
    quotas = _largest_remainder_quotas(populations, 60)
    assert sum(quotas.values()) == 60
    assert all(0 <= quotas[key] <= populations[key] for key in populations)


def test_custom_split_is_deterministic_and_scenario_disjoint() -> None:
    source = {
        "train": [_scenario("train", i, 1, 1) for i in range(3)],
        "dev": [_scenario("dev", i, 2, 1) for i in range(2)],
        "test_normal": [_scenario("test_normal", i, 2, 2) for i in range(4)],
        "test_challenge": [
            _scenario("test_challenge", i, 1 + i % 3, 1 + i % 2) for i in range(20)
        ],
    }
    first = build_custom_split(
        source, validation_scenarios=4, final_scenarios=6, seed=7
    )
    second = build_custom_split(
        source, validation_scenarios=4, final_scenarios=6, seed=7
    )
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 19,
        "validation": 4,
        "final": 6,
    }
    split_ids = [
        {scenario.scenario_id for scenario in first[name]}
        for name in ("train", "validation", "final")
    ]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])


def test_manifest_contains_ids_and_hashes_but_not_instructions(tmp_path: Path) -> None:
    source = {
        "train": [_scenario("train", 1, 1, 1)],
        "dev": [_scenario("dev", 1, 1, 1)],
        "test_normal": [_scenario("test_normal", 1, 2, 2)],
        "test_challenge": [
            _scenario("test_challenge", i, 1 + i % 3, 1 + i % 2) for i in range(6)
        ],
    }
    split = build_custom_split(
        source, validation_scenarios=1, final_scenarios=2, seed=11
    )
    metadata = write_split_manifests(split, tmp_path, seed=11, source_scenarios=source)
    rows = [
        json.loads(line) for line in (tmp_path / "final.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    assert set(rows[0]) == {
        "task_id",
        "scenario_id",
        "source_split",
        "difficulty",
        "num_apps",
    }
    assert metadata["manifest_sha256"]["final.jsonl"]


def test_generation_root_is_isolated_and_reusable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data" / "datasets").mkdir(parents=True)
    (source / "data" / "version.txt").write_text("0.2.0\n", encoding="utf-8")
    (source / "data" / "datasets" / "train.txt").write_text("abc_1\n", encoding="utf-8")
    output = tmp_path / "generated"
    first = prepare_isolated_root(source, output)
    second = prepare_isolated_root(source, output)
    assert first == second
    assert (output / "data" / "datasets" / "train.txt").read_text() == "abc_1\n"


def test_generated_scenario_completeness_requires_exact_contiguous_tasks(
    tmp_path: Path,
) -> None:
    tasks_directory = tmp_path / "tasks"
    for number in range(1, 4):
        task_directory = tasks_directory / f"abc1234_{number}"
        (task_directory / "ground_truth").mkdir(parents=True)
        (task_directory / "specs.json").write_text("{}", encoding="utf-8")
        (task_directory / "ground_truth" / "compiled_solution.py").write_text(
            "", encoding="utf-8"
        )
        (task_directory / "ground_truth" / "evaluation.py").write_text(
            "", encoding="utf-8"
        )

    assert _generated_scenario_is_complete(tasks_directory, "abc1234", 3)

    (tasks_directory / "abc1234_2" / "specs.json").unlink()
    assert not _generated_scenario_is_complete(tasks_directory, "abc1234", 3)

    (tasks_directory / "abc1234_4").mkdir()
    assert not _generated_scenario_is_complete(tasks_directory, "abc1234", 3)


def test_only_known_generator_setup_failures_are_searchable() -> None:
    sampling_error = ValueError("Could not sample enough items")
    exhausted_months_error = UnboundLocalError(
        "cannot access local variable 'num_relative_transactions'"
    )
    assert _known_searchable_setup_error("any-id", sampling_error)
    assert _known_searchable_setup_error("d0b1f43", exhausted_months_error)
    assert not _known_searchable_setup_error("another-id", exhausted_months_error)
    assert not _known_searchable_setup_error(
        "d0b1f43", UnboundLocalError("another variable")
    )
