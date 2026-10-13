"""Reproducible scenario-level splits for the AppWorld-RL benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "appworld-rl-split-v1"
OFFICIAL_SPLITS = ("train", "dev", "test_normal", "test_challenge")


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    source_split: str
    task_ids: tuple[str, ...]
    difficulty: int
    num_apps: int

    @property
    def stratum(self) -> tuple[int, int]:
        return self.difficulty, self.num_apps


def _stable_digest(seed: int, label: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{value}".encode("utf-8")).hexdigest()


def _largest_remainder_quotas(
    populations: Mapping[tuple[int, int], int], total: int
) -> dict[tuple[int, int], int]:
    population_total = sum(populations.values())
    if total < 0 or total > population_total:
        raise ValueError(f"requested {total} scenarios from a population of {population_total}")
    if population_total == 0:
        return {}

    exact = {key: count * total / population_total for key, count in populations.items()}
    quotas = {key: min(populations[key], int(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        populations,
        key=lambda key: (-(exact[key] - int(exact[key])), key),
    )
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < populations[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not apportion all requested scenarios")
    return quotas


def _select_stratified(
    scenarios: Sequence[ScenarioRecord],
    *,
    count: int,
    seed: int,
    label: str,
) -> tuple[list[ScenarioRecord], list[ScenarioRecord]]:
    by_stratum: dict[tuple[int, int], list[ScenarioRecord]] = defaultdict(list)
    for scenario in scenarios:
        by_stratum[scenario.stratum].append(scenario)
    quotas = _largest_remainder_quotas(
        {stratum: len(records) for stratum, records in by_stratum.items()}, count
    )

    selected_ids: set[str] = set()
    for stratum, records in by_stratum.items():
        ordered = sorted(
            records,
            key=lambda record: _stable_digest(seed, label, record.scenario_id),
        )
        selected_ids.update(record.scenario_id for record in ordered[: quotas[stratum]])
    selected = [scenario for scenario in scenarios if scenario.scenario_id in selected_ids]
    remaining = [scenario for scenario in scenarios if scenario.scenario_id not in selected_ids]
    return selected, remaining


def build_custom_split(
    scenarios_by_source: Mapping[str, Sequence[ScenarioRecord]],
    *,
    validation_scenarios: int = 20,
    final_scenarios: int = 60,
    seed: int = 20260814,
) -> dict[str, list[ScenarioRecord]]:
    missing = set(OFFICIAL_SPLITS) - set(scenarios_by_source)
    if missing:
        raise ValueError(f"missing official splits: {sorted(missing)}")

    all_scenarios = [
        scenario
        for split_name in OFFICIAL_SPLITS
        for scenario in scenarios_by_source[split_name]
    ]
    scenario_ids = [scenario.scenario_id for scenario in all_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("a scenario appears in more than one official split")

    challenge = list(scenarios_by_source["test_challenge"])
    final, challenge_remaining = _select_stratified(
        challenge, count=final_scenarios, seed=seed, label="final"
    )
    validation, challenge_train = _select_stratified(
        challenge_remaining,
        count=validation_scenarios,
        seed=seed,
        label="validation",
    )
    train = [
        *scenarios_by_source["train"],
        *scenarios_by_source["dev"],
        *scenarios_by_source["test_normal"],
        *challenge_train,
    ]
    result = {"train": train, "validation": validation, "final": final}
    _validate_custom_split(result, expected_scenarios=len(all_scenarios))
    return result


def _validate_custom_split(
    split: Mapping[str, Sequence[ScenarioRecord]], *, expected_scenarios: int
) -> None:
    seen_scenarios: set[str] = set()
    seen_tasks: set[str] = set()
    for split_name in ("train", "validation", "final"):
        for scenario in split[split_name]:
            if scenario.scenario_id in seen_scenarios:
                raise ValueError(f"scenario leakage detected: {scenario.scenario_id}")
            seen_scenarios.add(scenario.scenario_id)
            if len(scenario.task_ids) != 3:
                raise ValueError(
                    f"scenario {scenario.scenario_id} must contain exactly 3 official tasks"
                )
            for task_id in scenario.task_ids:
                if task_id in seen_tasks:
                    raise ValueError(f"task leakage detected: {task_id}")
                seen_tasks.add(task_id)
    if len(seen_scenarios) != expected_scenarios:
        raise ValueError(
            f"expected {expected_scenarios} scenarios, found {len(seen_scenarios)}"
        )


def load_official_scenarios(appworld_root: str | Path) -> dict[str, list[ScenarioRecord]]:
    from appworld import load_task_ids, update_root
    from appworld.task import task_id_to_generator_id, task_id_to_number

    root = Path(appworld_root).resolve()
    update_root(str(root))
    result: dict[str, list[ScenarioRecord]] = {}
    for split_name in OFFICIAL_SPLITS:
        grouped: dict[str, list[str]] = defaultdict(list)
        for task_id in load_task_ids(split_name):
            grouped[task_id_to_generator_id(task_id)].append(task_id)

        records: list[ScenarioRecord] = []
        for scenario_id, task_ids in grouped.items():
            task_ids = sorted(task_ids, key=task_id_to_number)
            metadata_values = []
            for task_id in task_ids:
                metadata_path = root / "data" / "tasks" / task_id / "ground_truth" / "metadata.json"
                metadata_values.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            difficulties = {int(metadata["difficulty"]) for metadata in metadata_values}
            num_apps_values = {int(metadata["num_apps"]) for metadata in metadata_values}
            if len(difficulties) != 1 or len(num_apps_values) != 1:
                raise ValueError(f"inconsistent metadata within scenario {scenario_id}")
            records.append(
                ScenarioRecord(
                    scenario_id=scenario_id,
                    source_split=split_name,
                    task_ids=tuple(task_ids),
                    difficulty=difficulties.pop(),
                    num_apps=num_apps_values.pop(),
                )
            )
        result[split_name] = sorted(records, key=lambda record: record.scenario_id)
    return result


def _jsonl_bytes(records: Iterable[ScenarioRecord]) -> bytes:
    rows = []
    for scenario in sorted(records, key=lambda record: (record.source_split, record.scenario_id)):
        for task_id in scenario.task_ids:
            row = {
                "task_id": task_id,
                "scenario_id": scenario.scenario_id,
                "source_split": scenario.source_split,
                "difficulty": scenario.difficulty,
                "num_apps": scenario.num_apps,
            }
            rows.append(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode("utf-8")


def _summary(records: Sequence[ScenarioRecord]) -> dict[str, object]:
    return {
        "scenarios": len(records),
        "tasks": sum(len(record.task_ids) for record in records),
        "source_scenarios": dict(sorted(Counter(record.source_split for record in records).items())),
        "difficulty_scenarios": {
            str(key): value
            for key, value in sorted(Counter(record.difficulty for record in records).items())
        },
        "num_apps_scenarios": {
            str(key): value
            for key, value in sorted(Counter(record.num_apps for record in records).items())
        },
    }


def write_split_manifests(
    split: Mapping[str, Sequence[ScenarioRecord]],
    output_dir: str | Path,
    *,
    seed: int,
    source_scenarios: Mapping[str, Sequence[ScenarioRecord]],
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split_name in ("train", "validation", "final"):
        content = _jsonl_bytes(split[split_name])
        path = output / f"{split_name}.jsonl"
        path.write_bytes(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "name": "AppWorld-RL-v1",
        "official_appworld_comparable": False,
        "selection": {
            "seed": seed,
            "holdout_source": "test_challenge",
            "strata": ["difficulty", "num_apps"],
            "unit": "scenario",
            "algorithm": "largest-remainder allocation plus sha256 ordering",
        },
        "official_source": {
            name: _summary(list(source_scenarios[name])) for name in OFFICIAL_SPLITS
        },
        "custom_split": {
            name: _summary(list(split[name])) for name in ("train", "validation", "final")
        },
        "manifest_sha256": hashes,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_split_manifests(
    *,
    appworld_root: str | Path,
    output_dir: str | Path,
    validation_scenarios: int = 20,
    final_scenarios: int = 60,
    seed: int = 20260814,
) -> dict[str, object]:
    source = load_official_scenarios(appworld_root)
    split = build_custom_split(
        source,
        validation_scenarios=validation_scenarios,
        final_scenarios=final_scenarios,
        seed=seed,
    )
    return write_split_manifests(split, output_dir, seed=seed, source_scenarios=source)
