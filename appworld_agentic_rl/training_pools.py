"""Deterministic, scenario-balanced training pools for AppWorld Agentic RL."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from appworld_agentic_rl.splits import _stable_digest


SCHEMA_VERSION = "appworld-training-pools-v1"
POOL_NAMES = (
    "teacher_sft",
    "online_rl_active_shared",
    "online_rl_active_novel",
    "online_rl_reserve",
)
DATASET_PREFIX = "appworld_rl_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _capacity_bucket(task_count: int) -> str:
    if task_count <= 3:
        return "official_only"
    if task_count <= 15:
        return "expanded_small"
    if task_count <= 40:
        return "expanded_medium"
    return "expanded_large"


def _sft_quota(task_count: int) -> int:
    if task_count < 1:
        raise ValueError("a scenario must contain at least one task")
    if task_count <= 3:
        return 1
    rounded_sqrt = int(math.sqrt(task_count) + 0.5)
    return min(task_count, 8, max(2, rounded_sqrt))


def _largest_remainder_quotas(
    populations: Mapping[tuple[Any, ...], int], total: int
) -> dict[tuple[Any, ...], int]:
    population_total = sum(populations.values())
    if total < 0 or total > population_total:
        raise ValueError(
            f"requested {total} novel scenarios from a population of {population_total}"
        )
    if not population_total:
        return {}
    exact = {key: count * total / population_total for key, count in populations.items()}
    quotas = {key: min(populations[key], int(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    for key in sorted(
        populations,
        key=lambda item: (-(exact[item] - int(exact[item])), tuple(map(str, item))),
    ):
        if not remaining:
            break
        if quotas[key] < populations[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not apportion all RL-novel scenarios")
    return quotas


def _group_scenarios(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_ids: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        missing = {
            "task_id",
            "scenario_id",
            "source_split",
            "difficulty",
            "num_apps",
        } - set(row)
        if missing:
            raise ValueError(f"training row is missing fields: {sorted(missing)}")
        task_id = str(row["task_id"])
        scenario_id = str(row["scenario_id"])
        if task_id in task_ids:
            raise ValueError(f"duplicate training task: {task_id}")
        task_ids.add(task_id)
        row["task_id"] = task_id
        row["scenario_id"] = scenario_id
        row["source_split"] = str(row["source_split"])
        row["difficulty"] = int(row["difficulty"])
        row["num_apps"] = int(row["num_apps"])
        row["generated"] = bool(row.get("generated", False))
        scenarios[scenario_id].append(row)

    for scenario_id, scenario_rows in scenarios.items():
        attributes = {
            (
                row["source_split"],
                row["difficulty"],
                row["num_apps"],
            )
            for row in scenario_rows
        }
        if len(attributes) != 1:
            raise ValueError(f"inconsistent metadata in scenario {scenario_id}")
    return dict(scenarios)


def _select_rl_novel_scenarios(
    scenarios: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    count: int,
    seed: int,
) -> set[str]:
    by_stratum: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for scenario_id, rows in scenarios.items():
        first = rows[0]
        stratum = (
            int(first["difficulty"]),
            int(first["num_apps"]),
            str(first["source_split"]),
            any(bool(row.get("generated")) for row in rows),
            _capacity_bucket(len(rows)),
        )
        by_stratum[stratum].append(scenario_id)
    quotas = _largest_remainder_quotas(
        {stratum: len(ids) for stratum, ids in by_stratum.items()}, count
    )
    selected: set[str] = set()
    for stratum, scenario_ids in by_stratum.items():
        ordered = sorted(
            scenario_ids,
            key=lambda scenario_id: (
                _stable_digest(seed, "rl_novel_scenario", scenario_id),
                scenario_id,
            ),
        )
        selected.update(ordered[: quotas[stratum]])
    return selected


def _rank_rows(
    rows: Sequence[dict[str, Any]], *, seed: int, label: str
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _stable_digest(seed, label, str(row["task_id"])),
            str(row["task_id"]),
        ),
    )


def _annotate_pool_rows(
    rows: Sequence[dict[str, Any]],
    *,
    pool_name: str,
    scenario_total_tasks: int,
    sft_visible: bool,
    rl_novel_scenario: bool,
) -> list[dict[str, Any]]:
    count = len(rows)
    output: list[dict[str, Any]] = []
    for rank, source in enumerate(rows, 1):
        row = dict(source)
        row.update(
            {
                "pool": pool_name,
                "selection_rank": rank,
                "scenario_total_tasks": scenario_total_tasks,
                "pool_scenario_task_count": count,
                "scenario_sampling_weight": 1.0 / count,
                "sampling_policy": "uniform_scenario_then_uniform_task",
                "sft_visible_scenario": sft_visible,
                "rl_novel_scenario": rl_novel_scenario,
            }
        )
        output.append(row)
    return output


def build_training_pools(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    holdout_rows: Sequence[Mapping[str, Any]] = (),
    rl_novel_scenarios: int = 32,
    rl_active_cap: int = 16,
    seed: int = 20260815,
) -> dict[str, list[dict[str, Any]]]:
    """Partition every train task exactly once into SFT, active RL, or RL reserve."""

    if rl_active_cap < 1:
        raise ValueError("rl_active_cap must be positive")
    scenarios = _group_scenarios(train_rows)
    if not scenarios:
        raise ValueError("training manifest is empty")
    if not 0 <= rl_novel_scenarios <= len(scenarios):
        raise ValueError("rl_novel_scenarios is outside the scenario population")

    holdout_task_ids = {str(row["task_id"]) for row in holdout_rows}
    holdout_scenario_ids = {str(row["scenario_id"]) for row in holdout_rows}
    train_task_ids = {str(row["task_id"]) for row in train_rows}
    if train_task_ids & holdout_task_ids:
        leaked = sorted(train_task_ids & holdout_task_ids)
        raise ValueError(f"train/holdout task leakage: {leaked[:5]}")
    if set(scenarios) & holdout_scenario_ids:
        leaked = sorted(set(scenarios) & holdout_scenario_ids)
        raise ValueError(f"train/holdout scenario leakage: {leaked[:5]}")

    novel_ids = _select_rl_novel_scenarios(
        scenarios, count=rl_novel_scenarios, seed=seed
    )
    pools = {name: [] for name in POOL_NAMES}
    for scenario_id in sorted(scenarios):
        scenario_rows = scenarios[scenario_id]
        is_novel = scenario_id in novel_ids
        if is_novel:
            active = _rank_rows(
                scenario_rows, seed=seed, label=f"rl_novel_active:{scenario_id}"
            )
            active, reserve = active[:rl_active_cap], active[rl_active_cap:]
            pools["online_rl_active_novel"].extend(
                _annotate_pool_rows(
                    active,
                    pool_name="online_rl_active_novel",
                    scenario_total_tasks=len(scenario_rows),
                    sft_visible=False,
                    rl_novel_scenario=True,
                )
            )
        else:
            sft_order = _rank_rows(
                scenario_rows, seed=seed, label=f"teacher_sft:{scenario_id}"
            )
            quota = _sft_quota(len(scenario_rows))
            sft_rows = sft_order[:quota]
            sft_task_ids = {row["task_id"] for row in sft_rows}
            rl_candidates = _rank_rows(
                [row for row in scenario_rows if row["task_id"] not in sft_task_ids],
                seed=seed,
                label=f"rl_shared_active:{scenario_id}",
            )
            active, reserve = rl_candidates[:rl_active_cap], rl_candidates[rl_active_cap:]
            pools["teacher_sft"].extend(
                _annotate_pool_rows(
                    sft_rows,
                    pool_name="teacher_sft",
                    scenario_total_tasks=len(scenario_rows),
                    sft_visible=True,
                    rl_novel_scenario=False,
                )
            )
            pools["online_rl_active_shared"].extend(
                _annotate_pool_rows(
                    active,
                    pool_name="online_rl_active_shared",
                    scenario_total_tasks=len(scenario_rows),
                    sft_visible=True,
                    rl_novel_scenario=False,
                )
            )
        pools["online_rl_reserve"].extend(
            _annotate_pool_rows(
                reserve,
                pool_name="online_rl_reserve",
                scenario_total_tasks=len(scenario_rows),
                sft_visible=not is_novel,
                rl_novel_scenario=is_novel,
            )
        )

    validate_training_pools(
        pools,
        expected_task_ids=train_task_ids,
        holdout_task_ids=holdout_task_ids,
        holdout_scenario_ids=holdout_scenario_ids,
        rl_active_cap=rl_active_cap,
    )
    return pools


def validate_training_pools(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_task_ids: set[str],
    holdout_task_ids: set[str] | None = None,
    holdout_scenario_ids: set[str] | None = None,
    rl_active_cap: int = 16,
) -> None:
    if set(pools) != set(POOL_NAMES):
        raise ValueError(f"expected pools {POOL_NAMES}, found {sorted(pools)}")
    pool_task_sets: dict[str, set[str]] = {}
    for name in POOL_NAMES:
        ids = [str(row["task_id"]) for row in pools[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate task within pool {name}")
        pool_task_sets[name] = set(ids)
    seen: set[str] = set()
    for name in POOL_NAMES:
        overlap = seen & pool_task_sets[name]
        if overlap:
            raise ValueError(f"task appears in multiple pools: {sorted(overlap)[:5]}")
        seen.update(pool_task_sets[name])
    if seen != expected_task_ids:
        raise ValueError(
            f"pool partition mismatch: missing={len(expected_task_ids - seen)}, "
            f"extra={len(seen - expected_task_ids)}"
        )

    sft_scenarios = {str(row["scenario_id"]) for row in pools["teacher_sft"]}
    novel_scenarios = {
        str(row["scenario_id"]) for row in pools["online_rl_active_novel"]
    }
    novel_scenarios.update(
        str(row["scenario_id"])
        for row in pools["online_rl_reserve"]
        if bool(row.get("rl_novel_scenario"))
    )
    if sft_scenarios & novel_scenarios:
        raise ValueError("an RL-novel scenario is visible to SFT")

    active_counts: Counter[str] = Counter()
    for name in ("online_rl_active_shared", "online_rl_active_novel"):
        active_counts.update(str(row["scenario_id"]) for row in pools[name])
    if active_counts and max(active_counts.values()) > rl_active_cap:
        raise ValueError("RL-active per-scenario cap exceeded")

    all_rows = [row for name in POOL_NAMES for row in pools[name]]
    all_task_ids = {str(row["task_id"]) for row in all_rows}
    all_scenario_ids = {str(row["scenario_id"]) for row in all_rows}
    if all_task_ids & (holdout_task_ids or set()):
        raise ValueError("training pool leaks holdout tasks")
    if all_scenario_ids & (holdout_scenario_ids or set()):
        raise ValueError("training pool leaks holdout scenarios")


def _pool_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(str(row["scenario_id"]) for row in rows)
    return {
        "tasks": len(rows),
        "scenarios": len(scenario_counts),
        "source_tasks": dict(sorted(Counter(str(row["source_split"]) for row in rows).items())),
        "difficulty_tasks": dict(
            sorted(Counter(str(row["difficulty"]) for row in rows).items())
        ),
        "num_apps_tasks": dict(
            sorted(Counter(str(row["num_apps"]) for row in rows).items())
        ),
        "generated_tasks": sum(bool(row.get("generated")) for row in rows),
        "tasks_per_scenario": dict(
            sorted(Counter(map(str, scenario_counts.values())).items(), key=lambda item: int(item[0]))
        ),
    }


def write_training_pool_manifests(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: str | Path,
    datasets_dir: str | Path,
    source_manifest: str | Path,
    holdout_manifests: Sequence[str | Path],
    seed: int,
    rl_novel_scenarios: int,
    rl_active_cap: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    datasets = Path(datasets_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, str] = {}
    dataset_hashes: dict[str, str] = {}

    for pool_name in POOL_NAMES:
        content = _jsonl_bytes(pools[pool_name])
        manifest_path = output / f"{pool_name}.jsonl"
        manifest_path.write_bytes(content)
        manifest_hashes[manifest_path.name] = _sha256(content)

        task_ids = [str(row["task_id"]) for row in pools[pool_name]]
        dataset_content = (("\n".join(task_ids) + "\n") if task_ids else "").encode("utf-8")
        dataset_path = datasets / f"{DATASET_PREFIX}_{pool_name}.txt"
        dataset_path.write_bytes(dataset_content)
        dataset_hashes[dataset_path.name] = _sha256(dataset_content)

    source_path = Path(source_manifest)
    novel_ids = sorted(
        {
            str(row["scenario_id"])
            for name in ("online_rl_active_novel", "online_rl_reserve")
            for row in pools[name]
            if bool(row.get("rl_novel_scenario"))
        }
    )
    all_rows = [row for name in POOL_NAMES for row in pools[name]]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "manifest": str(source_path),
            "manifest_sha256": _sha256(source_path.read_bytes()),
            "tasks": len(all_rows),
            "scenarios": len({str(row["scenario_id"]) for row in all_rows}),
            "holdout_manifests": {
                str(Path(path)): _sha256(Path(path).read_bytes()) for path in holdout_manifests
            },
        },
        "selection": {
            "seed": seed,
            "unit": "scenario_then_task",
            "task_order": "sha256",
            "rl_novel_scenarios": rl_novel_scenarios,
            "rl_novel_scenario_ids": novel_ids,
            "rl_novel_strata": [
                "difficulty",
                "num_apps",
                "source_split",
                "has_generated_tasks",
                "capacity_bucket",
            ],
            "sft_quota": "1 when n<=3, otherwise min(8, max(2, round(sqrt(n))))",
            "rl_active_cap_per_scenario": rl_active_cap,
            "rl_sampling": "uniform scenario then uniform task within scenario",
        },
        "pools": {name: _pool_summary(pools[name]) for name in POOL_NAMES},
        "manifest_sha256": manifest_hashes,
        "appworld_dataset_sha256": dataset_hashes,
        "invariants": {
            "all_source_tasks_partitioned_exactly_once": True,
            "sft_and_all_rl_task_disjoint": True,
            "sft_and_rl_novel_scenario_disjoint": True,
            "validation_final_task_and_scenario_disjoint": True,
            "rl_active_cap_satisfied": True,
        },
    }
    metadata_path = output / "training_pools_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_training_pool_manifests(
    *,
    appworld_root: str | Path = Path("artifacts/appworld_rl_v1_root"),
    manifest_dir: str | Path = Path("data/manifests/appworld_rl_v1"),
    output_dir: str | Path | None = None,
    rl_novel_scenarios: int = 32,
    rl_active_cap: int = 16,
    seed: int = 20260815,
) -> dict[str, Any]:
    root = Path(appworld_root).resolve()
    manifests = Path(manifest_dir)
    if not manifests.is_absolute():
        manifests = root / manifests
    destination = Path(output_dir) if output_dir is not None else manifests
    if not destination.is_absolute():
        destination = root / destination

    source_manifest = manifests / "train_expanded.jsonl"
    holdout_manifests = [manifests / "validation.jsonl", manifests / "final.jsonl"]
    for path in [source_manifest, *holdout_manifests]:
        if not path.is_file():
            raise FileNotFoundError(f"required manifest does not exist: {path}")
    train_rows = _read_jsonl(source_manifest)
    holdout_rows = [row for path in holdout_manifests for row in _read_jsonl(path)]
    pools = build_training_pools(
        train_rows,
        holdout_rows=holdout_rows,
        rl_novel_scenarios=rl_novel_scenarios,
        rl_active_cap=rl_active_cap,
        seed=seed,
    )
    return write_training_pool_manifests(
        pools,
        output_dir=destination,
        datasets_dir=root / "data" / "datasets",
        source_manifest=source_manifest,
        holdout_manifests=holdout_manifests,
        seed=seed,
        rl_novel_scenarios=rl_novel_scenarios,
        rl_active_cap=rl_active_cap,
    )
