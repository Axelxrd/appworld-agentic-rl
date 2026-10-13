"""Isolated, resumable expansion of official AppWorld train scenarios."""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence


GENERATION_SCHEMA_VERSION = "appworld-train-generation-v1"
ROOT_MARKER = ".appworld_rl_generation_root.json"
MAX_RECOVERABLE_SETUP_RETRIES = 32


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _install_winloop_policy_compatibility() -> None:
    if os.name != "nt":
        return
    import winloop

    if hasattr(winloop, "WinLoopPolicy"):
        return

    class WinLoopPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self) -> asyncio.AbstractEventLoop:
            return winloop.new_event_loop()

    winloop.WinLoopPolicy = WinLoopPolicy


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _ensure_interpreter_scripts_on_path() -> None:
    scripts = str(Path(sys.executable).resolve().parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if scripts.lower() not in {entry.lower() for entry in entries}:
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _source_fingerprint(source_root: Path) -> str:
    inputs = [
        source_root / "data" / "version.txt",
        source_root / "data" / "datasets" / "train.txt",
    ]
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_isolated_root(
    source_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output:
        raise ValueError(
            "generation output root must differ from the official AppWorld root"
        )
    marker_path = output / ROOT_MARKER
    fingerprint = _source_fingerprint(source)

    if output.exists() and any(output.iterdir()):
        if not marker_path.exists():
            raise ValueError(
                f"refusing to reuse unmarked non-empty output root: {output}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("source_fingerprint") != fingerprint:
            raise ValueError(
                "generation root was prepared from a different AppWorld data source"
            )
        return marker

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "data", output / "data", dirs_exist_ok=False)
    marker = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "source_root": str(source),
        "source_fingerprint": fingerprint,
        "purpose": "isolated AppWorld-RL train-task generation",
    }
    _write_json(marker_path, marker)
    return marker


def _base_main_user_count(root: Path) -> int:
    connection = sqlite3.connect(root / "data" / "base_dbs" / "admin.db")
    try:
        row = connection.execute("SELECT COUNT(*) FROM main_users").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def _clear_scenario_tasks(tasks_directory: Path, scenario_id: str) -> None:
    resolved_tasks = tasks_directory.resolve()
    for path in tasks_directory.glob(f"{scenario_id}_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix(scenario_id + "_")
        if not suffix.isdigit() or path.resolve().parent != resolved_tasks:
            raise ValueError(f"unsafe generated-task path: {path}")
        shutil.rmtree(path)


def _expected_exhaustion(error: BaseException | None) -> bool:
    text = str(error or "")
    return any(
        marker in text
        for marker in (
            "The setup was not successful for any of the main users",
            "No main users found in the database",
        )
    )


def _recoverable_setup_error(error: BaseException | None) -> bool:
    return "Could not sample enough items" in str(error or "")


def _known_searchable_setup_error(generator_id: str, error: BaseException) -> bool:
    if _recoverable_setup_error(error):
        return True
    return (
        generator_id == "d0b1f43"
        and isinstance(error, UnboundLocalError)
        and "num_relative_transactions" in str(error)
    )


def _make_setup_failures_searchable(generator_class: type) -> None:
    if getattr(generator_class, "_appworld_rl_setup_wrapper", False):
        return
    original_setup = generator_class._setup
    generator_id = generator_class.id()

    def setup_with_searchable_failures(*args: Any, **kwargs: Any):
        from generate.tasks.task_generators.base import Fail

        try:
            yield from original_setup(*args, **kwargs)
        except (UnboundLocalError, ValueError) as error:
            if not _known_searchable_setup_error(generator_id, error):
                raise
            yield Fail(997 if isinstance(error, UnboundLocalError) else 998)

    generator_class._setup = setup_with_searchable_failures
    generator_class._appworld_rl_setup_wrapper = True


def _task_ids_for_scenario(scenario_id: str, count: int) -> list[str]:
    return [f"{scenario_id}_{number}" for number in range(1, count + 1)]


def _generated_scenario_is_complete(
    tasks_directory: Path, scenario_id: str, count: int
) -> bool:
    if count < 3:
        return False
    expected_ids = set(_task_ids_for_scenario(scenario_id, count))
    actual_ids = {
        path.name for path in tasks_directory.glob(f"{scenario_id}_*") if path.is_dir()
    }
    if actual_ids != expected_ids:
        return False
    required_paths = (
        Path("specs.json"),
        Path("ground_truth") / "compiled_solution.py",
        Path("ground_truth") / "evaluation.py",
    )
    return all(
        (tasks_directory / task_id / relative_path).is_file()
        for task_id in expected_ids
        for relative_path in required_paths
    )


def _load_official_train_scenarios(manifest_dir: Path) -> list[str]:
    rows = _read_jsonl(manifest_dir / "train.jsonl")
    return sorted(
        {str(row["scenario_id"]) for row in rows if row.get("source_split") == "train"}
    )


def _load_status(path: Path, base: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return base
    status = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source_root", "output_root", "random_seed", "max_tasks_per_scenario"):
        if status.get(key) != base.get(key):
            raise ValueError(f"generation status mismatch for {key}")
    for result in status.get("scenarios", {}).values():
        if result.get("validated"):
            result.setdefault("generation_complete", True)
            result.setdefault("reused_generated_tasks", False)
            result.setdefault("validation_mode", "environment")
    return status


def _write_expanded_manifests(
    *,
    output_root: Path,
    source_manifest_dir: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    target_dir = output_root / "data" / "manifests" / "appworld_rl_v1"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "final.jsonl", "metadata.json"):
        shutil.copy2(source_manifest_dir / name, target_dir / name)

    train_rows = _read_jsonl(source_manifest_dir / "train.jsonl")
    completed = status.get("scenarios", {})
    expanded_rows: list[dict[str, Any]] = []
    replaced_scenarios: set[str] = set()
    for row in train_rows:
        scenario_id = str(row["scenario_id"])
        result = completed.get(scenario_id)
        if row.get("source_split") == "train" and result and result.get("validated"):
            if scenario_id in replaced_scenarios:
                continue
            replaced_scenarios.add(scenario_id)
            for task_id in result["task_ids"]:
                expanded_rows.append(
                    {
                        **{
                            key: value for key, value in row.items() if key != "task_id"
                        },
                        "task_id": task_id,
                        "generated": True,
                    }
                )
        else:
            expanded_rows.append({**row, "generated": False})

    content = _canonical_jsonl(expanded_rows)
    expanded_path = target_dir / "train_expanded.jsonl"
    expanded_path.write_bytes(content)

    datasets = output_root / "data" / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    dataset_sources = {
        "appworld_rl_v1_train": expanded_rows,
        "appworld_rl_v1_validation": _read_jsonl(
            source_manifest_dir / "validation.jsonl"
        ),
        "appworld_rl_v1_final": _read_jsonl(source_manifest_dir / "final.jsonl"),
    }
    for name, rows in dataset_sources.items():
        (datasets / f"{name}.txt").write_text(
            "\n".join(str(row["task_id"]) for row in rows) + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "completed_original_train_scenarios": len(replaced_scenarios),
        "total_original_train_scenarios": len(
            _load_official_train_scenarios(source_manifest_dir)
        ),
        "expanded_train_tasks": len(expanded_rows),
        "generated_tasks": sum(bool(row["generated"]) for row in expanded_rows),
        "train_expanded_sha256": hashlib.sha256(content).hexdigest(),
    }
    _write_json(target_dir / "generation_summary.json", summary)
    return summary


def generate_original_train_tasks(
    *,
    source_root: str | Path,
    output_root: str | Path,
    appworld_repo: str | Path,
    manifest_dir: str | Path,
    scenario_ids: Sequence[str] | None = None,
    max_tasks_per_scenario: int | None = None,
    random_seed: int = 1,
    validate_with_environment: bool = True,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "set PYTHONHASHSEED=0 before starting deterministic task generation"
        )
    _configure_utf8_stdio()
    _ensure_interpreter_scripts_on_path()

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    repository = Path(appworld_repo).resolve()
    manifests = Path(manifest_dir).resolve()
    prepare_isolated_root(source, output)

    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    os.environ["APPWORLD_ROOT"] = str(output)
    _install_winloop_policy_compatibility()
    from appworld import update_root
    from appworld.apps.lib.models.db import CachedDBHandler
    from appworld.collections.apis import ApiCollection
    from generate.tasks.task_generators.base import BaseTaskGenerator

    update_root(str(output))
    upper_bound = _base_main_user_count(output)
    limit = upper_bound if max_tasks_per_scenario is None else max_tasks_per_scenario
    if limit < 3 or limit > upper_bound:
        raise ValueError(f"max_tasks_per_scenario must be in [3, {upper_bound}]")

    official_scenarios = _load_official_train_scenarios(manifests)
    requested = list(scenario_ids) if scenario_ids else official_scenarios
    unknown = set(requested) - set(official_scenarios)
    if unknown:
        raise ValueError(f"not official train scenarios: {sorted(unknown)}")

    status_path = output / "generation" / "status.json"
    base_status: dict[str, Any] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "source_root": str(source),
        "output_root": str(output),
        "manifest_dir": str(manifests),
        "random_seed": random_seed,
        "max_tasks_per_scenario": limit,
        "base_main_users": upper_bound,
        "official_train_scenarios": official_scenarios,
        "scenarios": {},
    }
    status = _load_status(status_path, base_status)

    for scenario_id in requested:
        previous = status["scenarios"].get(scenario_id)
        if previous and previous.get("validated"):
            continue
        tasks_directory = output / "data" / "tasks"
        previous_count = int((previous or {}).get("generated_tasks", 0))
        reuse_generated_tasks = bool(
            previous
            and not previous.get("validated")
            and previous.get("generation_complete")
            and _generated_scenario_is_complete(
                tasks_directory, scenario_id, previous_count
            )
        )
        generation_error: BaseException | None = None
        if reuse_generated_tasks:
            generated = previous_count
            stop_reason = str(previous.get("stop_reason", "unknown"))
            setup_retries = int(previous.get("recoverable_setup_retries", 0))
        else:
            _clear_scenario_tasks(tasks_directory, scenario_id)
            generator = BaseTaskGenerator.by_id(scenario_id)()
            generated = 0
            stop_reason = "upper_bound"
            setup_retries = 0
            for index in range(limit):
                task_generated = False
                for retry_index in range(MAX_RECOVERABLE_SETUP_RETRIES + 1):
                    attempt_seed = random_seed + retry_index * 1009
                    with _working_directory(repository):
                        suppressor = generator.generate_task(
                            compile_solution=True,
                            skip_rule_checks=index > 0,
                            random_seed=attempt_seed,
                            suppress_errors=True,
                        )
                    if not suppressor.raised_error:
                        generated += 1
                        task_generated = True
                        if generated == 1:
                            _make_setup_failures_searchable(generator.__class__)
                        break
                    generation_error = suppressor.exception
                    if _expected_exhaustion(generation_error):
                        stop_reason = "feasible_tasks_exhausted"
                        break
                    if _recoverable_setup_error(generation_error):
                        setup_retries += 1
                        CachedDBHandler.reset()
                        ApiCollection.close_all()
                        continue
                    stop_reason = "generation_error"
                    break
                if task_generated:
                    generation_error = None
                    continue
                if _recoverable_setup_error(generation_error):
                    stop_reason = "recoverable_setup_retries_exhausted"
                break

        CachedDBHandler.reset()
        ApiCollection.close_all()
        if generated < 3 or (
            generation_error is not None and stop_reason == "generation_error"
        ):
            result = {
                "generated_tasks": generated,
                "task_ids": _task_ids_for_scenario(scenario_id, generated),
                "generation_complete": False,
                "validated": False,
                "stop_reason": stop_reason,
                "recoverable_setup_retries": setup_retries,
                "reused_generated_tasks": reuse_generated_tasks,
                "validation_mode": None,
                "error_type": type(generation_error).__name__
                if generation_error
                else None,
                "error": str(generation_error) if generation_error else None,
            }
            status["scenarios"][scenario_id] = result
            _write_json(status_path, status)
            raise RuntimeError(
                f"generation failed for {scenario_id}: {result['error']}"
            )

        generation_checkpoint = {
            "generated_tasks": generated,
            "task_ids": _task_ids_for_scenario(scenario_id, generated),
            "generation_complete": True,
            "validated": False,
            "stop_reason": stop_reason,
            "recoverable_setup_retries": setup_retries,
            "reused_generated_tasks": reuse_generated_tasks,
            "validation_mode": "environment" if validate_with_environment else "direct",
            "error_type": None,
            "error": None,
        }
        status["scenarios"][scenario_id] = generation_checkpoint
        _write_json(status_path, status)

        validator = BaseTaskGenerator.by_id(scenario_id)()
        with _working_directory(repository):
            validation = validator.validate_generated_tasks(
                num_tasks_per_generator=generated,
                use_environment=validate_with_environment,
                use_compiled_solution=True,
                update_validation_metadata=True,
                suppress_errors=True,
            )
        validated = not validation.raised_error
        result = {
            "generated_tasks": generated,
            "task_ids": _task_ids_for_scenario(scenario_id, generated),
            "generation_complete": True,
            "validated": validated,
            "stop_reason": stop_reason,
            "recoverable_setup_retries": setup_retries,
            "reused_generated_tasks": reuse_generated_tasks,
            "validation_mode": "environment" if validate_with_environment else "direct",
            "error_type": type(validation.exception).__name__
            if validation.exception
            else None,
            "error": str(validation.exception) if validation.exception else None,
        }
        status["scenarios"][scenario_id] = result
        _write_json(status_path, status)
        if not validated:
            raise RuntimeError(
                f"validation failed for {scenario_id}: {result['error']}"
            )

    _write_json(status_path, status)
    summary = _write_expanded_manifests(
        output_root=output,
        source_manifest_dir=manifests,
        status=status,
    )
    return {"status_path": str(status_path), "summary": summary, **status}
