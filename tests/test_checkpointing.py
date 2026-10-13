import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from appworld_agentic_rl.checkpointing import (
    find_huggingface_model,
    prepare_checkpoint_model,
    validate_fsdp_shards,
)


def _write_fsdp_checkpoint(path: Path, world_size: int = 2) -> None:
    path.mkdir(parents=True)
    (path / "fsdp_config.json").write_text(
        json.dumps({"world_size": world_size}), encoding="utf-8"
    )
    for rank in range(world_size):
        (path / f"model_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"shard")


def test_prepare_checkpoint_returns_existing_huggingface_model(tmp_path: Path) -> None:
    model = tmp_path / "global_step_10" / "actor" / "huggingface"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    assert find_huggingface_model(tmp_path / "global_step_10") == model.resolve()
    assert prepare_checkpoint_model(tmp_path / "global_step_10") == model.resolve()


def test_validate_fsdp_shards_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    actor = tmp_path / "global_step_10" / "actor"
    _write_fsdp_checkpoint(actor)
    (actor / "model_world_size_2_rank_1.pt").unlink()

    with pytest.raises(FileNotFoundError, match="incomplete FSDP checkpoint"):
        validate_fsdp_shards(actor)


def test_prepare_checkpoint_merges_fsdp_model(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_20"
    actor = checkpoint / "actor"
    _write_fsdp_checkpoint(actor)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        target = Path(command[command.index("--target_dir") + 1])
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("appworld_agentic_rl.checkpointing.subprocess.run", fake_run)
    result = prepare_checkpoint_model(checkpoint, python_executable="python-test")

    assert result == (actor / "huggingface").resolve()
    assert calls[0][:5] == ["python-test", "-m", "verl.model_merger", "merge", "--backend"]
    assert str(actor.resolve()) in calls[0]
