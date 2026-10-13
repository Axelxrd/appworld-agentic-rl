"""Resolve or merge veRL checkpoints into vLLM-loadable model directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _has_model_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def find_huggingface_model(checkpoint_dir: str | Path) -> Path | None:
    """Return an already merged HuggingFace model below a veRL checkpoint."""

    root = Path(checkpoint_dir).resolve()
    candidates = (
        root / "actor" / "huggingface",
        root / "huggingface",
        root / "actor",
        root,
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file() and _has_model_weights(candidate):
            return candidate
    return None


def find_fsdp_model(checkpoint_dir: str | Path) -> Path | None:
    """Locate the model-only FSDP shard directory in a veRL checkpoint."""

    root = Path(checkpoint_dir).resolve()
    for candidate in (root / "actor", root / "actor" / "model", root):
        if (candidate / "fsdp_config.json").is_file():
            return candidate
    return None


def validate_fsdp_shards(fsdp_dir: str | Path) -> int:
    """Validate a complete model-only veRL FSDP checkpoint and return world size."""

    root = Path(fsdp_dir).resolve()
    config_path = root / "fsdp_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing FSDP config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        world_size = int(config["world_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid world_size in {config_path}") from exc
    if world_size < 1:
        raise ValueError(f"world_size must be positive in {config_path}")
    missing = [
        root / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        if not (root / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "incomplete FSDP checkpoint; missing " + ", ".join(str(path) for path in missing)
        )
    return world_size


def prepare_checkpoint_model(
    checkpoint_dir: str | Path,
    *,
    target_dir: str | Path | None = None,
    python_executable: str | Path = sys.executable,
) -> Path:
    """Return a loadable model, merging model-only FSDP shards when needed."""

    checkpoint = Path(checkpoint_dir).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    existing = find_huggingface_model(checkpoint)
    if existing is not None:
        return existing

    fsdp_dir = find_fsdp_model(checkpoint)
    if fsdp_dir is None:
        raise FileNotFoundError(
            f"checkpoint has neither a merged HuggingFace model nor FSDP shards: {checkpoint}"
        )
    validate_fsdp_shards(fsdp_dir)
    target = (
        Path(target_dir).resolve()
        if target_dir is not None
        else (checkpoint / "actor" / "huggingface").resolve()
    )
    if target.exists():
        raise FileExistsError(f"incomplete merge target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(fsdp_dir),
        "--target_dir",
        str(target),
    ]
    subprocess.run(command, check=True, stdout=sys.stderr)
    if not (target / "config.json").is_file() or not _has_model_weights(target):
        raise RuntimeError(f"veRL model merge did not produce a loadable model: {target}")
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve or merge a veRL checkpoint into a HuggingFace model directory."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = prepare_checkpoint_model(args.checkpoint_dir, target_dir=args.target_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "find_fsdp_model",
    "find_huggingface_model",
    "prepare_checkpoint_model",
    "validate_fsdp_shards",
]
