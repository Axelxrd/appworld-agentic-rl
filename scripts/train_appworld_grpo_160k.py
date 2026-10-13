"""Launch the canonical AppWorld 160K full-parameter GRPO recipe."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # The agent-loop YAML is intentionally project-relative so public configs
    # do not embed a machine-specific checkout path.
    os.chdir(ROOT)
    model = os.environ.get(
        "APPWORLD_MODEL",
        str(ROOT / "models" / "Qwen3.5-4B-sft-final"),
    )
    train = os.environ.get(
        "APPWORLD_GRPO_TRAIN",
        str(ROOT / "artifacts" / "appworld_grpo" / "train_custom.parquet"),
    )
    val = os.environ.get(
        "APPWORLD_GRPO_VAL",
        str(ROOT / "artifacts" / "appworld_grpo" / "dev_custom.parquet"),
    )
    output = os.environ.get(
        "APPWORLD_GRPO_OUTPUT",
        str(ROOT / "artifacts" / "appworld_grpo" / "checkpoints"),
    )
    config_name = os.environ.get(
        "APPWORLD_GRPO_CONFIG", "appworld_grpo_lightweight_event_credit_160k"
    )

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={ROOT / 'configs'}",
        f"--config-name={config_name}",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.model.tokenizer_path={model}",
        f"data.train_files={train}",
        f"data.val_files={val}",
        f"trainer.default_local_dir={output}",
    ]

    # Keep checkpoint resumption explicit at launch time.  This avoids
    # silently loading the newest directory and allows a stopped run to be
    # resumed from an operator-selected global_step folder.
    resume_mode = os.environ.get("APPWORLD_RESUME_MODE")
    resume_from_path = os.environ.get("APPWORLD_RESUME_FROM_PATH")
    if resume_mode:
        cmd.append(f"trainer.resume_mode={resume_mode}")
    if resume_from_path:
        cmd.append(f"trainer.resume_from_path={resume_from_path}")

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
