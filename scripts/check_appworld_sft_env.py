"""Validate a new server before launching AppWorld SFT."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--verl-root", type=Path)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    if args.verl_root:
        sys.path.insert(0, str(args.verl_root.resolve()))

    failures: list[str] = []
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {_version('torch')}")
    print(f"transformers: {_version('transformers')}")
    print(f"verl: {_version('verl')}")
    print(f"pandas: {_version('pandas')}, pyarrow: {_version('pyarrow')}")

    try:
        import torch

        print(f"torch.cuda: {torch.version.cuda}, available={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable")
        else:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            print(f"bf16_supported: {bf16_supported}")
            if not bf16_supported:
                failures.append("GPU does not support bfloat16, but the launcher is configured for bf16")
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                memory_gib = props.total_memory / (1024**3)
                print(f"gpu[{index}]: {props.name}, {memory_gib:.1f} GiB, cc={props.major}.{props.minor}")
    except ImportError as exc:
        failures.append(f"torch import failed: {exc}")

    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        print(f"model_type: {getattr(config, 'model_type', 'unknown')}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if not tokenizer.chat_template:
            failures.append("tokenizer has no chat_template")
        else:
            print(f"tokenizer: {tokenizer.__class__.__name__}, chat_template=present")
    except Exception as exc:
        failures.append(f"model/tokenizer compatibility check failed: {type(exc).__name__}: {exc}")
        tokenizer = None

    if tokenizer is not None:
        try:
            from appworld_agentic_rl.verl_sft_dataset import AppWorldMultiTurnSFTDataset

            dataset = AppWorldMultiTurnSFTDataset(
                str(args.data),
                tokenizer,
                config={
                    "pad_mode": "no_padding",
                    "max_length": args.max_length,
                    "truncation": "left",
                    "multiturn": {
                        "messages_key": "messages",
                        "tools_key": "tools",
                        "enable_thinking_key": "enable_thinking",
                    },
                },
                max_samples=1,
            )
            sample = dataset[0]
            decoded = tokenizer.decode(sample["input_ids"].tolist())
            print(f"dataset: {len(dataset)} rows loaded; sample_tokens={len(sample['input_ids'])}")
            print(f"sample_loss_tokens: {int(sample['loss_mask'].sum())}")
            if int(sample["loss_mask"].sum()) == 0:
                failures.append("sample has zero loss tokens")
            if "<function=complete_task>" not in decoded and "<function=finish_without_completion>" not in decoded:
                failures.append("sample has no terminal tool action after preprocessing")
        except Exception as exc:
            failures.append(f"verl dataset smoke test failed: {type(exc).__name__}: {exc}")

    if failures:
        print("\nNOT READY")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nREADY FOR SFT SMOKE RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
