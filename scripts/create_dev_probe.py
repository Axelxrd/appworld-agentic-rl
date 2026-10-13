"""Create a deterministic, scenario-complete AppWorld dev probe manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from appworld_agentic_rl.splits import _select_stratified, load_official_scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.scenarios <= 0:
        raise ValueError("--scenarios must be positive")

    source = load_official_scenarios(args.appworld_root)
    selected, _ = _select_stratified(
        source["dev"], count=args.scenarios, seed=args.seed, label="sft-dev-probe"
    )
    rows: list[dict[str, object]] = []
    for scenario in sorted(selected, key=lambda item: item.scenario_id):
        for task_id in scenario.task_ids:
            rows.append(
                {
                    "task_id": task_id,
                    "scenario_id": scenario.scenario_id,
                    "source_split": "dev",
                    "difficulty": scenario.difficulty,
                    "num_apps": scenario.num_apps,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    args.output.write_bytes(content)
    metadata = {
        "schema_version": "appworld-dev-probe-v1",
        "source_split": "dev",
        "seed": args.seed,
        "scenario_count": len(selected),
        "task_count": len(rows),
        "scenario_ids": [scenario.scenario_id for scenario in sorted(selected, key=lambda item: item.scenario_id)],
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
    }
    args.output.with_name("metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
