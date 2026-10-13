"""Convert AppWorld SFT JSONL rows to the parquet input expected by verl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    rows = _read_jsonl(args.input)
    # Keep heterogeneous nested tool arguments losslessly. Arrow otherwise
    # promotes dicts with different keys to one wide struct with null fields.
    for row in rows:
        for key in ("messages", "tools"):
            if key in row:
                row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
