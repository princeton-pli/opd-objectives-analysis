from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from datasets import Dataset, DatasetDict

from boundary import read_jsonl


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "completion": row["continuation"] + row["call_suffix"] + "\n",
        "api": row["api"],
        "lib": row["lib"],
        "task_name": row["task_name"],
        "variant_index": row["variant_index"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_rows = read_jsonl(args.prompts)
    rows = [sft_row(row) for row in prompt_rows]

    train = sorted(rows, key=lambda row: (row["api"], row["variant_index"]))
    apis = len({row["api"] for row in train})
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train)}).save_to_disk(str(args.output_dir))

    print(f"train {len(train)} rows | {apis} APIs -> {args.output_dir}")
    print("\nfirst example:")
    print(f"  prompt (last 60 chars): ...{train[0]['prompt'][-60:]!r}")
    print(f"  completion:             {train[0]['completion']!r}")


if __name__ == "__main__":
    main()
