from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from boundary import candidate_id, read_jsonl

PROMPT_KEY_FIELDS = ("api", "task_name", "variant_index")


def api_expression(row: dict[str, Any]) -> str:
    return row["assignment_prefix"].split("=", 1)[1].strip() + row["continuation"]


def dataset_row(prompt_row: dict[str, Any], example_id: str, *, split: str) -> dict[str, Any]:
    target_continuation = prompt_row["continuation"] + prompt_row["call_suffix"]
    return {
        "prompt": prompt_row["prompt"],
        "completion": target_continuation,
        "example_id": example_id,
        "source": "new_api_v2",
        "prompt_style": "authored_syntactic_boundary",
        "split_role": split,
        "new_api_task": prompt_row["task_name"],
        "new_api": api_expression(prompt_row),
        "new_api_boundary_kind": prompt_row["boundary_kind"],
        "new_api_continuation": prompt_row["continuation"],
        "new_api_target_continuation": target_continuation,
        "new_api_behavioral_evaluation": False,
        "new_api_expected_json": None,
        "new_api_result_variable": "result",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_rows = read_jsonl(args.prompts)
    validation = [
        dataset_row(row, candidate_id(row), split="cohort")
        for row in sorted(prompt_rows, key=lambda row: tuple(row[f] for f in PROMPT_KEY_FIELDS))
    ]
    train = [
        dataset_row(row, f"train::{candidate_id(row)}", split="train")
        for row in prompt_rows
    ]
    for split_name, rows in (("train", train), ("validation", validation)):
        ids = [row["example_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Duplicate example_id in {split_name} split")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(
        train=Dataset.from_list(train), validation=Dataset.from_list(validation)
    ).save_to_disk(str(args.output_dir))

    print(f"train {len(train)} rows | validation {len(validation)} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
