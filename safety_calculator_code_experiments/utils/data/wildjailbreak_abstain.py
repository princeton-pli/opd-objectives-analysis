from __future__ import annotations

import argparse
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset

from utils.prompts import REFUSAL_PROMPT_TEMPLATES, build_refusal_user_prompt
from utils import PLACEHOLDER_ASSISTANT_CONTENT, WILDJAILBREAK_DATA_TYPES


def _load_wildjailbreak(config: str, cache_dir: str | None = None) -> Dataset:
    ds = load_dataset(
        "allenai/wildjailbreak",
        config,
        delimiter="\t",
        keep_default_na=False,
        cache_dir=cache_dir,
    )
    return ds["train"]


def _to_common_schema(example: dict, template: str) -> dict:
    data_type = str(example["data_type"])
    prompt_style, harm_label = data_type.split("_")
    request_column = "adversarial" if prompt_style == "adversarial" else "vanilla"
    request = str(example[request_column]).strip()

    return {
        "messages": [
            {"role": "user", "content": build_refusal_user_prompt(request, template)},
            {"role": "assistant", "content": PLACEHOLDER_ASSISTANT_CONTENT},
        ],
        "data_type": data_type,
        "should_abstain": harm_label == "harmful",
        "prompt_style": prompt_style,
        "source": "wildjailbreak",
    }


def _has_usable_request(example: dict) -> bool:
    data_type = str(example["data_type"])
    if data_type not in WILDJAILBREAK_DATA_TYPES:
        return False
    column = "adversarial" if data_type.startswith("adversarial") else "vanilla"
    return bool(str(example.get(column, "")).strip())


def build_wildjailbreak_dataset(
    *,
    template: str,
    train_examples_per_type: int = -1,
    validation_examples_per_type: int = 250,
    cache_dir: str | None = None,
    seed: int = 42,
) -> DatasetDict:
    if validation_examples_per_type < 1:
        raise ValueError(
            f"validation_examples_per_type must be positive, got {validation_examples_per_type}."
        )
    if train_examples_per_type != -1 and train_examples_per_type < 1:
        raise ValueError(
            f"train_examples_per_type must be -1 or positive, got {train_examples_per_type}."
        )

    pool = _load_wildjailbreak("train", cache_dir=cache_dir)
    pool = pool.filter(_has_usable_request, load_from_cache_file=False)
    pool = pool.shuffle(seed=seed)

    indices_by_type: dict[str, list[int]] = defaultdict(list)
    for index, data_type in enumerate(pool["data_type"]):
        indices_by_type[data_type].append(index)

    validation_indices: list[int] = []
    train_indices: list[int] = []
    for data_type in WILDJAILBREAK_DATA_TYPES:
        type_indices = indices_by_type[data_type]
        if len(type_indices) <= validation_examples_per_type:
            raise ValueError(
                f"Not enough {data_type!r} examples ({len(type_indices)}) for a "
                f"{validation_examples_per_type}-example validation slice."
            )
        validation_indices.extend(type_indices[:validation_examples_per_type])
        remaining = type_indices[validation_examples_per_type:]
        if train_examples_per_type != -1:
            remaining = remaining[:train_examples_per_type]
        train_indices.extend(remaining)

    def _map_split(split: Dataset) -> Dataset:
        return split.map(
            _to_common_schema,
            fn_kwargs={"template": template},
            remove_columns=split.column_names,
            load_from_cache_file=False,
        )

    return DatasetDict(
        train=_map_split(pool.select(train_indices)).shuffle(seed=seed),
        validation=_map_split(pool.select(validation_indices)).shuffle(seed=seed + 1),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True, help="Directory to save_to_disk() the built DatasetDict.")
    parser.add_argument(
        "--template",
        required=True,
        choices=sorted(REFUSAL_PROMPT_TEMPLATES),
        help="Prompt template around each request: qwen3 (wildjailbreak_v4) or gemma3 (wildjailbreak_v4_gemma).",
    )
    parser.add_argument(
        "--train_examples_per_type",
        type=int,
        default=-1,
        help="Cap the training examples kept per data_type (-1 keeps everything).",
    )
    parser.add_argument(
        "--validation_examples_per_type",
        type=int,
        default=250,
        help="Held-out examples per data_type carved from the train config.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset = build_wildjailbreak_dataset(
        template=args.template,
        train_examples_per_type=args.train_examples_per_type,
        validation_examples_per_type=args.validation_examples_per_type,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    dataset.save_to_disk(args.output_dir)
    n_total = sum(len(d) for d in dataset.values())
    print(
        f"Saved {n_total} examples ({ {k: len(v) for k, v in dataset.items()} }) to {args.output_dir}"
    )
