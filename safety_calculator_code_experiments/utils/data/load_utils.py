from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


@dataclass
class DataSubsetArguments:
    max_train_samples: int = field(default=-1)
    max_train_eval_samples: int = field(
        default=0,
        metadata={"help": "Fixed train-probe size (0 disables, -1 uses the full filtered training pool)."},
    )
    max_test_eval_samples: int = field(
        default=-1,
        metadata={"help": "Fixed test-probe size (0 disables, -1 uses the full filtered test pool)."},
    )
    max_prompt_tokens: int = field(default=-1)
    max_response_tokens: int = field(default=-1)

    def __post_init__(self) -> None:
        for name in ("max_train_samples", "max_train_eval_samples", "max_test_eval_samples"):
            value = getattr(self, name)
            if value < -1:
                raise ValueError(f"{name} must be -1 or nonnegative, got {value}.")


@dataclass
class CacheArguments:
    cache_dir: str | None = field(default=None)


def select_train_subset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples == -1 or max_samples >= len(dataset):
        return dataset
    subset = dataset.shuffle(seed=seed).select(range(max_samples))
    logger.info(
        "Dataset subsampled: %d → %d examples (data_seed=%d)",
        len(dataset),
        len(subset),
        seed,
    )
    return subset


def filter_by_length(
    dataset: Dataset,
    tokenizer,
    max_prompt_tokens: int = -1,
    max_response_tokens: int = -1,
    messages_key: str = "messages",
    prompt_key: str = "prompt",
) -> Dataset:
    if max_prompt_tokens == -1 and max_response_tokens == -1:
        return dataset

    has_messages = messages_key in dataset.column_names

    def _keep(example: dict) -> bool:
        if has_messages:
            msgs = example[messages_key]
            if max_prompt_tokens != -1:
                n_prompt_tokens = sum(
                    len(tokenizer(m["content"], add_special_tokens=False)["input_ids"])
                    for m in msgs[:-1]
                    if m["role"] == "user"
                )
                if n_prompt_tokens > max_prompt_tokens:
                    return False
            if max_response_tokens != -1:
                ids = tokenizer(msgs[-1]["content"], add_special_tokens=False)["input_ids"]
                if len(ids) > max_response_tokens:
                    return False
        else:
            if max_prompt_tokens != -1:
                ids = tokenizer(example[prompt_key])["input_ids"]
                if len(ids) > max_prompt_tokens:
                    return False
        return True

    n_before = len(dataset)
    filtered = dataset.filter(_keep, desc="Filtering by length")
    n_after = len(filtered)
    logger.info(
        "Length filter: %d → %d examples (dropped %d, %.2f%%; "
        "max_prompt_tokens=%s, max_response_tokens=%s)",
        n_before,
        n_after,
        n_before - n_after,
        100 * (n_before - n_after) / n_before if n_before else 0.0,
        max_prompt_tokens if max_prompt_tokens != -1 else "∞",
        max_response_tokens if max_response_tokens != -1 else "∞",
    )
    return filtered


def save_run_config(output_dir: str, **configs: Any) -> None:
    def _to_dict(obj: Any) -> dict:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return vars(obj)

    combined = {name: _to_dict(cfg) for name, cfg in configs.items()}
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Run config saved to %s", config_path)


def load_maybe_local_dataset(
    name_or_path: str, config: str | None = None, cache_dir: str | None = None
) -> DatasetDict:
    if os.path.isdir(name_or_path):
        return load_from_disk(name_or_path)
    return load_dataset(name_or_path, name=config, cache_dir=cache_dir)
