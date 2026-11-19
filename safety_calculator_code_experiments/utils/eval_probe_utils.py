from __future__ import annotations

import numpy as np
from datasets import Dataset


def select_eval_subset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    """Select a fixed probe without consuming any process-global RNG state."""
    pool_size = len(dataset)
    if max_samples == 0:
        indices: list[int] = []
    elif max_samples == -1 or max_samples >= pool_size:
        indices = list(range(pool_size))
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(pool_size)[:max_samples].tolist()

    return dataset.select(indices)


def add_eval_probe(
    eval_datasets: dict[str, Dataset],
    *,
    name: str,
    pool: Dataset,
    max_samples: int,
    seed: int,
) -> None:
    """Add one deterministic probe unless it is disabled."""
    if max_samples == 0:
        return

    probe = select_eval_subset(pool, max_samples=max_samples, seed=seed)
    if len(probe) == 0:
        raise ValueError(
            f"The {name!r} evaluation pool is empty after filtering; disable it with the corresponding "
            "max_*_eval_samples=0 option or adjust the filters."
        )

    probe = probe.add_column("eval_example_index", list(range(len(probe))))
    eval_datasets[name] = probe


def build_eval_probes(
    *,
    train_pool: Dataset,
    max_train_samples: int,
    test_pool: Dataset | None,
    max_test_samples: int,
    data_seed: int,
) -> dict[str, Dataset]:
    """Build deterministic named train/test probes."""
    eval_datasets: dict[str, Dataset] = {}

    add_eval_probe(
        eval_datasets,
        name="train",
        pool=train_pool,
        max_samples=max_train_samples,
        seed=data_seed,
    )

    if max_test_samples != 0:
        add_eval_probe(
            eval_datasets,
            name="test",
            pool=test_pool,
            max_samples=max_test_samples,
            seed=data_seed,
        )

    return eval_datasets
