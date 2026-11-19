from __future__ import annotations

import argparse
import random
import re
from collections.abc import Callable
import numpy as np
from datasets import Dataset, DatasetDict

from mathematics_dataset.modules import modules as official_modules

from utils import PLACEHOLDER_ASSISTANT_CONTENT, OFFICIAL_REPOSITORY, \
    OFFICIAL_COMMIT, OFFICIAL_MAX_QUESTION_LENGTH, OFFICIAL_MAX_ANSWER_LENGTH, DEFAULT_MODULES
from utils.calculator_metrics import parse_number
from utils.prompts import PREFERENCE_TEXT, build_system_prompt

_NUMBER_LITERAL_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d+)?|\.\d+))?(?!\.\d|\w)"
)


def extract_numeric_literals(question: str) -> tuple[str, ...]:
    return tuple(_NUMBER_LITERAL_RE.findall(str(question)))


def normalize_exact_answer(answer: str | int) -> str | None:
    normalized = str(answer).strip().replace(",", "")
    return normalized if parse_number(normalized) is not None else None


def difficulty_from_literals(literals: tuple[str, ...]) -> tuple[str, int]:
    if not literals:
        raise ValueError("at least one numeric literal is required")
    operand_digits = max(len(re.sub(r"\D", "", literal)) for literal in literals)
    if operand_digits <= 2:
        difficulty = "easy"
    elif operand_digits <= 4:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return difficulty, operand_digits


def is_supported_arithmetic_example(example: dict) -> bool:
    return (
        len(str(example["question"])) <= OFFICIAL_MAX_QUESTION_LENGTH
        and len(str(example["answer"])) <= OFFICIAL_MAX_ANSWER_LENGTH
        and len(extract_numeric_literals(example["question"])) >= 2
        and normalize_exact_answer(example["answer"]) is not None
    )


def to_calculator_example(
    example: dict,
    *,
    route_preference: str = "direct",
) -> dict:
    literals = extract_numeric_literals(example["question"])
    gold_answer = normalize_exact_answer(example["answer"])
    if len(literals) < 2 or gold_answer is None:
        raise ValueError("example is not supported exact arithmetic")

    operand_difficulty, operand_digits = difficulty_from_literals(literals)
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(route_preference)},
            {"role": "user", "content": str(example["question"]).strip()},
            {"role": "assistant", "content": PLACEHOLDER_ASSISTANT_CONTENT},
        ],
        "gold_answer": gold_answer,
        "difficulty": str(example["difficulty"]),
        "operand_difficulty": operand_difficulty,
        "operand_digits": operand_digits,
        "num_operands": len(literals),
        "module": str(example["module"]),
        "generator_module": str(example["generator_module"]),
        "source": f"{OFFICIAL_REPOSITORY}@{OFFICIAL_COMMIT}",
    }


def _make_entropy_fn(level: int, num_levels: int = 3):
    """Equivalent to the curriculum helper in the official generate.py."""
    lower = level / num_levels
    upper = (level + 1) / num_levels

    def modify_entropy(range_: tuple[float, float]) -> tuple[float, float]:
        length = range_[1] - range_[0]
        return range_[0] + lower * length, range_[0] + upper * length

    return modify_entropy


def _flatten_modules(
    nested_modules: dict,
    prefix: str | None = None,
) -> dict[str, Callable]:
    flattened = {}
    for name, module_or_group in nested_modules.items():
        full_name = f"{prefix}__{name}" if prefix else name
        if isinstance(module_or_group, dict):
            flattened.update(_flatten_modules(module_or_group, full_name))
        else:
            flattened[full_name] = module_or_group
    return flattened


def _official_regime_modules() -> dict[str, dict[str, Callable]]:
    return {
        "train-easy": _flatten_modules(official_modules.train(_make_entropy_fn(0))),
        "train-medium": _flatten_modules(official_modules.train(_make_entropy_fn(1))),
        "train-hard": _flatten_modules(official_modules.train(_make_entropy_fn(2))),
        "interpolate": _flatten_modules(official_modules.test()),
    }


def _generate_examples(
    regime_modules: dict[str, Callable],
    *,
    regime: str,
    modules: tuple[str, ...],
    examples_per_module: int,
    route_preference: str,
) -> list[dict]:
    rows = []
    for module in modules:
        generator_module = module
        if generator_module not in regime_modules:
            raise ValueError(
                f"Official generator does not expose {generator_module!r} in {regime!r}."
            )
        generator = regime_modules[generator_module]
        accepted = 0
        attempts = 0
        max_attempts = max(100, examples_per_module * 20)
        while accepted < examples_per_module and attempts < max_attempts:
            attempts += 1
            problem = generator()
            raw_example = {
                "question": str(problem.question),
                "answer": str(problem.answer),
                "difficulty": regime,
                "module": module,
                "generator_module": generator_module,
            }
            if not is_supported_arithmetic_example(raw_example):
                continue
            rows.append(
                to_calculator_example(
                    raw_example,
                    route_preference=route_preference,
                )
            )
            accepted += 1
        if accepted != examples_per_module:
            raise RuntimeError(
                f"Generated only {accepted}/{examples_per_module} supported examples for "
                f"{regime}/{generator_module} after {attempts} attempts."
            )
    return rows


def build_deepmind_calculator_dataset(
    *,
    modules: tuple[str, ...] = DEFAULT_MODULES,
    train_examples_per_module: int = 10_000,
    validation_examples_per_module: int = 1_000,
    route_preference: str = "direct",
    seed: int = 42,
) -> DatasetDict:
    unknown_modules = sorted(set(modules) - set(DEFAULT_MODULES))
    if unknown_modules:
        raise ValueError(f"Unsupported arithmetic modules: {unknown_modules}")
    for name, count in (
        ("train_examples_per_module", train_examples_per_module),
        ("validation_examples_per_module", validation_examples_per_module),
    ):
        if count < 1:
            raise ValueError(f"{name} must be positive, got {count}.")

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        regimes = _official_regime_modules()
        per_train_regime, remainder = divmod(train_examples_per_module, 3)
        train_rows = []
        for index, regime in enumerate(("train-easy", "train-medium", "train-hard")):
            count = per_train_regime + int(index < remainder)
            train_rows.extend(
                _generate_examples(
                    regimes[regime],
                    regime=regime,
                    modules=modules,
                    examples_per_module=count,
                    route_preference=route_preference,
                )
            )
        validation_rows = _generate_examples(
            regimes["interpolate"],
            regime="interpolate",
            modules=modules,
            examples_per_module=validation_examples_per_module,
            route_preference=route_preference,
        )
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows).shuffle(
                seed=seed,
                load_from_cache_file=False,
                keep_in_memory=True,
            ),
            "validation": Dataset.from_list(validation_rows).shuffle(
                seed=seed + 1,
                load_from_cache_file=False,
                keep_in_memory=True,
            ),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_examples_per_module", type=int, default=10_000)
    parser.add_argument("--validation_examples_per_module", type=int, default=1_000)
    parser.add_argument(
        "--route_preference",
        choices=tuple(PREFERENCE_TEXT),
        default="direct",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset = build_deepmind_calculator_dataset(
        train_examples_per_module=args.train_examples_per_module,
        validation_examples_per_module=args.validation_examples_per_module,
        route_preference=args.route_preference,
        seed=args.seed,
    )
    dataset.save_to_disk(args.output_dir)
    split_sizes = {name: len(split) for name, split in dataset.items()}
    print(
        f"Saved {sum(len(split) for split in dataset.values())} examples "
        f"({split_sizes}) to {args.output_dir}"
    )
