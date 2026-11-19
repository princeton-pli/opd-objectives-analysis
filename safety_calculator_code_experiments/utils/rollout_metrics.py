from __future__ import annotations

from typing import Any, Sequence

from utils.calculator_metrics import (
    CALCULATOR_ROLLOUT_COUNT_NAMES,
    calculator_metrics_from_counts,
    compute_calculator_metric_counts,
)
from utils.completion_text import prompt_prefills_thinking


def compute_rollout_metric_counts(
    completions: Sequence[str],
    prompt_texts: Sequence[str],
    metadata: dict[str, Sequence[Any]],
) -> dict[str, int]:
    """Dispatch one generated probe to the same protocol metrics used by OPD."""
    thinking_prefilled = any(prompt_prefills_thinking(prompt) for prompt in prompt_texts)

    if "gold_answer" in metadata:
        return compute_calculator_metric_counts(
            completions,
            metadata["gold_answer"],
            thinking_prefilled=thinking_prefilled,
        )
    return {}


def rollout_count_names(counts: dict[str, int]) -> tuple[str, ...]:
    if counts.get("calculator_total", 0):
        return CALCULATOR_ROLLOUT_COUNT_NAMES
    return ()


def rollout_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    if counts.get("calculator_total", 0):
        return calculator_metrics_from_counts(counts)
    return {}
