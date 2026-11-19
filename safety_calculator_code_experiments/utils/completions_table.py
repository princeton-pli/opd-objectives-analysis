from __future__ import annotations

import random
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def print_opd_completions_table(
    prompts: list[str],
    completions: list[str],
    answer_letters: list[str] | None,
    step: int,
    num_samples: int | None = None,
    max_completion_chars: int = 2000,
    *,
    protocol: Literal["calculator"],
    thinking_prefilled: bool = False,
) -> None:

    n = len(prompts)
    if n == 0:
        return

    if answer_letters is None:
        answer_letters = [None] * n

    if num_samples is not None and 0 < num_samples < n:
        indices = random.sample(range(n), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        answer_letters = [answer_letters[i] for i in indices]
    console = Console(width=200, height=50)

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Prompt", ratio=2)
    table.add_column("Completion", ratio=3, no_wrap=False)
    table.add_column("Action", ratio=1, justify="center")
    table.add_column("Result", ratio=1, justify="center")

    for prompt, completion, gold in zip(prompts, completions, answer_letters):
        if len(completion) > max_completion_chars:
            completion_display = completion[:max_completion_chars] + "\n[…truncated]"
        else:
            completion_display = completion

        if protocol == "calculator":
            from utils.calculator_metrics import parse_calculator_completion, parse_number

            parsed_calculator = parse_calculator_completion(
                completion,
                thinking_prefilled=thinking_prefilled,
            )
            if parsed_calculator.action == "calculator":
                action_cell = Text(
                    "calculator"
                    if parsed_calculator.valid_calculator_call
                    else "calculator (malformed)"
                )
                predicted = parsed_calculator.calculator_value
            elif parsed_calculator.action == "direct":
                action_cell = Text("direct")
                predicted = parsed_calculator.answer
            else:
                action_cell = Text("invalid")
                predicted = None

            if gold is not None:
                gold_value = parse_number(str(gold))
                is_correct = (
                    predicted is not None
                    and gold_value is not None
                    and predicted == gold_value
                )
                result_cell = Text(
                    "correct"
                    if is_correct
                    else f"wrong  pred={predicted} gold={gold}"
                )
            else:
                result_cell = Text("?")
        else:
            raise ValueError(f"Unknown completion-table protocol: {protocol!r}")

        table.add_row(Text(prompt), Text(completion_display), action_cell, result_cell)
        table.add_section()

    panel = Panel(table, expand=True, title=f"Step {step}")
    console.print(panel)
