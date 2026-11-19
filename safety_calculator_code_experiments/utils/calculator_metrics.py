from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from utils import CALCULATOR_PREFIX, CALCULATOR_ROLLOUT_COUNT_NAMES

from .completion_text import extract_final_response

_NUMBER_ATOM_PATTERN = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_NUMBER_PATTERN = rf"{_NUMBER_ATOM_PATTERN}(?:/{_NUMBER_ATOM_PATTERN})?"
_FINAL_NUMBER_PATTERN = (
    rf"{_NUMBER_ATOM_PATTERN}(?:\s*/\s*{_NUMBER_ATOM_PATTERN})?"
)
_FINAL_ANSWER_RE = re.compile(
    rf"FINAL:\s*({_FINAL_NUMBER_PATTERN})\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(_NUMBER_PATTERN)


@dataclass(frozen=True)
class ParsedCalculatorCompletion:
    action: str
    answer: Fraction | None = None
    expression: str | None = None
    calculator_value: Fraction | None = None
    valid_calculator_call: bool = False


def parse_number(text: str) -> Fraction | None:
    normalized = str(text).strip().replace(",", "").replace("−", "-")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    if not _NUMBER_RE.fullmatch(normalized):
        return None
    pieces = normalized.split("/")
    try:
        numerator = Fraction(pieces[0])
        if len(pieces) == 1:
            return numerator
        denominator = Fraction(pieces[1])
        if denominator == 0:
            return None
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return None


def _evaluate_arithmetic_node(node: ast.AST, source: str, depth: int = 0) -> Fraction:
    if depth > 16:
        raise ValueError("calculator expression is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        literal = ast.get_source_segment(source, node)
        value = parse_number(literal or "")
        if value is None:
            raise ValueError("calculator constants must be ordinary numbers")
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_arithmetic_node(node.operand, source, depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_arithmetic_node(node.left, source, depth + 1)
        right = _evaluate_arithmetic_node(node.right, source, depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("calculator division by zero")
            return left / right
    raise ValueError("calculator supports only numbers, parentheses, +, -, *, and /")


def execute_calculator_expression(expression: str) -> Fraction:
    """Safely evaluate an exact arithmetic expression without using eval()."""
    normalized = (
        expression.strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    if not re.fullmatch(r"[\d\s+\-*/().]+", normalized):
        raise ValueError("calculator expression contains unsupported characters")
    parsed = ast.parse(normalized, mode="eval")
    if sum(1 for _ in ast.walk(parsed)) > 64:
        raise ValueError("calculator expression is too large")
    return _evaluate_arithmetic_node(parsed.body, normalized)


def parse_calculator_completion(
    text: str,
    *,
    thinking_prefilled: bool = False,
) -> ParsedCalculatorCompletion:
    final_response = extract_final_response(text, thinking_prefilled=thinking_prefilled)
    if final_response is None:
        return ParsedCalculatorCompletion(action="invalid")
    stripped = final_response
    # A call wrapped in backticks, as the system prompt writes it, still counts as a call.
    call = stripped[1:-1].strip() if len(stripped) > 1 and stripped[0] == stripped[-1] == "`" else stripped
    if call.startswith(CALCULATOR_PREFIX):
        expression = call[len(CALCULATOR_PREFIX) :].strip()
        # The tool action is terminal: any additional line is malformed.
        if not expression or "\n" in expression or "\r" in expression:
            return ParsedCalculatorCompletion(action="calculator", expression=expression)
        try:
            value = execute_calculator_expression(expression)
        except (SyntaxError, ValueError):
            return ParsedCalculatorCompletion(action="calculator", expression=expression)
        return ParsedCalculatorCompletion(
            action="calculator",
            expression=expression,
            calculator_value=value,
            valid_calculator_call=True,
        )

    if CALCULATOR_PREFIX not in stripped:
        matches = _FINAL_ANSWER_RE.findall(stripped)
        answer = parse_number(matches[-1]) if matches else None
        if answer is not None:
            return ParsedCalculatorCompletion(action="direct", answer=answer)

    return ParsedCalculatorCompletion(action="invalid")


def compute_calculator_metric_counts(
    completions: Sequence[str],
    gold_answers: Sequence[str | int],
    *,
    thinking_prefilled: bool = False,
) -> dict[str, int]:
    if len(completions) != len(gold_answers):
        raise ValueError("completions and gold_answers must have the same length")
    if not completions:
        return {}

    counts = {name: 0 for name in CALCULATOR_ROLLOUT_COUNT_NAMES}
    counts["calculator_total"] = len(completions)

    for index, (completion, raw_gold) in enumerate(zip(completions, gold_answers, strict=True)):
        gold = parse_number(str(raw_gold))
        if gold is None:
            raise ValueError(f"gold answer must be an exact number, got {raw_gold!r}")

        parsed = parse_calculator_completion(
            completion,
            thinking_prefilled=thinking_prefilled,
        )
        is_correct = False
        if parsed.action == "calculator":
            counts["calculator"] += 1
            if parsed.valid_calculator_call:
                if parsed.calculator_value == gold:
                    is_correct = True
        elif parsed.action == "direct":
            if parsed.answer == gold:
                is_correct = True

        if is_correct:
            counts["correct"] += 1

    return counts


def calculator_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = counts.get("calculator_total", 0)
    if total == 0:
        return {}

    metrics = {
        "accuracy": counts.get("correct", 0) / total,
        "calculator_rate": counts.get("calculator", 0) / total,
    }

    return metrics


def compute_calculator_metrics(
    completions: Sequence[str],
    gold_answers: Sequence[str | int],
    *,
    thinking_prefilled: bool = False,
) -> dict[str, float]:
    counts = compute_calculator_metric_counts(
        completions,
        gold_answers,
        thinking_prefilled=thinking_prefilled,
    )
    return calculator_metrics_from_counts(counts)
