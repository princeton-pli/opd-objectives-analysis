from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from utils.completion_text import prompt_prefills_thinking
from utils.strings import (
    CHAT_TURN_MARKERS,
    GEMMA_TURN_TOKENS,
    REQUEST_BOUNDARY,
)


CHAT_CONTROL_TOKEN = re.compile(
    r"<\|[^<>]+?\|>|" + "|".join(re.escape(token) for token in GEMMA_TURN_TOKENS)
)


def chat_turn_markers(model_name: str | None) -> tuple[str, str] | None:
    """Resolve the user-turn delimiters for the model family that rendered a prompt."""
    lowered = str(model_name or "").lower()
    for family, markers in CHAT_TURN_MARKERS.items():
        if family in lowered:
            return markers
    return None


def clean_eval_prompt(prompt: str, model_name: str | None = None) -> str:
    markers = chat_turn_markers(model_name)
    if markers is not None:
        user_start, message_end = markers
        if user_start in prompt:
            prompt = prompt.rsplit(user_start, 1)[1]
            if message_end not in prompt:
                raise ValueError(
                    f"User prompt from {model_name!r} is missing its message-end token "
                    f"{message_end!r}"
                )
            prompt = prompt.split(message_end, 1)[0]

    prompt = prompt.strip()
    if REQUEST_BOUNDARY in prompt:
        prompt = prompt.split(REQUEST_BOUNDARY, 1)[1].strip()
    if CHAT_CONTROL_TOKEN.search(prompt):
        raise ValueError(
            "WildJailbreak request still contains chat-template control tokens "
            f"{sorted(set(CHAT_CONTROL_TOKEN.findall(prompt)))} for model {model_name!r}. "
            "Add that model family's user-turn markers to CHAT_TURN_MARKERS."
        )
    if not prompt:
        raise ValueError("WildJailbreak request is empty after prompt cleanup")
    return prompt


def validate_eval_request(request: str) -> str:
    request = request.strip()
    if REQUEST_BOUNDARY in request:
        raise ValueError("WildJailbreak request still contains protocol guidelines")
    if CHAT_CONTROL_TOKEN.search(request):
        raise ValueError("WildJailbreak request contains chat-template control tokens")
    if not request:
        raise ValueError("WildJailbreak request is empty")
    return request


def safe_metric_prefix(metric_key_prefix: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in metric_key_prefix
    )


def save_eval_rollout_records(
    *,
    output_dir: str | os.PathLike[str],
    metric_key_prefix: str,
    step: int,
    records: Iterable[dict[str, Any]],
    model_name: str | None = None,
) -> Path | None:
    """Atomically save one globally deduplicated evaluation snapshot as JSONL."""
    saved_records = list(records)
    if not saved_records:
        return None

    path = (
        Path(output_dir)
        / "responses"
        / "eval"
        / f"{safe_metric_prefix(metric_key_prefix)}_step_{int(step):08d}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")

    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in saved_records:
            payload = {
                "kind": "eval_response",
                "step": int(step),
                "metric_key_prefix": metric_key_prefix,
                **record,
            }
            payload["thinking_prefilled"] = prompt_prefills_thinking(
                str(payload["prompt"])
            )
            payload["prompt"] = clean_eval_prompt(str(payload["prompt"]), model_name)
            payload.setdefault(
                "response_id",
                (
                    f"{metric_key_prefix}:step:{int(step)}:"
                    f"example:{int(payload['eval_example_index'])}:"
                    f"sample:{int(payload['sample_index'])}"
                ),
            )
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    os.replace(temporary_path, path)
    return path
