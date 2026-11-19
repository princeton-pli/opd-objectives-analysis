from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def candidate_id(row: dict) -> str:
    return f"{row['lib']}:{row['api']}::{row['task_name']}"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    api: str
    prompt: str
    continuation: str

    @classmethod
    def from_row(cls, row: dict) -> "Candidate":
        """One prompt of the cohort dataset, keyed by example_id, the key the evaluator looks it up by."""
        return cls(
            candidate_id=str(row["example_id"]),
            api=str(row["new_api"]),
            prompt=str(row["prompt"]),
            continuation=str(row["new_api_continuation"]),
        )


def encode_stable_boundary(
    tokenizer,
    candidate: Candidate,
) -> tuple[list[int], list[int]]:
    """Encode a predetermined syntactic split and reject tokenizer boundary changes."""
    prefix = candidate.prompt
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(candidate.continuation, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        prefix + candidate.continuation,
        add_special_tokens=False,
    )["input_ids"]
    if full_ids != prompt_ids + continuation_ids:
        raise ValueError(
            f"Unstable tokenizer boundary for {candidate.candidate_id!r}: "
            "tok(prompt + continuation) != tok(prompt) + tok(continuation)"
        )
    if not prompt_ids or not continuation_ids:
        raise ValueError(f"Empty prompt or continuation for {candidate.candidate_id!r}")
    return prompt_ids, continuation_ids


def continuation_token_span(tokenizer, prompt: str, continuation: str) -> tuple[int | None, list[int]]:
    """Where continuation starts in the tokens of prompt + continuation, None if the boundary retokenises."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False).input_ids
    return (len(prompt_ids) if full_ids[: len(prompt_ids)] == prompt_ids else None), full_ids


def shared_tokenizer_fingerprint(tokenizer, shared_vocab_size: int) -> str:
    tokens = tokenizer.convert_ids_to_tokens(list(range(shared_vocab_size)))
    payload = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
