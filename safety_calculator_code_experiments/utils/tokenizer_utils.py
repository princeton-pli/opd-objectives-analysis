from __future__ import annotations
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

FALLBACK_PAD_TOKENS = ("<|fim_pad|>",)

def ensure_distinct_pad_token(
    tokenizer: PreTrainedTokenizerBase,
    fallback_pad_tokens: Sequence[str] = FALLBACK_PAD_TOKENS,
) -> None:
    """Ensure padding cannot be the same as eos token"""
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return

    vocabulary = tokenizer.get_vocab()
    for candidate in fallback_pad_tokens:
        if candidate in vocabulary and vocabulary[candidate] != tokenizer.eos_token_id:
            tokenizer.pad_token = candidate
            break

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "OPD requires a padding token whose ID is distinct from eos_token_id. "
            f"Tried {tuple(fallback_pad_tokens)}; none is in this "
            "tokenizer's vocabulary with an id different from "
            f"eos_token_id={tokenizer.eos_token_id}."
        )
