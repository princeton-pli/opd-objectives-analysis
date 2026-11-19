from __future__ import annotations

from typing import Any

from transformers import GenerationConfig


CHAT_END_OF_TURN_TOKENS = ("<end_of_turn>",)


def chat_end_of_turn_token_ids(tokenizer: Any) -> list[int]:
    """Ids of the CHAT_END_OF_TURN_TOKENS that the tokenizer's chat template emits."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, dict):
        chat_template = "\n".join(value for value in chat_template.values() if isinstance(value, str))
    if not isinstance(chat_template, str):
        return []

    token_ids = []
    for token in CHAT_END_OF_TURN_TOKENS:
        if token not in chat_template:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            token_ids.append(token_id)
    return token_ids


def resolve_eos_token_id(model: Any, tokenizer: Any) -> int | list[int] | None:
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        end_of_turn_ids = [
            token_id
            for token_id in chat_end_of_turn_token_ids(tokenizer)
            if token_id != tokenizer_eos
        ]
        return [tokenizer_eos, *end_of_turn_ids] if end_of_turn_ids else tokenizer_eos

    model_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    return model_eos


def build_opd_generation_config(
    model: Any,
    tokenizer: Any,
    *,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> GenerationConfig:

    return GenerationConfig(
        max_new_tokens=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        top_k=top_k,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=resolve_eos_token_id(model, tokenizer),
        return_dict_in_generate=True,
    )


def build_vllm_sampling_params(
    generation_config,
    sampling_params_class,
    *,
    ignore_eos: bool = True,
    seed: int | None = None,
):
    """Build OPDTrainer's vLLM SamplingParams. seed gives the request its own generator."""
    eos_token_id = generation_config.eos_token_id
    if eos_token_id is None:
        stop_token_ids = None
    elif isinstance(eos_token_id, int):
        stop_token_ids = [eos_token_id]
    else:
        stop_token_ids = list(eos_token_id)

    return sampling_params_class(
        n=1,
        temperature=generation_config.temperature,
        top_p=(
            generation_config.top_p
            if generation_config.top_p is not None
            else 1.0
        ),
        top_k=generation_config.top_k or 0,
        max_tokens=generation_config.max_new_tokens,
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos,
        detokenize=False,
        **({} if seed is None else {"seed": seed}),
    )
