from __future__ import annotations


def generate_with_vllm_colocate(
    engine,
    prompt_token_lists: list[list[int]],
    generation_config,
    *,
    sampling_params_class=None,
    tokens_prompt_class=None,
) -> list[list[int]]:
    """Generate from exact prompt token IDs with the shared OPD/SFT settings."""
    if sampling_params_class is None or tokens_prompt_class is None:
        from vllm import SamplingParams, TokensPrompt

        sampling_params_class = sampling_params_class or SamplingParams
        tokens_prompt_class = tokens_prompt_class or TokensPrompt

    eos_token_id = generation_config.eos_token_id
    if eos_token_id is None:
        stop_token_ids = None
    elif isinstance(eos_token_id, int):
        stop_token_ids = [eos_token_id]
    else:
        stop_token_ids = list(eos_token_id)

    sampling_params = sampling_params_class(
        n=1,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p if generation_config.top_p is not None else 1.0,
        top_k=generation_config.top_k or 0,
        max_tokens=generation_config.max_new_tokens,
        stop_token_ids=stop_token_ids,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = engine.generate(
        [tokens_prompt_class(prompt_token_ids=token_ids) for token_ids in prompt_token_lists],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return [list(output.outputs[0].token_ids) for output in outputs]
