from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

MULTIMODAL_WITH_TEXT_STACK = (Gemma3ForConditionalGeneration,)


def model_vocab_size(model: Any) -> int:
    """Return the vocabulary size for both text-only and composite HF models."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if embeddings is not None:
            num_embeddings = getattr(embeddings, "num_embeddings", None)
            if isinstance(num_embeddings, int) and num_embeddings > 0:
                return num_embeddings

            weight = getattr(embeddings, "weight", None)
            if weight is not None and len(weight.shape) > 0:
                return int(weight.shape[0])

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for candidate in (config, text_config):
        vocab_size = getattr(candidate, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size

    raise AttributeError(
        f"Could not determine the vocabulary size for {type(model).__name__}: "
        "the model has no input-embedding size and neither config.vocab_size "
        "nor config.text_config.vocab_size is available."
    )


def text_only_language_model(model):
    """Keep only the causal LM part of a multimodal model in MULTIMODAL_WITH_TEXT_STACK."""
    if type(model) not in MULTIMODAL_WITH_TEXT_STACK:
        return model
    with torch.device("meta"):
        text_model = MODEL_FOR_CAUSAL_LM_MAPPING[type(model.config.text_config)](model.config.text_config)
    text_model.model = model.model.language_model
    text_model.lm_head = model.lm_head
    text_model.tie_weights()
    text_model.generation_config = model.generation_config
    text_model.config._name_or_path = model.config._name_or_path
    return text_model


def load_causal_lm(name_or_path: str, **from_pretrained_kwargs):
    return AutoModelForCausalLM.from_pretrained(name_or_path, **from_pretrained_kwargs)


def load_causal_lm_text_only(name_or_path: str, **from_pretrained_kwargs):
    return text_only_language_model(load_causal_lm(name_or_path, **from_pretrained_kwargs))


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)
