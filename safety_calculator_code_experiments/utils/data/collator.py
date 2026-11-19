from __future__ import annotations
from dataclasses import dataclass, field

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl.trainer.utils import DataCollatorForChatML, pad

from utils import DEFAULT_METADATA_KEYS

class _ChatTemplateKwargsTokenizer:
    """Tokenizer that injects extra kwargs into apply_chat_template."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **chat_template_kwargs):
        self._tokenizer = tokenizer
        self._chat_template_kwargs = {
            key: value for key, value in chat_template_kwargs.items() if value is not None
        }

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        for key, value in self._chat_template_kwargs.items():
            kwargs.setdefault(key, value)
        return self._tokenizer.apply_chat_template(*args, **kwargs)


@dataclass
class MetadataPassthroughChatMLCollator:
    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int | None = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"
    metadata_keys: tuple[str, ...] = field(default_factory=lambda: DEFAULT_METADATA_KEYS)
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        self._tokenizer = _ChatTemplateKwargsTokenizer(
            self.tokenizer,
            enable_thinking=self.enable_thinking,
        )
        self._collator = DataCollatorForChatML(
            tokenizer=self._tokenizer,
            ignore_index=self.ignore_index,
            max_length=self.max_length,
            prompt_key=self.prompt_key,
            messages_key=self.messages_key,
        )

    def __call__(self, examples: list[dict]) -> dict:
        batch = self._collator(examples)
        batch["prompts"], batch["prompt_attention_mask"] = self._collate_generation_prompts(
            examples, self.tokenizer
        )
        for key in self.metadata_keys:
            if examples and key in examples[0]:
                batch[key] = [example[key] for example in examples]
        return batch

    def _collate_generation_prompts(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = []
        prompt_masks = []
        for example in examples:
            formatted_prompt = example.get(self.prompt_key)
            if formatted_prompt is None:
                formatted_prompt = _ChatTemplateKwargsTokenizer(
                    tokenizer, enable_thinking=self.enable_thinking
                ).apply_chat_template(
                    example[self.messages_key][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ids = tokenizer(
                formatted_prompt,
                truncation=False,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]
            if self.max_length is not None and len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            prompt_ids.append(torch.tensor(ids, dtype=torch.long))
            prompt_masks.append(torch.ones(len(ids), dtype=torch.long))

        return (
            pad(prompt_ids, padding_side="left", padding_value=tokenizer.pad_token_id),
            pad(prompt_masks, padding_side="left", padding_value=0),
        )
