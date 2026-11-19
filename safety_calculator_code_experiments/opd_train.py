import logging
import os

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from opd import OPDConfig, OPDTrainer
from opd.teacher_scoring import (
    check_teacher_eos,
)

from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
)
from trl.trainer.utils import disable_dropout_in_model

from utils.data import (
    CacheArguments,
    DataSubsetArguments,
    MetadataPassthroughChatMLCollator,
    filter_by_length,
    load_maybe_local_dataset,
    save_run_config,
    select_train_subset,
)
from utils import (
    build_eval_probes,
    ensure_distinct_pad_token,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def pretokenize_raw_text_sft_without_eos(train_dataset, tokenizer):
    """Tokenize raw-text prompt/completion rows ourselves so that TRL appends no EOS."""

    def encode(example):
        completion = example["completion"]
        if completion.endswith("\n"):
            completion = completion[:-1]
        prompt_ids = tokenizer(text=example["prompt"])["input_ids"]
        prompt_completion_ids = tokenizer(text=example["prompt"] + completion)["input_ids"]
        if prompt_completion_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Prompt is not a token prefix of prompt+completion for a row of the raw-text "
                "SFT dataset, so the completion mask cannot be built."
            )
        return {
            "input_ids": prompt_completion_ids,
            "completion_mask": [0] * len(prompt_ids)
            + [1] * (len(prompt_completion_ids) - len(prompt_ids)),
        }

    return train_dataset.map(encode, remove_columns=["prompt", "completion"])


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OPDConfig, ModelConfig, DataSubsetArguments, CacheArguments))
    script_args, training_args, model_args, subset_args, cache_args = parser.parse_args_and_config()

    max_prompt_length = subset_args.max_prompt_tokens
    if (
        (training_args.algorithm == "opd" or training_args.eval_strategy != "no")
        and max_prompt_length != -1
        and training_args.max_length is not None
        and max_prompt_length + training_args.max_completion_length > training_args.max_length
    ):
        raise ValueError(
            "max_prompt_tokens + max_completion_length must be <= max_length, got "
            f"{max_prompt_length} + {training_args.max_completion_length} > {training_args.max_length}."
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        cache_dir=cache_args.cache_dir,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model = None
    if training_args.algorithm == "opd":
        training_args.teacher_model_init_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            dtype=model_args.dtype,
            cache_dir=cache_args.cache_dir,
        )
        teacher_model = training_args.teacher_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
        cache_dir=cache_args.cache_dir,
    )
    ensure_distinct_pad_token(tokenizer)

    teacher_tokenizer_name = training_args.teacher_model_name_or_path
    if teacher_tokenizer_name:
        loaded_teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_name,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="left",
            cache_dir=cache_args.cache_dir,
        )
        check_teacher_eos(
            model_args.model_name_or_path, tokenizer, loaded_teacher_tokenizer
        )

    dataset = load_maybe_local_dataset(
        script_args.dataset_name, config=script_args.dataset_config, cache_dir=cache_args.cache_dir
    )
    train_dataset = select_train_subset(
        dataset[script_args.dataset_train_split],
        max_samples=subset_args.max_train_samples,
        seed=training_args.data_seed,
    )
    train_dataset = filter_by_length(
        train_dataset,
        tokenizer,
        max_prompt_tokens=subset_args.max_prompt_tokens,
        max_response_tokens=subset_args.max_response_tokens,
    )

    eval_dataset = None
    if training_args.eval_strategy != "no":
        test_pool = None
        if subset_args.max_test_eval_samples != 0:
            test_pool = filter_by_length(
                dataset[script_args.dataset_test_split],
                tokenizer,
                max_prompt_tokens=subset_args.max_prompt_tokens,
                max_response_tokens=subset_args.max_response_tokens,
            )

        eval_dataset = build_eval_probes(
            train_pool=train_dataset,
            max_train_samples=subset_args.max_train_eval_samples,
            test_pool=test_pool,
            max_test_samples=subset_args.max_test_eval_samples,
            data_seed=training_args.data_seed,
        )

    if training_args.algorithm == "sft":
        train_dataset = train_dataset.select_columns(["prompt", "completion"])
        train_dataset = pretokenize_raw_text_sft_without_eos(train_dataset, tokenizer)
        logging.info(
            "Pre-tokenized %d rows, so no end-of-sequence token is appended and the "
            "completion's trailing newline is dropped.",
            len(train_dataset),
        )

    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":
        save_run_config(
            training_args.output_dir,
            script_args=script_args,
            training_args=training_args,
            model_args=model_args,
            subset_args=subset_args,
            cache_args=cache_args,
        )

    data_selection_seed = training_args.data_seed
    training_args.data_seed = training_args.seed
    logging.info(
        "Seeds: data_seed=%s chose the examples; seed=%s orders them each epoch and drives "
        "init, dropout and rollout sampling.",
        data_selection_seed,
        training_args.seed,
    )

    # Training
    if training_args.algorithm == "opd":
        trainer = OPDTrainer(
            model=model_args.model_name_or_path,
            teacher_model=teacher_model,
            args=training_args,
            data_collator=MetadataPassthroughChatMLCollator(
                tokenizer=tokenizer,
                max_length=training_args.max_length,
                enable_thinking=training_args.enable_thinking,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    else:
        logging.info(
            "Starting completion-only SFT on %d fixed off-policy responses "
            "(data_seed=%s, trainer seed=%s)",
            len(train_dataset),
            data_selection_seed,
            training_args.seed,
        )
        trainer = SFTTrainer(
            model=model_args.model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        if training_args.disable_dropout:
            disable_dropout_in_model(trainer.model)
    trainer.model.config.use_cache = not training_args.gradient_checkpointing

    trainer.train()
    trainer.save_state()

    if training_args.save_strategy != "no" or training_args.save_final_model:
        trainer.save_model(training_args.output_dir)
