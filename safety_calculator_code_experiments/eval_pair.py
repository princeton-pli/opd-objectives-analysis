from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_compat import apply_vllm_compat_shims

apply_vllm_compat_shims()

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.trainer.utils import disable_dropout_in_model

from utils.calculator_metrics import (
    compute_calculator_metrics,
    parse_calculator_completion,
    parse_number,
)
from utils.completion_text import prompt_prefills_thinking, truncate_token_ids_at_stop
from utils.data import (
    CacheArguments,
    MetadataPassthroughChatMLCollator,
    filter_by_length,
    load_maybe_local_dataset,
    select_train_subset,
)

from opd.eval_rollouts import save_eval_rollout_records
from opd.opd_metrics import kl_diagnostics
from opd.teacher_scoring import (
    check_teacher_eos,
)
from utils import (
    build_opd_generation_config,
    ensure_distinct_pad_token,
    model_vocab_size,
    position_ids_from_attention_mask,
)
from utils import load_causal_lm
from utils import text_only_language_model as text_only_language_model_fn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EvaluationProtocol = Literal["calculator", "refusal"]

_PROTOCOL_TARGET_KEYS: dict[str, str] = {
    "calculator": "gold_answer",
    "refusal": "data_type",
}


@dataclass
class EvalConfig:
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional teacher model to evaluate alongside the student."},
    )
    eval_split: str = field(
        default="train",
        metadata={"help": "Dataset split to sample from."},
    )
    max_train_samples: int = field(
        default=-1,
        metadata={"help": "First subset size, same as --max_train_samples in training (-1 = all)."},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed for the training-pool shuffle; must match DATA_SEED in training."},
    )
    max_prompt_tokens: int = field(
        default=-1,
        metadata={"help": "Drop examples whose prompt exceeds this token count (-1 = no filter)."},
    )
    max_response_tokens: int = field(
        default=-1,
        metadata={"help": "Drop examples whose response exceeds this token count (-1 = no filter)."},
    )
    n_eval_samples: int = field(
        default=500,
        metadata={"help": "Number of examples to actually evaluate (-1 = full filtered set)."},
    )
    eval_seed: int = field(
        default=0,
        metadata={"help": "Seed for the eval subsample (independent of data_seed)."},
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max completion tokens; matches GOLDConfig.max_completion_length (default 128)."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling (0 = disabled)."},
    )
    enable_thinking: bool | None = field(
        default=False,
        metadata={"help": "Pass enable_thinking to the chat template; matches OPDConfig."},
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Max sequence length passed to the collator; matches --max_length in training."},
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Per-device batch size for generation."},
    )
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": (
                "Use vLLM for generation while preserving the HF prompt token IDs and sampling "
                "configuration. HF models are still loaded when KL/objective scoring requires them."
            )
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "Fraction of GPU memory available to the vLLM engine."},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": (
                "Generation seed; matches the TrainingArguments seed. With vLLM, example i is "
                "sampled with its own seed (seed + i), so runs are repeatable."
            )
        },
    )
    student_use_cache: bool = field(
        default=False,
        metadata={
            "help": "Student model use_cache setting; False matches OPD runs with gradient checkpointing."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable student dropout, matching OPDConfig."},
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "If set, results are saved as results.json inside this directory."},
    )
    save_tokenized_generations: bool = field(
        default=False,
        metadata={
            "help": (
                "Save exact completion token IDs for student and teacher generations under "
                "<output_dir>/responses/eval (alongside the decoded text)."
            )
        },
    )
    num_completions_to_print: int = field(
        default=10,
        metadata={
            "help": "Number of (prompt, completion) pairs to print with rich after each model's eval. "
            "-1 = print all, 0 = disable printing."
        },
    )
    print_seed: int = field(
        default=0,
        metadata={"help": "RNG seed for choosing which completions to print (independent of eval_seed)."},
    )

    def __post_init__(self) -> None:
        if self.save_tokenized_generations and self.output_dir is None:
            raise ValueError(
                "save_tokenized_generations=True requires output_dir."
            )
        if not self.use_vllm:
            raise ValueError("eval_pair.py generates with vLLM only; pass --use_vllm true.")
        if self.use_vllm and not 0.0 < self.vllm_gpu_memory_utilization <= 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1].")


def _load_model(
    name_or_path: str,
    *,
    revision: str | None,
    dtype: str | torch.dtype | None,
    attn_implementation: str | None,
    trust_remote_code: bool,
    cache_dir: str | None,
    use_cache: bool,
    disable_dropout: bool,
    device: torch.device,
    text_only_language_model: bool = False,
) -> AutoModelForCausalLM:
    logger.info("Loading model: %s", name_or_path)
    model = load_causal_lm(
        name_or_path,
        revision=revision,
        dtype=dtype or "auto",
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if text_only_language_model:
        model = text_only_language_model_fn(model)
    model.config.use_cache = use_cache
    model = model.to(device)
    if disable_dropout:
        disable_dropout_in_model(model)
    model.eval()
    return model


@dataclass
class SampledRolloutBatch:
    """CPU copy of a student rollout batch needed for subsequent teacher scoring."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_width: int
    mask: torch.Tensor
    student_logprobs_sampled: torch.Tensor


def _vllm_generate_worker(
    engine_kwargs: dict,
    prompt_token_lists: list[list[int]],
    generation_config,
    request_seeds: list[int],
) -> list[tuple[list[int], int | str | None]]:
    # In-process engine core: all requests are queued before the first step, so batches are repeatable.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from utils.generation_utils import build_vllm_sampling_params
    from vllm import LLM, SamplingParams, TokensPrompt

    engine = LLM(**engine_kwargs)
    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_lists],
        sampling_params=[
            build_vllm_sampling_params(generation_config, SamplingParams, seed=seed)
            for seed in request_seeds
        ],
        use_tqdm=True,
    )
    return [
        (list(output.outputs[0].token_ids), output.outputs[0].stop_reason)
        for output in outputs
    ]


def _generate_sequences_with_vllm(
    name_or_path: str,
    batches: list[dict],
    tokenizer: AutoTokenizer,
    generation_config,
    *,
    eval_args: EvalConfig,
    model_kwargs: dict,
) -> list[torch.Tensor]:
    """Generate HF-shaped sequences from the exact unpadded prompt token IDs."""
    if not batches:
        return []

    prompt_token_lists: list[list[int]] = []
    for batch in batches:
        prompt_ids = batch["prompts"]
        prompt_mask = batch["prompt_attention_mask"].bool()
        prompt_token_lists.extend(
            prompt_ids[row][prompt_mask[row]].tolist()
            for row in range(prompt_ids.shape[0])
        )
    required_model_len = max(map(len, prompt_token_lists)) + generation_config.max_new_tokens
    # Per-request seeds (seed + example index) make sampling independent of vLLM's scheduling order.
    request_seeds = [eval_args.seed + index for index in range(len(prompt_token_lists))]
    engine_kwargs = {
        "model": name_or_path,
        "revision": model_kwargs.get("revision"),
        "dtype": model_kwargs.get("dtype") or "auto",
        "trust_remote_code": model_kwargs.get("trust_remote_code", False),
        "download_dir": model_kwargs.get("cache_dir"),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": eval_args.vllm_gpu_memory_utilization,
        "max_num_seqs": eval_args.eval_batch_size,
        "max_model_len": required_model_len,
        "seed": eval_args.seed,
        "language_model_only": True,
    }
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        outputs = executor.submit(
            _vllm_generate_worker,
            engine_kwargs,
            prompt_token_lists,
            generation_config,
            request_seeds,
        ).result()

    sequences: list[torch.Tensor] = []
    output_cursor = 0
    for batch in batches:
        prompt_ids = batch["prompts"]
        batch_outputs = outputs[output_cursor : output_cursor + prompt_ids.shape[0]]
        output_cursor += prompt_ids.shape[0]
        completion_ids = []
        for token_ids, stop_reason in batch_outputs:
            if isinstance(stop_reason, int) and (
                not token_ids or token_ids[-1] != stop_reason
            ):
                token_ids.append(stop_reason)
            completion_ids.append(token_ids)
        completion_width = max(map(len, completion_ids), default=0)
        batch_sequences = torch.full(
            (prompt_ids.shape[0], prompt_ids.shape[1] + completion_width),
            tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
        )
        batch_sequences[:, : prompt_ids.shape[1]] = prompt_ids
        for row, token_ids in enumerate(completion_ids):
            if token_ids:
                batch_sequences[
                    row,
                    prompt_ids.shape[1] : prompt_ids.shape[1] + len(token_ids),
                ] = torch.tensor(token_ids, dtype=prompt_ids.dtype)
        sequences.append(batch_sequences)
    return sequences


def _sampled_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_attention_mask(attention_mask),
        )
        shifted_logits = outputs.logits[:, prompt_width - 1 : -1, :]
        sampled_ids = input_ids[:, prompt_width:]
        if sampled_ids.numel() and int(sampled_ids.max().item()) >= shifted_logits.shape[-1]:
            raise ValueError(
                "A generated token id is outside the scoring model's vocabulary. "
                "Initial KL diagnostics require aligned student and teacher tokenizers/vocabularies."
            )
        return torch.log_softmax(shifted_logits / temperature, dim=-1).gather(
            -1, sampled_ids.unsqueeze(-1)
        ).squeeze(-1)


def _completion_token_ids_including_stop(
    token_ids: list[int],
    *,
    stop_token_ids: int | list[int] | None,
    pad_token_id: int | None,
) -> list[int]:
    """Keep the sampled stop token while removing only batch-padding suffixes."""
    if stop_token_ids is None:
        stops: set[int] = set()
    elif isinstance(stop_token_ids, int):
        stops = {stop_token_ids}
    else:
        stops = set(stop_token_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in stops:
            return token_ids[: index + 1]
    if pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
    return token_ids


def _generate_completions(
    model: AutoModelForCausalLM | None,
    dataloader: DataLoader | list[dict],
    tokenizer: AutoTokenizer,
    *,
    protocol: EvaluationProtocol,
    generation_config,
    capture_rollouts: bool,
    capture_response_records: bool,
    device: torch.device,
    generated_sequences: list[torch.Tensor] | None = None,
    capture_token_ids: bool = False,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[SampledRolloutBatch],
    list[dict],
]:
    """Run the same batched sampling path used by OPD training."""
    prompts: list[str] = []
    completions: list[str] = []
    targets: list[str] = []
    rollout_batches: list[SampledRolloutBatch] = []
    response_records: list[dict] = []

    for batch_index, batch in enumerate(tqdm(dataloader, desc="  generating", leave=False)):
        prompt_ids = batch["prompts"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        prompt_len = prompt_ids.shape[1]
        output_ids = generated_sequences[batch_index].to(device)

        generated_attention_mask = torch.ones_like(output_ids)
        generated_attention_mask[output_ids == tokenizer.pad_token_id] = 0
        completion_mask = output_ids[:, prompt_len:] != tokenizer.pad_token_id

        if capture_rollouts:
            if model is None:
                raise ValueError("Rollout capture requires an HF scoring model.")
            student_logprobs = _sampled_token_logprobs(
                model,
                output_ids,
                generated_attention_mask,
                prompt_len,
                generation_config.temperature,
            )
            rollout_batches.append(
                SampledRolloutBatch(
                    input_ids=output_ids.cpu(),
                    attention_mask=generated_attention_mask.cpu(),
                    prompt_width=prompt_len,
                    mask=completion_mask.cpu(),
                    student_logprobs_sampled=student_logprobs.cpu(),
                )
            )

        generated_ids = output_ids[:, prompt_len:]
        batch_completion_token_ids: list[list[int]] = []
        for idx, seq in enumerate(generated_ids):
            # Mirror the decoding used in OPDTrainer._attach_vllm_rollouts
            prompt_tokens = prompt_ids[idx][prompt_mask[idx].bool()]
            prompts.append(
                tokenizer.decode(
                    prompt_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            scored_completion_token_ids = _completion_token_ids_including_stop(
                seq.tolist(),
                stop_token_ids=generation_config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_token_ids = truncate_token_ids_at_stop(
                scored_completion_token_ids, generation_config.eos_token_id
            )
            batch_completion_token_ids.append(scored_completion_token_ids)
            text = tokenizer.decode(
                completion_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            completions.append(text)

        target_key = _PROTOCOL_TARGET_KEYS[protocol]
        targets.extend(str(target) for target in batch[target_key])
        if capture_response_records:
            batch_size = len(batch[target_key])
            for index in range(batch_size):
                record: dict[str, object] = {
                    "eval_example_index": int(batch["eval_example_index"][index]),
                    "sample_index": 0,
                    "prompt": prompts[-batch_size + index],
                    "completion": completions[-batch_size + index],
                }
                if capture_token_ids:
                    record["completion_token_ids"] = batch_completion_token_ids[index]
                metadata_converters = {
                    "data_type": str,
                    "should_abstain": bool,
                    "prompt_style": str,
                    "source": str,
                }
                for key, converter in metadata_converters.items():
                    if key in batch:
                        record[key] = converter(batch[key][index])
                response_records.append(record)

    return (
        prompts,
        completions,
        targets,
        rollout_batches,
        response_records,
    )


def _score_student_rollouts_with_teacher(
    teacher_model: AutoModelForCausalLM,
    rollout_batches: list[SampledRolloutBatch],
    *,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    """Compute the same sampled-token KL diagnostics emitted by OPDTrainer.compute_loss."""
    if not rollout_batches:
        return {}

    values: defaultdict[str, list[float]] = defaultdict(list)

    for batch in tqdm(rollout_batches, desc="  teacher scoring", leave=False):
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        teacher_lp = _sampled_token_logprobs(
            teacher_model,
            input_ids,
            attention_mask,
            batch.prompt_width,
            temperature,
        ).cpu()
        diagnostics = kl_diagnostics(
            student_logprobs_sampled=batch.student_logprobs_sampled,
            teacher_logprobs_sampled=teacher_lp,
            mask=batch.mask,
        )
        for name, value in diagnostics.items():
            values[name].append(value.item())

    # One value per compute_loss call, averaged as GOLDTrainer.log does rather than pooled over tokens.
    return {name: sum(batch_values) / len(batch_values) for name, batch_values in values.items()}


def _training_style_rollout_metrics(
    completions: list[str],
    answer_letters: list[str],
    batch_size: int,
    *,
    protocol: EvaluationProtocol,
    thinking_prefilled: bool = False,
) -> dict[str, float]:
    """Mirror OPDTrainer/GOLDTrainer's per-batch metric accumulation and averaging."""
    targets = answer_letters
    values: defaultdict[str, list[float]] = defaultdict(list)
    for start in range(0, len(completions), batch_size):
        end = start + batch_size
        batch_completions = completions[start:end]
        batch_targets = targets[start:end]
        if protocol == "calculator":
            metrics = compute_calculator_metrics(
                batch_completions,
                batch_targets,
                thinking_prefilled=thinking_prefilled,
            )
        else:
            raise ValueError(f"Unknown evaluation protocol: {protocol!r}")
        for name, value in metrics.items():
            values[name].append(value)
    return {name: sum(batch_values) / len(batch_values) for name, batch_values in values.items()}


def _aggregate_rollout_metrics(
    completions: list[str],
    targets: list[str],
    *,
    protocol: EvaluationProtocol,
    thinking_prefilled: bool = False,
) -> dict[str, float]:
    """Compute whole-evaluation metrics, including boxed completion diagnostics."""
    if protocol == "calculator":
        return compute_calculator_metrics(
            completions,
            targets,
            thinking_prefilled=thinking_prefilled,
        )
    raise ValueError(f"Unknown evaluation protocol: {protocol!r}")


def _eval_model(
    label: str,
    name_or_path: str,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    eval_args: EvalConfig,
    model_kwargs: dict,
    device: torch.device,
    *,
    protocol: EvaluationProtocol,
    capture_rollouts: bool = False,
    score_rollouts: list[SampledRolloutBatch] | None = None,
    resize_vocab_to: int | None = None,
    use_cache: bool = True,
    capture_response_records: bool = False,
    capture_token_ids: bool = False,
    text_only_language_model: bool = False,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[dict],
    list[SampledRolloutBatch],
    dict[str, float],
    int,
    list[dict],
]:
    print(f"\n>>> Evaluating: {label}  ({name_or_path})")
    model: AutoModelForCausalLM | None = None
    generation_batches: DataLoader | list[dict] = dataloader
    generated_sequences: list[torch.Tensor] | None = None
    if eval_args.use_vllm:
        if tokenizer.eos_token_id is None:
            raise ValueError("vLLM parity requires tokenizer.eos_token_id to be set.")
        generation_config = build_opd_generation_config(
            None,
            tokenizer,
            max_completion_length=eval_args.max_new_tokens,
            temperature=eval_args.temperature,
            top_p=eval_args.top_p,
            top_k=eval_args.top_k,
        )
        generation_batches = list(dataloader)
        generated_sequences = _generate_sequences_with_vllm(
            name_or_path,
            generation_batches,
            tokenizer,
            generation_config,
            eval_args=eval_args,
            model_kwargs=model_kwargs,
        )
        needs_hf_model = capture_rollouts or score_rollouts is not None
        evaluated_vocab_size = len(tokenizer)

    if needs_hf_model:
        model = _load_model(
            name_or_path,
            **model_kwargs,
            use_cache=use_cache,
            disable_dropout=eval_args.disable_dropout,
            device=device,
            text_only_language_model=text_only_language_model,
        )
        current_vocab_size = model_vocab_size(model)
        if resize_vocab_to is not None and current_vocab_size != resize_vocab_to:
            if eval_args.use_vllm:
                raise ValueError(
                    "vLLM cannot reproduce the HF vocabulary resize required for this model pair."
                )
        evaluated_vocab_size = model_vocab_size(model)
    logger.info("%s eos_token_id resolved to: %s", label, generation_config.eos_token_id)

    diagnostics = {}
    if score_rollouts is not None:
        if model is None:
            raise RuntimeError("Teacher rollout scoring requires an HF model.")
        diagnostics = _score_student_rollouts_with_teacher(
            model,
            score_rollouts,
            temperature=eval_args.temperature,
            device=device,
        )

    (
        prompts,
        completions,
        targets,
        rollout_batches,
        response_records,
    ) = _generate_completions(
        model,
        generation_batches,
        tokenizer,
        protocol=protocol,
        generation_config=generation_config,
        capture_rollouts=capture_rollouts,
        capture_response_records=capture_response_records,
        device=device,
        generated_sequences=generated_sequences,
        capture_token_ids=capture_token_ids,
    )
    thinking_prefilled = any(prompt_prefills_thinking(prompt) for prompt in prompts)
    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if protocol == "refusal":
        samples = []
        aggregate_metrics = {}
        training_style_metrics = {}
    else:
        shown_prompts, shown_completions, shown_targets = _select_samples(
            prompts, completions, targets,
            num_samples=eval_args.num_completions_to_print,
            seed=eval_args.print_seed,
        )
        _print_completions_table(
            f"{label}  |  {name_or_path}",
            shown_prompts,
            shown_completions,
            shown_targets,
            protocol=protocol,
            thinking_prefilled=thinking_prefilled,
        )
        target_name = _PROTOCOL_TARGET_KEYS[protocol]
        samples = [
            {"prompt": prompt, "completion": completion, target_name: target}
            for prompt, completion, target in zip(
                shown_prompts,
                shown_completions,
                shown_targets,
                strict=True,
            )
        ]
        aggregate_metrics = _aggregate_rollout_metrics(
            completions,
            targets,
            protocol=protocol,
            thinking_prefilled=thinking_prefilled,
        )
        training_style_metrics = _training_style_rollout_metrics(
            completions,
            targets,
            eval_args.eval_batch_size,
            protocol=protocol,
            thinking_prefilled=thinking_prefilled,
        )
    return (
        aggregate_metrics,
        training_style_metrics,
        samples,
        rollout_batches,
        diagnostics,
        evaluated_vocab_size,
        response_records,
    )


def _select_samples(
    prompts: list[str],
    completions: list[str],
    answer_letters: list[str],
    num_samples: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Pick a random subset of (prompt, completion, answer) triples to display; -1 keeps all, 0 none."""
    n = len(prompts)
    if num_samples == 0 or n == 0:
        return [], [], []
    if 0 < num_samples < n:
        indices = random.Random(seed).sample(range(n), num_samples)
        return [prompts[i] for i in indices], [completions[i] for i in indices], [answer_letters[i] for i in indices]
    return prompts, completions, answer_letters


def _print_completions_table(
    label: str,
    prompts: list[str],
    completions: list[str],
    targets: list[str],
    *,
    protocol: EvaluationProtocol,
    max_completion_chars: int = 2000,
    thinking_prefilled: bool = False,
) -> None:
    """Mirrors OPDTrainer's `_print_opd_completions_table` so eval output looks familiar."""
    if len(prompts) == 0:
        return

    console = Console(width=200, height=50)
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Prompt", ratio=2)
    table.add_column("Completion", ratio=3, no_wrap=False)
    table.add_column("Action", ratio=1, justify="center")
    table.add_column("Result", ratio=1, justify="center")

    for prompt, completion, gold in zip(prompts, completions, targets, strict=True):
        if len(completion) > max_completion_chars:
            completion_display = completion[:max_completion_chars] + "\n[…truncated]"
        else:
            completion_display = completion

        if protocol == "calculator":
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
            gold_value = parse_number(gold)
            result_cell = Text(
                "correct"
                if predicted is not None and predicted == gold_value
                else f"wrong  pred={predicted} gold={gold}"
            )
        else:
            raise ValueError(f"Unknown evaluation protocol: {protocol!r}")

        table.add_row(Text(prompt), Text(completion_display), action_cell, result_cell)
        table.add_section()

    panel = Panel(table, expand=True, title=label)
    console.print(panel)


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    bar = "─" * 48
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    for k, v in metrics.items():
        print(f"  {k:<38}  {v:.4f}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, EvalConfig, ModelConfig, CacheArguments))
    script_args, eval_args, model_args, cache_args = parser.parse_args_and_config()

    max_prompt_length = eval_args.max_prompt_tokens
    if (
        max_prompt_length != -1
        and eval_args.max_length is not None
        and max_prompt_length + eval_args.max_new_tokens > eval_args.max_length
    ):
        raise ValueError(
            "max_prompt_tokens + max_new_tokens must be <= max_length, got "
            f"{max_prompt_length} + {eval_args.max_new_tokens} > {eval_args.max_length}."
        )
    if eval_args.use_vllm:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise RuntimeError(
                "Single-GPU vLLM generation requires exactly one visible CUDA device, "
                f"but found {visible_gpus}."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
        cache_dir=cache_args.cache_dir,
    )
    ensure_distinct_pad_token(tokenizer)
    if eval_args.teacher_model_name_or_path:
        check_teacher_eos(
            model_args.model_name_or_path,
            tokenizer,
            AutoTokenizer.from_pretrained(
                eval_args.teacher_model_name_or_path,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
                cache_dir=cache_args.cache_dir,
            ),
        )

    dataset = load_maybe_local_dataset(
        script_args.dataset_name,
        config=script_args.dataset_config,
        cache_dir=cache_args.cache_dir,
    )
    split_data = dataset[eval_args.eval_split]

    if eval_args.max_train_samples != -1:
        split_data = select_train_subset(
            split_data,
            max_samples=eval_args.max_train_samples,
            seed=eval_args.data_seed,
        )

    split_data = filter_by_length(
        split_data,
        tokenizer,
        max_prompt_tokens=eval_args.max_prompt_tokens,
        max_response_tokens=eval_args.max_response_tokens,
    )

    if eval_args.n_eval_samples != -1:
        split_data = select_train_subset(
            split_data,
            max_samples=eval_args.n_eval_samples,
            seed=eval_args.eval_seed,
        )

    if "gold_answer" in split_data.column_names:
        protocol: EvaluationProtocol = "calculator"
    elif "data_type" in split_data.column_names:
        protocol = "refusal"
    else:
        raise ValueError(
            "eval_pair.py requires a gold_answer column (calculator protocol) "
            "or a data_type column (refusal protocol)."
        )

    persist_generation_records = bool(
        eval_args.save_tokenized_generations
    )
    if persist_generation_records:
        split_data = split_data.add_column(
            "eval_example_index", list(range(len(split_data)))
        )

    logger.info(
        "Evaluating %d examples from %r with %s metrics (data_seed=%d, eval_seed=%d)",
        len(split_data),
        eval_args.eval_split,
        protocol,
        eval_args.data_seed,
        eval_args.eval_seed,
    )

    collator = MetadataPassthroughChatMLCollator(
        tokenizer=tokenizer,
        max_length=eval_args.max_length,
        enable_thinking=eval_args.enable_thinking,
    )
    dataloader = DataLoader(
        split_data,
        batch_size=eval_args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=cache_args.cache_dir,
    )

    results: dict[str, dict[str, float]] = {}
    all_samples: dict[str, list[dict]] = {}

    (
        student_metrics,
        student_training_metrics,
        student_samples,
        student_rollouts,
        _,
        student_vocab_size,
        student_response_records,
    ) = _eval_model(
        "STUDENT", model_args.model_name_or_path,
        dataloader, tokenizer, eval_args, model_kwargs, device,
        protocol=protocol,
        capture_rollouts=bool(eval_args.teacher_model_name_or_path),
        capture_response_records=persist_generation_records,
        capture_token_ids=eval_args.save_tokenized_generations,
        use_cache=eval_args.student_use_cache,
        text_only_language_model=True,  # matches OPDTrainer: a multimodal Gemma-3 student keeps only its text stack
    )
    results["student"] = student_metrics
    all_samples["student"] = student_samples

    initial_training_metrics = {
        f"rollout/{name}": value for name, value in student_training_metrics.items()
    }

    teacher_response_records = []
    if eval_args.teacher_model_name_or_path:
        (
            teacher_metrics,
            _,
            teacher_samples,
            _,
            diagnostics,
            _,
            teacher_response_records,
        ) = _eval_model(
            "TEACHER", eval_args.teacher_model_name_or_path,
            dataloader, tokenizer, eval_args, model_kwargs, device,
            protocol=protocol,
            score_rollouts=student_rollouts,
            resize_vocab_to=student_vocab_size,
            capture_response_records=persist_generation_records,
            capture_token_ids=eval_args.save_tokenized_generations,
        )
        results["teacher"] = teacher_metrics
        all_samples["teacher"] = teacher_samples
        initial_training_metrics.update(
            {f"diagnostics/{name}": value for name, value in diagnostics.items()}
        )

    results["initial_training"] = initial_training_metrics
    if student_metrics:
        _print_metrics(f"STUDENT  |  {model_args.model_name_or_path}", student_metrics)
    _print_metrics("INITIAL TRAINING METRICS  |  step 0", initial_training_metrics)
    if eval_args.teacher_model_name_or_path and results["teacher"]:
        _print_metrics(f"TEACHER  |  {eval_args.teacher_model_name_or_path}", results["teacher"])

    print()

    if eval_args.output_dir:
        os.makedirs(eval_args.output_dir, exist_ok=True)
        results_path = os.path.join(eval_args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "student_model": model_args.model_name_or_path,
                    "teacher_model": eval_args.teacher_model_name_or_path,
                    "eval_split": eval_args.eval_split,
                    "max_train_samples": eval_args.max_train_samples,
                    "data_seed": eval_args.data_seed,
                    "max_prompt_tokens": eval_args.max_prompt_tokens,
                    "max_response_tokens": eval_args.max_response_tokens,
                    "n_eval_samples": len(split_data),
                    "eval_seed": eval_args.eval_seed,
                    "seed": eval_args.seed,
                    "eval_batch_size": eval_args.eval_batch_size,
                    "student_use_cache": eval_args.student_use_cache,
                    "disable_dropout": eval_args.disable_dropout,
                    "save_tokenized_generations": eval_args.save_tokenized_generations,
                    "max_length": eval_args.max_length,
                    "enable_thinking": eval_args.enable_thinking,
                    "generation": {
                        "backend": "vllm" if eval_args.use_vllm else "hf",
                        "do_sample": True,
                        "max_new_tokens": eval_args.max_new_tokens,
                        "temperature": eval_args.temperature,
                        "top_p": eval_args.top_p,
                        "top_k": eval_args.top_k,
                    },
                    "metrics": results,
                },
                f,
                indent=2,
            )
        logger.info("Results saved to %s", results_path)

        if any(all_samples.values()):
            samples_path = os.path.join(eval_args.output_dir, "completions.json")
            with open(samples_path, "w") as f:
                json.dump(all_samples, f, indent=2)
            logger.info("Sample completions saved to %s", samples_path)

        if persist_generation_records:
            student_responses_path = save_eval_rollout_records(
                output_dir=eval_args.output_dir,
                metric_key_prefix=f"eval_{eval_args.eval_split}_student",
                step=0,
                records=student_response_records,
                model_name=getattr(tokenizer, "name_or_path", None),
            )
            logger.info("Student evaluation responses saved to %s", student_responses_path)
            if eval_args.teacher_model_name_or_path:
                teacher_responses_path = save_eval_rollout_records(
                    output_dir=eval_args.output_dir,
                    metric_key_prefix=f"eval_{eval_args.eval_split}_teacher",
                    step=0,
                    records=teacher_response_records,
                    # _eval_model renders both arms with the student tokenizer.
                    model_name=getattr(tokenizer, "name_or_path", None),
                )
                logger.info("Teacher evaluation responses saved to %s", teacher_responses_path)
