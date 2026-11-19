from __future__ import annotations

import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import gather_object
from transformers import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from trl.experimental.gold import GOLDTrainer
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
)

from utils.completion_text import prompt_prefills_thinking, truncate_token_ids_at_stop
from utils.data.collator import DEFAULT_METADATA_KEYS

from .opd_config import OPDConfig
from .eval_rollouts import save_eval_rollout_records
from .opd_losses import (
    divergence_loss_full_vocab,
    logprobs_at_ids,
    sequence_reverse_kl_loss,
)
from .opd_metrics import kl_diagnostics
from utils.rollout_metrics import (
    compute_rollout_metric_counts,
    rollout_count_names,
    rollout_metrics_from_counts,
)
from .vllm_rollout import generate_with_vllm_colocate
from .code_completion.recovery_evaluator import NewAPIRecoveryEvaluator
from .teacher_scoring import LocalHFTeacherScorer
from utils import (
    build_opd_generation_config,
    load_causal_lm,
    load_causal_lm_text_only,
    model_vocab_size,
    position_ids_from_attention_mask,
    print_opd_completions_table,
)
from utils.eval_cadence import (
    FractionalEpochEvalCallback as _FractionalEpochEvalCallback,
)

if is_vllm_available():
    from vllm import LLM


class _RolloutCountPersistenceCallback(TrainerCallback):
    """Persist raw count chunks at optimizer-step boundaries and at the end of training."""

    def __init__(self, trainer: "OPDTrainer") -> None:
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        self.trainer._record_training_step_counts(state.global_step)
        if state.global_step % self.trainer.metrics_flush_steps == 0:
            self.trainer._flush_training_count_records()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        del args, kwargs
        self.trainer._record_training_step_counts(state.global_step)
        self.trainer._flush_training_count_records()
        return control


class _OPDVLLMWeightsStaleCallback(TrainerCallback):
    """Mark the colocated vLLM engine's weights stale after each optimizer update."""

    def __init__(self, trainer: "OPDTrainer") -> None:
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.trainer._vllm_weights_current = False
        return control


class OPDTrainer(GOLDTrainer):
    _tag_names = ["trl", "opd"]
    _name = "OPD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str | None = None,
        args: OPDConfig | None = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        if args.teacher_device is not None:
            student_device = args._setup_devices
            if torch.cuda.device_count() != 2 or torch.device(args.teacher_device) == student_device:
                raise ValueError(
                    f"teacher_device={args.teacher_device!r}: only the 2-GPU setup is supported for now (student "
                    f"on {student_device}, teacher on the other GPU); found {torch.cuda.device_count()} visible "
                    "GPU(s). More GPUs would need a sharded teacher or multi-process training, not implemented."
                )
            args._n_gpu = 1
        if teacher_model is None:
            teacher_model = args.teacher_model_name_or_path

        if args.use_liger_kernel:
            raise NotImplementedError(
                "OPDTrainer does not support use_liger_kernel; it is only wired up for GOLD's native "
                "tok_fkl/tok_rkl/tok_js/tok_skl loss, not the OPD losses implemented here."
            )
        if args.use_uld_loss:
            raise NotImplementedError("OPDTrainer does not support use_uld_loss (cross-tokenizer distillation).")

        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        model_init_kwargs = dict(args.model_init_kwargs or {})
        self.model_revision = model_init_kwargs.get("revision")
        student_model_cache_dir = model_init_kwargs.get("cache_dir")

        if isinstance(model, str):
            # A multimodal Gemma-3 student (4B+) is reduced to its text stack; other models load as before.
            model = load_causal_lm_text_only(
                model,
                **model_init_kwargs,
            )
            args.model_init_kwargs = None

        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the OPDConfig, but your teacher_model is "
                "already instantiated."
            )
        else:
            teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs)
            dtype = teacher_model_init_kwargs.pop("torch_dtype", teacher_model_init_kwargs.pop("dtype", None))
            if dtype not in ("auto", None) and isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            if dtype is not None:
                teacher_model_init_kwargs["dtype"] = dtype

        if isinstance(teacher_model, str):
            teacher_model = load_causal_lm(
                teacher_model,
                **teacher_model_init_kwargs,
            )
            args.teacher_model_init_kwargs = None

        SFTTrainer.__init__(
            self,
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        self.model_accepts_loss_kwargs = True

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        teacher_model.resize_token_embeddings(model_vocab_size(self.model))
        if args.teacher_device is not None:
            if self.is_deepspeed_enabled or self.accelerator.num_processes > 1:
                raise ValueError("teacher_device supports single-process training without DeepSpeed only.")
            # Same autocast/fp32-output wrapping as below, but on the teacher's own GPU.
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model.to(args.teacher_device), evaluation_mode=True, device_placement=False
            )
        elif self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_scorer = LocalHFTeacherScorer(self.teacher_model)

        self.opd_loss_type = args.opd_loss_type

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.log_rollout_metrics = args.log_rollout_metrics
        self.save_eval_rollout_records = args.save_eval_rollout_records
        self.metrics_flush_steps = args.metrics_flush_steps
        self.loss_reduction = args.loss_reduction

        # Track per-step loss statistics for on/off-policy batches (used in logging), mirroring GOLDTrainer.
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0
        # GOLDTrainer.log's ULD accumulators stay at zero; OPDTrainer never sets use_uld_loss.
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False
        # Required by inherited GOLDTrainer codepaths (_prepare_dataset, log); OPDTrainer never uses ULD loss.
        self.use_uld_loss = False
        self.teacher_tokenizer = None

        self.generation_config = build_opd_generation_config(
            self.model,
            self.processing_class,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._rollout_metric_counts = {
            "train": defaultdict(int),
            "eval": defaultdict(int),
        }
        self._training_count_records: list[dict[str, Any]] = []
        self._pending_eval_rollout_counts: dict[str, dict[str, int]] = {}
        self._eval_rollout_records: list[dict[str, Any]] = []
        self._new_api_recovery = NewAPIRecoveryEvaluator.from_path(
            args.new_api_recovery_screen_path,
            self.processing_class,
            eval_dataset,
        )
        self._opd_eval_active = False
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "gold_answer": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }
        self.add_callback(_RolloutCountPersistenceCallback(self))
        if args.evals_per_epoch > 0:
            self.add_callback(_FractionalEpochEvalCallback(args.evals_per_epoch))

        # Student-side vLLM integration for on-policy generation - identical to GOLDTrainer.
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            ensure_master_addr_port()

            self.vllm_engine = LLM(
                model=self.model_name_or_path,
                revision=self.model_revision,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_num_seqs=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                max_model_len=args.max_length,
                distributed_executor_backend="external_launcher",
                seed=self.accelerator.process_index,
                enable_sleep_mode=self.vllm_enable_sleep_mode,
                download_dir=student_model_cache_dir,
                language_model_only=True,
            )
            self._repair_preexisting_transformers_modules_after_vllm_init(self.model)
            if self.teacher_model is not None:
                # vLLM's Transformers backend decorates the HF class, so the in-process teacher needs the same bypass.
                self._repair_preexisting_transformers_modules_after_vllm_init(self.teacher_model)

            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
            if hasattr(llm_model, "language_model"):
                self._wrap_vllm_load_weights_for_text_only_hf_names(llm_model)

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
            self.accelerator.wait_for_everyone()

            self._vllm_weights_current = False
            self.add_callback(_OPDVLLMWeightsStaleCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # Keep the raw "prompt" column: GOLDTrainer only protects "messages", and the collator reads "prompt".
        for column in ("prompt", *DEFAULT_METADATA_KEYS):
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    def _log_rollout_metrics(self, inputs: dict) -> None:
        if not self.log_rollout_metrics:
            return
        mode = "eval" if self._opd_eval_active else "train"
        completions = inputs.get("original_completion_text")
        if completions is None:
            return

        counts = compute_rollout_metric_counts(
            completions,
            inputs.get("original_prompt_text", ()),
            inputs,
        )
        if not counts:
            return
        metrics = rollout_metrics_from_counts(counts)

        for name, value in metrics.items():
            self._metrics[mode][f"rollout/{name}"].append(value)
        for name, value in counts.items():
            self._rollout_metric_counts[mode][name] += value

    def _record_eval_rollouts(self, inputs: dict[str, Any]) -> None:
        """Retain this worker's generated evaluation text for deferred judging."""
        if not self.save_eval_rollout_records or not self._opd_eval_active:
            return
        indices = inputs.get("eval_example_index")
        prompts = inputs.get("original_prompt_text")
        completions = inputs.get("original_completion_text")
        if indices is None or prompts is None or completions is None:
            return

        for row, (index, prompt, completion) in enumerate(
            zip(indices, prompts, completions, strict=True)
        ):
            record: dict[str, Any] = {
                "eval_example_index": int(index),
                # One rollout per prompt; the snapshot writer and the judge dedupe on this key.
                "sample_index": 0,
                "prompt": str(prompt),
                "completion": str(completion),
            }
            # Record whichever metadata keys the active dataset provides.
            for key in ("data_type", "should_abstain", "prompt_style", "source"):
                if key in inputs:
                    record[key] = inputs[key][row]
            self._eval_rollout_records.append(record)

    def _save_eval_rollouts(self, metric_key_prefix: str, expected_count: int) -> None:
        if not self.save_eval_rollout_records:
            return
        gathered_records = gather_object([list(self._eval_rollout_records)])
        records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for process_records in gathered_records:
            for record in process_records:
                key = (
                    int(record["eval_example_index"]),
                    int(record["sample_index"]),
                )
                records_by_key.setdefault(key, record)
        if len(records_by_key) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} unique responses for {metric_key_prefix}, gathered "
                f"{len(records_by_key)}. Refusing to write an incomplete evaluation snapshot."
            )
        if not self.accelerator.is_main_process:
            return
        ordered_records = [records_by_key[key] for key in sorted(records_by_key)]
        save_eval_rollout_records(
            output_dir=self.args.output_dir,
            metric_key_prefix=metric_key_prefix,
            step=int(self.state.global_step),
            records=ordered_records,
            # Prompts were decoded with this tokenizer, so its family owns the turn markers.
            model_name=getattr(self.processing_class, "name_or_path", None),
        )

    @staticmethod
    def _rollout_count_names(counts: dict[str, int]) -> tuple[str, ...]:
        return rollout_count_names(counts)

    @staticmethod
    def _rollout_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
        return rollout_metrics_from_counts(counts)

    def _record_training_step_counts(self, step: int) -> None:
        """Sum and retain raw rollout counts across calls and workers for one optimizer step."""
        local_counts = dict(self._rollout_metric_counts["train"])
        self._rollout_metric_counts["train"].clear()

        gathered_counts = gather_object([local_counts])
        if not self.accelerator.is_main_process:
            return

        totals: defaultdict[str, int] = defaultdict(int)
        for process_counts in gathered_counts:
            for name, value in process_counts.items():
                totals[name] += value

        count_names = self._rollout_count_names(totals)
        if not count_names:
            return

        record: dict[str, Any] = {
            "step": int(step),
            "counts": {
                name: int(totals.get(name, 0))
                for name in count_names
            },
        }
        self._training_count_records.append(record)

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)

    def _flush_training_count_records(self) -> None:
        if not self.accelerator.is_main_process or not self._training_count_records:
            return

        records = self._training_count_records
        first_step = int(records[0]["step"])
        last_step = int(records[-1]["step"])
        path = (
            Path(self.args.output_dir)
            / "metrics"
            / "training"
            / f"metrics_steps_{first_step:08d}_{last_step:08d}.pt"
        )
        payload = {
            "kind": "training",
            "count_names": list(records[0]["counts"]),
            "records": records,
        }
        self._atomic_torch_save(payload, path)
        self._training_count_records = []

    def _save_eval_rollout_counts(self, metric_key_prefix: str) -> None:
        if not self.accelerator.is_main_process:
            return

        saved_counts = self._pending_eval_rollout_counts.pop(metric_key_prefix)
        if not saved_counts:
            return
        safe_prefix = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in metric_key_prefix
        )
        step = int(self.state.global_step)
        path = (
            Path(self.args.output_dir)
            / "metrics"
            / "eval"
            / f"{safe_prefix}_step_{step:08d}.pt"
        )
        payload = {
            "kind": "eval",
            "step": step,
            "metric_key_prefix": metric_key_prefix,
            "count_names": list(saved_counts),
            "counts": saved_counts,
        }
        self._atomic_torch_save(payload, path)

    def _log_kl_diagnostics(
        self,
        student_logprobs_sampled: torch.Tensor,
        teacher_logprobs_sampled: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        mode = "eval" if self._opd_eval_active else "train"
        diagnostics = kl_diagnostics(
            student_logprobs_sampled=student_logprobs_sampled,
            teacher_logprobs_sampled=teacher_logprobs_sampled,
            mask=mask,
        )
        for name, value in diagnostics.items():
            self._metrics[mode][f"diagnostics/{name}"].append(value.item())

    @staticmethod
    def _wrap_vllm_load_weights_for_text_only_hf_names(llm_model) -> None:
        inner_load_weights = llm_model.load_weights

        def remapped(weights):
            for name, tensor in weights:
                if name.startswith("model.") and not name.startswith(
                    ("model.language_model.", "model.visual.")
                ):
                    name = "model.language_model." + name.removeprefix("model.")
                yield name, tensor

        llm_model.load_weights = lambda weights: inner_load_weights(remapped(weights))

    @staticmethod
    def _repair_preexisting_transformers_modules_after_vllm_init(model: nn.Module) -> None:
        """Bypass vLLM's compile wrapper on HF modules created before ``LLM()``."""
        for module in model.modules():
            module_call = getattr(type(module), "__call__", None)
            if (
                getattr(module_call, "__module__", None) == "vllm.compilation.decorators"
                and not hasattr(module, "do_not_compile")
            ):
                module.do_not_compile = True

    def _ensure_vllm_weights_current(self) -> None:
        """Sync student weights into the colocated engine, waking it first in sleep mode."""
        if self._vllm_weights_current:
            return
        self._move_model_to_vllm()  # in sleep mode this wakes the "weights" allocation and reloads it
        self._wake_vllm_if_needed()  # in sleep mode this wakes the KV-cache allocation
        self._vllm_weights_current = True

    def _sleep_vllm_if_needed(self) -> None:
        if self.vllm_enable_sleep_mode:
            self.vllm_engine.sleep(level=2)
            # Level-2 sleep discards the engine weights; force a re-sync before the next generation.
            self._vllm_weights_current = False

    @staticmethod
    def _unpadded_prompt_token_lists(inputs: dict[str, Any]) -> list[list[int]]:
        """Recover each row's exact, contiguous prompt token ids from the left-padded block."""
        prompts = inputs["prompts"]
        prompt_mask = inputs["prompt_attention_mask"].bool()
        return [prompts[row][prompt_mask[row]].tolist() for row in range(prompts.shape[0])]

    def _generate_with_vllm_colocate(self, prompt_token_lists: list[list[int]]) -> list[list[int]]:
        """Generate completions from exact prompt token ids, avoiding GOLD's decode/re-tokenize round trip."""
        return generate_with_vllm_colocate(
            self.vllm_engine,
            prompt_token_lists,
            self.generation_config,
        )

    @staticmethod
    def _teacher_scoring_inputs(
        inputs: dict[str, Any],
        prompt_lengths: int,
    ) -> dict[str, Any]:
        """The teacher scores the student's own batch: the shared prompt block plus its rollout."""
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "prompt_lengths": prompt_lengths,
            "prompt_attention_mask": inputs["prompt_attention_mask"],
        }

    def _attach_vllm_rollouts(self, inputs: dict[str, Any], completion_token_lists: list[list[int]]) -> None:
        """Rebuild the collated batch around vLLM completions to preserve the left-padded prompt block."""
        prompts = inputs["prompts"]
        device = prompts.device
        pad_token_id = self.processing_class.pad_token_id
        batch_size, prompt_width = prompts.shape

        completion_width = max(1, max(len(token_ids) for token_ids in completion_token_lists))
        completions = torch.full(
            (batch_size, completion_width), pad_token_id, dtype=prompts.dtype, device=device
        )
        for row, token_ids in enumerate(completion_token_lists):
            if token_ids:
                completions[row, : len(token_ids)] = torch.tensor(
                    token_ids, dtype=prompts.dtype, device=device
                )

        new_input_ids = torch.cat([prompts, completions], dim=1)
        new_attention_mask = torch.ones_like(new_input_ids)
        new_attention_mask[new_input_ids == pad_token_id] = 0
        new_labels = torch.full_like(new_input_ids, -100)
        new_labels[:, prompt_width:] = new_input_ids[:, prompt_width:]
        new_labels[new_input_ids == pad_token_id] = -100

        prompt_mask = inputs["prompt_attention_mask"]
        prompt_texts = []
        completion_texts = []
        for row in range(batch_size):
            prompt_tokens = prompts[row][prompt_mask[row].bool()]
            prompt_texts.append(
                self.processing_class.decode(
                    prompt_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            completion_token_ids = truncate_token_ids_at_stop(
                completion_token_lists[row],
                self.generation_config.eos_token_id,
            )
            completion_texts.append(
                self.processing_class.decode(
                    completion_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["labels"] = new_labels
        inputs["original_prompt_text"] = prompt_texts
        inputs["original_completion_text"] = completion_texts

    def _prepare_on_policy_batches(self, batch_samples: list[dict[str, Any]]) -> None:
        """Generate every rollout in an optimizer step before its reduction denominator is counted."""
        if self.use_vllm:
            self._ensure_vllm_weights_current()
            per_batch_prompts = [self._unpadded_prompt_token_lists(inputs) for inputs in batch_samples]
            flat_prompts = [token_ids for batch in per_batch_prompts for token_ids in batch]
            flat_completions = self._generate_with_vllm_colocate(flat_prompts)
            offset = 0
            for inputs, batch_prompts in zip(batch_samples, per_batch_prompts):
                batch_completions = flat_completions[offset : offset + len(batch_prompts)]
                offset += len(batch_prompts)
                self._attach_vllm_rollouts(inputs, batch_completions)
                inputs["_opd_on_policy_prepared"] = True
            self._sleep_vllm_if_needed()

    def _count_loss_normalization_items(
        self,
        batch_samples: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        """Count the reduction units in one generated accumulation window."""
        if self.loss_reduction == "trajectory_mean":
            num_items_in_batch = torch.tensor(
                sum(batch["labels"].shape[0] for batch in batch_samples),
                device=device,
            )
        else:
            raise ValueError(f"Unknown loss_reduction: {self.loss_reduction!r}.")

        if (
            getattr(self.args, "average_tokens_across_devices", False)
            and getattr(self.accelerator, "num_processes", 1) > 1
        ):
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

        parallelism_config = getattr(self.accelerator, "parallelism_config", None)
        if parallelism_config is not None:
            num_items_in_batch = num_items_in_batch // parallelism_config.non_data_parallel_size

        return num_items_in_batch.clamp_min(1)

    def get_batch_samples(self, epoch_iterator, num_batches: int, device: torch.device):
        """Prefetch one accumulation window so its OPD reduction denominator is exact."""
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        if not batch_samples:
            return batch_samples, None

        self._prepare_on_policy_batches(batch_samples)
        return batch_samples, self._count_loss_normalization_items(batch_samples, device)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train on a prefetched rollout and extend GOLD's protocol-specific logs."""
        if inputs.pop("_opd_on_policy_prepared", False):
            loss = SFTTrainer.training_step(self, model, inputs, num_items_in_batch)
            self._textual_logs["prompt"].extend(gather_object(list(inputs["original_prompt_text"])))
            self._textual_logs["completion"].extend(gather_object(list(inputs["original_completion_text"])))

            loss_scalar = float(loss.detach())
            accumulation_steps = max(
                1,
                int(getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)),
            )
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += 1.0 / accumulation_steps

            if "gold_answer" in inputs:
                self._textual_logs["gold_answer"].extend(gather_object(list(inputs["gold_answer"])))
            return loss

        n_before = len(self._textual_logs["completion"])
        loss = super().training_step(model, inputs, None)
        if len(self._textual_logs["completion"]) > n_before:
            if "gold_answer" in inputs:
                self._textual_logs["gold_answer"].extend(
                    gather_object(list(inputs["gold_answer"]))
                )
        return loss

    @contextmanager
    def _isolated_eval_rng(self, metric_key_prefix: str):
        """Give recurring evaluation a fixed RNG stream without perturbing training."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

        # Do not use Python's randomized hash(): this offset must be stable across interpreter runs.
        split_offset = sum((index + 1) * ord(char) for index, char in enumerate(metric_key_prefix))
        process_offset = 1_000_003 * self.accelerator.process_index
        eval_seed = (int(self.args.seed) + split_offset + process_offset) % (2**63 - 1)

        try:
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                random.seed(eval_seed)
                np.random.seed(eval_seed % (2**32))
                torch.manual_seed(eval_seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Let Trainer recurse over named datasets; each leaf gets its own deterministic stream and persisted log.
        selected_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(selected_dataset, dict):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        with self._isolated_eval_rng(metric_key_prefix):
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        self._save_eval_rollout_counts(metric_key_prefix)
        self.save_state()
        return metrics

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Attach OPD's per-batch metrics to the standard, split-aware evaluation result."""
        self._metrics["eval"].clear()
        self._rollout_metric_counts["eval"].clear()
        self._eval_rollout_records = []
        self._new_api_recovery.reset()
        self._opd_eval_active = True
        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            if self.save_eval_rollout_records:
                self._save_eval_rollouts(metric_key_prefix, expected_count=len(dataloader.dataset))

            local_stats = {
                name: (float(sum(values)), len(values))
                for name, values in self._metrics["eval"].items()
                if values and not name.startswith("rollout/")
            }
            gathered_stats = gather_object([local_stats])
            totals: dict[str, list[float | int]] = {}
            for process_stats in gathered_stats:
                for name, (value_sum, value_count) in process_stats.items():
                    total = totals.setdefault(name, [0.0, 0])
                    total[0] += value_sum
                    total[1] += value_count

            for name, (value_sum, value_count) in totals.items():
                if value_count:
                    # Non-rollout diagnostics are means of per-batch scalar diagnostics.
                    output.metrics[f"{metric_key_prefix}_{name}"] = value_sum / value_count

            gathered_counts = gather_object([dict(self._rollout_metric_counts["eval"])])
            global_counts: defaultdict[str, int] = defaultdict(int)
            for process_counts in gathered_counts:
                for name, value in process_counts.items():
                    global_counts[name] += value
            count_names = self._rollout_count_names(global_counts)
            self._pending_eval_rollout_counts[metric_key_prefix] = {
                name: int(global_counts.get(name, 0))
                for name in count_names
            }
            for name, value in self._rollout_metrics_from_counts(global_counts).items():
                output.metrics[f"{metric_key_prefix}_rollout/{name}"] = value

            loss_key = f"{metric_key_prefix}_loss"
            if loss_key in output.metrics:
                output.metrics[f"{metric_key_prefix}_on_policy_loss"] = output.metrics[loss_key]
            if self._new_api_recovery.cohort:
                self._new_api_recovery.finish(
                    model=getattr(self, "model_wrapped", self.model),
                    accelerator=self.accelerator,
                    batch_size=self.args.per_device_eval_batch_size,
                    step=int(self.state.global_step),
                    objective=self.opd_loss_type,
                    metric_key_prefix=metric_key_prefix,
                    output_dir=self.args.output_dir,
                    save=self._atomic_torch_save,
                )
            return output
        finally:
            self._opd_eval_active = False
            self._new_api_recovery.reset()
            self._metrics["eval"].clear()
            self._rollout_metric_counts["eval"].clear()
            self._eval_rollout_records = []
            if self.use_vllm:
                self._sleep_vllm_if_needed()

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Generate an on-policy trajectory, then run the normal OPD loss without gradients."""
        with torch.no_grad():
            if self.use_vllm:
                # The engine stays awake across eval batches; evaluation_loop puts it back to sleep.
                self._ensure_vllm_weights_current()
                completion_token_lists = self._generate_with_vllm_colocate(
                    self._unpadded_prompt_token_lists(inputs)
                )
                self._attach_vllm_rollouts(inputs, completion_token_lists)

            self._record_eval_rollouts(inputs)

            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """GOLDTrainer.log with per-completion Action and Result columns in the completions table."""
        was_training = self.model.training
        orig_log_completions = self.log_completions
        self.log_completions = False
        super().log(logs, start_time)
        self.log_completions = orig_log_completions

        should_log = (
            was_training
            and orig_log_completions
            and self.accelerator.is_main_process
            and self.state.global_step % self.log_completion_steps == 0
        )
        if not should_log:
            return

        prompts = list(self._textual_logs["prompt"])
        completions = list(self._textual_logs["completion"])
        gold_answers = list(self._textual_logs["gold_answer"]) or None
        if gold_answers is not None:
            targets = gold_answers
            protocol = "calculator"
        else:
            # Only calculator batches carry answers to classify.
            return

        print_opd_completions_table(
            prompts,
            completions,
            targets,
            self.state.global_step,
            self.num_completions_to_print,
            protocol=protocol,
            thinking_prefilled=any(
                prompt_prefills_thinking(prompt) for prompt in prompts
            ),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._log_rollout_metrics(inputs)

        prompt_lengths = inputs["prompts"].shape[1]
        token_objective = self.opd_loss_type in ("tok_rkl", "tok_fkl", "tok_js", "tok_skl")

        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids_from_attention_mask(inputs["attention_mask"]),
        )
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]
        mask = shifted_labels != -100
        if self._opd_eval_active and self._new_api_recovery.cohort:
            self._new_api_recovery.record(inputs)

        teacher_inputs = self._teacher_scoring_inputs(inputs, prompt_lengths)
        scores = self.teacher_scorer.score(
            input_ids=teacher_inputs["input_ids"],
            attention_mask=teacher_inputs["attention_mask"],
            prompt_lengths=teacher_inputs["prompt_lengths"],
            prompt_attention_mask=teacher_inputs["prompt_attention_mask"],
            temperature=self.temperature,
            need_full_distribution=token_objective,
        )

        realized_ids = inputs["input_ids"][:, prompt_lengths:].unsqueeze(-1)
        student_logprobs_realized = logprobs_at_ids(
            shifted_student_logits,
            realized_ids,
            self.temperature,
        ).squeeze(-1)
        teacher_logprobs_realized = scores.realized_logprobs
        if self.opd_loss_type == "seq_rkl":
            loss = sequence_reverse_kl_loss(
                student_logprobs_sampled=student_logprobs_realized,
                teacher_logprobs_sampled=teacher_logprobs_realized,
                mask=mask,
                reduction=self.loss_reduction,
                normalization_count=num_items_in_batch,
            )
        elif token_objective:
            if scores.full_logits is None:
                raise RuntimeError("Token-level KL requires full teacher logits.")
            if self.opd_loss_type == "tok_skl":
                forward_loss = divergence_loss_full_vocab(
                    student_logits=shifted_student_logits,
                    teacher_logits=scores.full_logits,
                    mask=mask,
                    beta=0.0,
                    temperature=self.temperature,
                    reduction=self.loss_reduction,
                    normalization_count=num_items_in_batch,
                )
                reverse_loss = divergence_loss_full_vocab(
                    student_logits=shifted_student_logits,
                    teacher_logits=scores.full_logits,
                    mask=mask,
                    beta=1.0,
                    temperature=self.temperature,
                    reduction=self.loss_reduction,
                    normalization_count=num_items_in_batch,
                )
                loss = 0.5 * (forward_loss + reverse_loss)
            else:
                if self.opd_loss_type == "tok_fkl":
                    beta = 0.0
                elif self.opd_loss_type == "tok_rkl":
                    beta = 1.0
                else:
                    beta = self.beta
                loss = divergence_loss_full_vocab(
                    student_logits=shifted_student_logits,
                    teacher_logits=scores.full_logits,
                    mask=mask,
                    beta=beta,
                    temperature=self.temperature,
                    reduction=self.loss_reduction,
                    normalization_count=num_items_in_batch,
                )
        else:
            raise ValueError(f"Unknown opd_loss_type: {self.opd_loss_type!r}")

        uses_global_mean = (
            self.loss_reduction == "trajectory_mean"
            and getattr(self.args, "average_tokens_across_devices", False)
        )
        if num_items_in_batch is not None and uses_global_mean:
            # DDP averages gradients across ranks.
            parallelism_config = getattr(self.accelerator, "parallelism_config", None)
            non_data_parallel_size = (
                parallelism_config.non_data_parallel_size
                if parallelism_config is not None
                else 1
            )
            data_parallel_size = self.accelerator.num_processes // non_data_parallel_size
            loss = loss * data_parallel_size

        self._log_kl_diagnostics(
            student_logprobs_realized,
            teacher_logprobs_realized,
            mask,
        )

        empty_cache()
        return (loss, outputs_student) if return_outputs else loss
