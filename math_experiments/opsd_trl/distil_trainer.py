# adapted from https://github.com/idanshen/Self-Distillation

import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional, Union

import datasets
import torch
import torch.utils.checkpoint
import torch.utils.data
from accelerate import logging
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from trl.data_utils import maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp
from trl.trainer.base_trainer import BaseTrainer
from .distil_config import DistilConfig
from trl.trainer.utils import (
    RepeatSampler,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    pad,
    shuffle_sequence_dict,
    split_tensor_dict,
)
from torch.nn.functional import log_softmax, kl_div


if is_vllm_available():
    from vllm import LLM, SamplingParams


logger = logging.get_logger(__name__)


def _sel_logps_chunk(logits_chunk: torch.Tensor, index_chunk: torch.Tensor) -> torch.Tensor:
    """log_softmax -> gather for one sequence slice. Kept as a module-level fn so
    torch.utils.checkpoint can recompute it without capturing `self`."""
    lp = torch.nn.functional.log_softmax(logits_chunk.float(), dim=-1)
    return lp.gather(dim=-1, index=index_chunk.unsqueeze(-1)).squeeze(-1)


def chunked_selective_log_softmax(logits: torch.Tensor, index: torch.Tensor,
                                  chunk: int = 1024) -> torch.Tensor:
    """Memory-lean equivalent of trl.selective_log_softmax.

    log_softmax sits on autocast's float32 promotion list, so on a (1, S, V) bf16
    logits tensor it materializes a float32 (S, V) intermediate -- 9.96 GB at
    S=16384, V=151936 -- and autograd retains it for backward. TRL's helper loops
    over the BATCH dim, which is a no-op at batch size 1.

    Here we slice along the SEQUENCE dim and wrap each slice in
    torch.utils.checkpoint, so only the bf16 logits (already retained by the
    lm_head graph) plus the tiny (B, chunk) outputs survive into backward; the
    float32 intermediates are freed and recomputed. Same ops, same values, same
    gradients -- only the retention schedule changes.
    """
    outs = []
    S = logits.size(1)
    for i in range(0, S, chunk):
        lg = logits[:, i:i + chunk, :]
        ix = index[:, i:i + chunk]
        if torch.is_grad_enabled() and lg.requires_grad:
            o = torch.utils.checkpoint.checkpoint(_sel_logps_chunk, lg, ix, use_reentrant=False)
        else:
            o = _sel_logps_chunk(lg, ix)
        outs.append(o)
    return torch.cat(outs, dim=1)


class DistilTrainer(BaseTrainer):
    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ):
        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        # Multi-step
        self.num_iterations = args.num_iterations
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO-like algorithms, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        if hasattr(model, "warnings_issued"):          # removed in transformers 5
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.alpha = args.alpha
        self.teacher_eos_remap = None
        if getattr(args, "teacher_eos_remap", None):
            a, b = (int(x) for x in str(args.teacher_eos_remap).split(":"))
            self.teacher_eos_remap = (a, b)
        self.ref_model = ref_model

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                vllm_model_path = model.name_or_path
                self.llm = LLM(
                    model=vllm_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                # Student and teacher can be different architectures (cross-tokenizer). accelerate raises if any
                # listed layer class is missing from a model, and caches the policy after the first call, so give
                # each model only the listed classes it contains: teacher now, student (wrapped later in train()).
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                from accelerate.utils.dataclasses import get_module_class_from_name
                _plugin = self.accelerator.state.fsdp_plugin
                _names = list(_plugin.transformer_cls_names_to_wrap or [])
                def _restrict(m):
                    if _names:
                        _plugin.transformer_cls_names_to_wrap = [n for n in _names if get_module_class_from_name(m, n) is not None]
                        _plugin.auto_wrap_policy = transformer_auto_wrap_policy
                _restrict(self.ref_model)
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
                _restrict(self.model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_all_logps=True,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = chunked_selective_log_softmax(logits, completion_ids)  # compute logprobs
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return selected_logps, logps, entropies


    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # When world_size=1, FSDP falls back to NO_SHARD mode and parameters are never
        # actually sharded. Calling summon_full_params on non-root FSDP sub-modules
        # triggers an _is_root assertion error in PyTorch's FSDP1 runtime.
        # Bypass FSDP machinery and sync parameters directly in this case.
        if self.accelerator.num_processes == 1:
            for name, param in module.named_parameters():
                name = self._fix_param_name_to_vllm(name, extra_prefixes=["_fsdp_wrapped_module."])
                if self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(name, param.data)])
            return

        # For FSDP1 with actual sharding, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        model_to_sync = self.model
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # For non-PEFT models, simply gather (if needed) and update each parameter individually.
        if self.is_fsdp_enabled:
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
            if fsdp_version == 1:
                self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
            elif fsdp_version == 2:
                self._sync_fsdp2_params_to_vllm(model_to_sync)
        else:
            for name, param in model_to_sync.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = generation_batches
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_single_turn(self, prompts: list[str]):
        chat_template_kwargs = {}
        if self.args.enable_thinking != "default":
            chat_template_kwargs["enable_thinking"] = (self.args.enable_thinking == "true")

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in prompts
        ]

        # Generate completions using vLLM
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            should_sync = self.state.global_step != self._last_loaded_step
            if should_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            if self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                else:
                    all_prompts_text = prompts_text

                vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        return prompt_ids, completion_ids

    def _generate(self, prompts: list[str]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids = self._generate_single_turn(prompts)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        # Process student prompts (always used for student training, regardless of generation source)
        chat_template_kwargs = {}
        if self.args.enable_thinking != "default":
            chat_template_kwargs["enable_thinking"] = (self.args.enable_thinking == "true")

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in prompts
        ]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        if self.use_vllm:
            self.processing_class.truncation_side = "right"

        (
            _generation_prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        return output


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        
        loss_completion_mask = completion_mask
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_completion_ids = completion_ids
        if getattr(self.args, "teacher_eos_remap", False):
            # A Base student terminates with <|endoftext|> (151643); a post-trained Qwen3 teacher
            # expects <|im_end|> (151645) and assigns ~e^-23 to the student's stop token, which
            # trains the student never to stop (mean length -> cap within ~100 steps). Show the
            # teacher <|im_end|> at the FIRST 151643 of each completion only: pad is also 151643.
            # Mirrors EOPD's _replace_eos_token_for_teacher. Student-side tensors are untouched.
            src_id, dst_id = self.teacher_eos_remap
            teacher_completion_ids = completion_ids.clone()
            hit = (teacher_completion_ids == src_id)
            first = hit.int().cumsum(1) == 1
            teacher_completion_ids[hit & first] = dst_id
        teacher_input_ids = torch.cat([prompt_ids, teacher_completion_ids], dim=1)
        teacher_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
        )

        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model, teacher_input_ids, teacher_attention_mask,
                logits_to_keep, compute_entropy=True,
            )
            if getattr(self.args, "teacher_eos_remap", False) and teacher_all_logps is not None:
                src_id, dst_id = self.teacher_eos_remap
                teacher_all_logps[..., [src_id, dst_id]] = teacher_all_logps[..., [dst_id, src_id]]

        # Compute KL divergences between student and teacher distributions.
        # PyTorch kl_div differs from the standard mathematical definition,
        # so the order of probability distributions is swapped vs. the paper.
        # Full vocab path: chunked over sequence dimension
        _kl_chunk = 256  # tokens per chunk
        S = all_logps.size(1)
        if self.alpha == 0:  # Forward KL
            per_token_loss_parts = []
            for i in range(0, S, _kl_chunk):
                per_token_loss_parts.append(
                    kl_div(all_logps[:, i:i+_kl_chunk], teacher_all_logps[:, i:i+_kl_chunk],
                           reduction="none", log_target=True).sum(-1)
                )
            per_token_loss = torch.cat(per_token_loss_parts, dim=1)
        elif self.alpha == 1:  # Reverse KL
            per_token_loss_parts = []
            for i in range(0, S, _kl_chunk):
                per_token_loss_parts.append(
                    kl_div(teacher_all_logps[:, i:i+_kl_chunk], all_logps[:, i:i+_kl_chunk],
                           reduction="none", log_target=True).sum(-1)
                )
            per_token_loss = torch.cat(per_token_loss_parts, dim=1)
        del all_logps, teacher_all_logps  # free the large vocab-sized tensors

        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())
        
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
