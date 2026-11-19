# adapted from https://github.com/idanshen/Self-Distillation

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DistilConfig(TrainingArguments):
    r"""
    Configuration class for the [`DistilTrainer`].

    This class includes only the parameters that are specific to Distil training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes * per_device_batch_size
            * gradient_accumulation_steps) must be evenly divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        generation_batch_size: (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation: (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step and woken
            for weight sync and generation.

        > Parameters that control the training

        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step "
            "and woken for weight sync and generation."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    alpha: float = field(
        default=0.0,
        metadata={
            "help": "Alpha coefficient. If `0.0` (default), the forward KL is used. If `1.0`, the reverse KL is used."
        },
    )
    teacher_eos_remap: Optional[str] = field(
        default=None,
        metadata={"help": "'SRC:DST' token ids. Show the teacher DST at the student's first SRC per "
                          "completion and read the teacher's DST column in the SRC slot. "
                          "151643:151645 for a Base student with a post-trained Qwen3 teacher."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    enable_thinking: str = field(
        default="default",
        metadata={
            "help": "Control thinking mode passed to the chat template during training. "
            "'true' = enable thinking, 'false' = disable thinking, "
            "'default' = don't pass enable_thinking (uses tokenizer default, i.e. thinking on for Qwen3)."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )
