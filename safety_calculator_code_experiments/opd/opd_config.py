from dataclasses import dataclass, field
from typing import Optional, get_args

from trl import SFTConfig
from trl.experimental.gold import GOLDConfig

from utils.strings import (
    LossReduction,
    LossType,
    TrainingAlgorithm,
)


@dataclass
class OPDConfig(GOLDConfig):
    algorithm: TrainingAlgorithm = field(
        default="opd",
        metadata={"help": "Training algorithm: on-policy distillation (opd) or off-policy SFT (sft)."},
    )
    lmbda: float = field(
        default=1.0,
        metadata={"help": "Fraction of steps trained on-policy (student rollouts scored by the teacher)"})
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling and distillation temperature."})
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability; 1.0 disables top-p truncation."})
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling; 0 disables top-k truncation."})
    loss_reduction: LossReduction = field(
        default="trajectory_mean",
        metadata={
            "help": (
                "OPD loss reduction over the complete gradient-accumulation window: "
                "trajectory_mean (write-up objective)."
            )
        })
    opd_loss_type: LossType = field(
        default="tok_rkl",
        metadata={
            "help": (
                "Which OPD objective to optimize: seq_rkl, tok_rkl, tok_fkl, tok_js, or tok_skl."
            )
        })
    teacher_device: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Put the in-process teacher on this device (e.g. 'cuda:1') instead of the student's, "
                "for teachers that do not fit next to the trained student. Single-process runs only. "
                "Default: the student's device."
            )
        })
    enable_thinking: Optional[bool] = field(
        default=False)
    log_rollout_metrics: bool = field(
        default=True)
    evals_per_epoch: int = field(
        default=0,
        metadata={
            "help": (
                "Evaluate when training first crosses each 1/N epoch boundary, where N is "
                "this value. Zero disables fractional-epoch evaluation."
            )
        },
    )
    save_eval_rollout_records: bool = field(
        default=False,
        metadata={
            "help": (
                "Save full, globally deduplicated evaluation prompts and completions under "
                "<output_dir>/responses/eval for deferred judging."
            )
        },
    )
    metrics_flush_steps: int = field(
        default=20)
    new_api_recovery_screen_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Teacher scores written by opd/code_completion/build_screen.py. During each "
                "evaluation, score the first token of each API name and persist recovery rows."
            )
        },
    )
    save_final_model: bool = field(
        default=False,
        metadata={
            "help": "Save final model weights even when save_strategy='no'."
        },
    )

    def __post_init__(self):
        if self.algorithm not in get_args(TrainingAlgorithm):
            raise ValueError(
                f"Unknown algorithm: {self.algorithm!r}. Expected one of {get_args(TrainingAlgorithm)}."
            )
        if self.save_eval_rollout_records and self.eval_strategy == "no":
            raise ValueError(
                "save_eval_rollout_records requires periodic evaluation; set eval_strategy to "
                "'steps' or 'epoch'."
            )
        if self.evals_per_epoch < 0:
            raise ValueError(
                f"evals_per_epoch must be nonnegative, got {self.evals_per_epoch}."
            )
        if self.evals_per_epoch > 0 and self.eval_strategy == "no":
            raise ValueError(
                "evals_per_epoch requires periodic evaluation; set eval_strategy to "
                "'steps' or 'epoch'."
            )
        if self.algorithm == "sft":
            # GOLDConfig's validation is OPD-only; run the SFT base validation for the SFT path.
            SFTConfig.__post_init__(self)
            self.completion_only_loss = True
            if self.packing:
                raise ValueError(
                    "packing=True changes the example stream and is incompatible with an exact "
                    "OPD/SFT prompt-order comparison."
                )
            return

        super().__post_init__()

        if self.lmbda < 1.0:
            raise NotImplementedError(
                "OPDTrainer currently supports only fully on-policy training (lmbda=1.0); "
                "the off-policy padding/template alignment path has not been fixed."
            )
        if not self.use_vllm:
            raise NotImplementedError("OPDTrainer generates its rollouts with vLLM only; set use_vllm=True.")
        if self.use_vllm:
            if self.vllm_mode != "colocate":
                raise NotImplementedError(
                    "use_vllm=True requires vllm_mode='colocate'. TRL's server-mode VLLMClient only "
                    "accepts decoded prompt text, which re-tokenizes on the server and does not "
                    "preserve chat-template control tokens."
                )
            if self.vllm_tensor_parallel_size != 1:
                raise NotImplementedError(
                    "OPDTrainer's student vLLM generation currently supports vllm_tensor_parallel_size=1 only."
                )
        if self.max_length is not None and self.max_completion_length > self.max_length:
            raise ValueError(
                f"max_completion_length ({self.max_completion_length}) cannot exceed max_length ({self.max_length})."
            )
        if self.metrics_flush_steps < 1:
            raise ValueError(f"metrics_flush_steps must be >= 1, got {self.metrics_flush_steps}.")
        if self.new_api_recovery_screen_path is not None and self.eval_strategy == "no":
            raise ValueError(
                "new_api_recovery_screen_path requires periodic evaluation; set eval_strategy "
                "to 'steps' or 'epoch'."
            )

        if self.loss_reduction not in get_args(LossReduction):
            raise ValueError(
                f"Unknown loss_reduction: {self.loss_reduction!r}. Expected one of {get_args(LossReduction)}."
            )

        if self.teacher_model_name_or_path is None:
            raise ValueError("teacher_model_name_or_path (from GOLDConfig) must be set.")
        if self.opd_loss_type == "tok_js" and not (0.0 < self.beta < 1.0):
            raise ValueError(f"opd_loss_type='tok_js' requires 0 < beta < 1, got beta={self.beta}.")
