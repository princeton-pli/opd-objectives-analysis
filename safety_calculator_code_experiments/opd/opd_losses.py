from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.tensor_utils import reverse_cumsum


def _reduce_loss(
    per_trajectory: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str,
    normalization_count: int | torch.Tensor | None,
) -> torch.Tensor:
    """Reduce per-trajectory loss sums (shape ``[batch]``) to a scalar."""
    if reduction == "trajectory_mean":
        local_count = mask.shape[0]
    else:
        raise NotImplementedError(f"Unknown loss reduction: {reduction}")

    loss_sum = per_trajectory.sum()
    if normalization_count is None:
        denominator = torch.as_tensor(local_count, device=loss_sum.device, dtype=loss_sum.dtype)
    else:
        denominator = torch.as_tensor(normalization_count, device=loss_sum.device, dtype=loss_sum.dtype)
    return loss_sum / denominator.clamp_min(1.0)


def sequence_reverse_kl_loss(
    student_logprobs_sampled: torch.Tensor,
    teacher_logprobs_sampled: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "trajectory_mean",
    normalization_count: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """REINFORCE estimator for sequence reverse KL using reward to go."""
    mask_f = mask.to(student_logprobs_sampled.dtype)
    reward = (teacher_logprobs_sampled - student_logprobs_sampled) * mask_f
    weight = reverse_cumsum(reward)
    per_token = weight.detach() * student_logprobs_sampled * mask_f
    return _reduce_loss(
        -per_token.sum(dim=-1),
        mask,
        reduction=reduction,
        normalization_count=normalization_count,
    )


def logprobs_at_ids(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Gather full-vocabulary log probabilities without materializing log-softmax."""
    scaled_logits = logits if temperature == 1.0 else logits / temperature
    return scaled_logits.gather(-1, token_ids) - torch.logsumexp(
        scaled_logits,
        dim=-1,
        keepdim=True,
    )


def _generalized_jsd_from_logprobs(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, beta: float
) -> torch.Tensor:
    """
    beta=0 -> KL(teacher || student) (tok_fkl);
    beta=1 -> KL(student || teacher) (tok_rkl, closed form)
    0 < beta < 1 -> tok_js

    Taken directly from TRL's GOLDTrainer.generalized_jsd_loss.
    Refactored to take log-probs rather than logits.
    """
    if beta == 0:
        return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    if beta == 1:
        return F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)

    beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
    )
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
    return beta * kl_teacher + (1 - beta) * kl_student


def divergence_loss_full_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    temperature: float = 1.0,
    reduction: str = "trajectory_mean",
    normalization_count: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact generalized-JS loss from full-vocabulary student and teacher logits."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_position = _generalized_jsd_from_logprobs(student_log_probs, teacher_log_probs, beta)
    mask_f = mask.to(per_position.dtype)
    return _reduce_loss(
        (per_position * mask_f).sum(dim=-1),
        mask,
        reduction=reduction,
        normalization_count=normalization_count,
    )
