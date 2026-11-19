from __future__ import annotations

import torch

from utils.tensor_utils import reverse_cumsum


def kl_diagnostics(
    student_logprobs_sampled: torch.Tensor,
    teacher_logprobs_sampled: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Sampled KL and REINFORCE weight stds, for diagnostics only."""
    with torch.no_grad():
        mask_f = mask.to(student_logprobs_sampled.dtype)
        reward = (teacher_logprobs_sampled - student_logprobs_sampled) * mask_f
        denom = mask_f.sum().clamp_min(1.0)

        tok_kl = -(reward.sum() / denom)

        per_seq_reward = reward.sum(dim=-1)
        has_tokens = mask_f.sum(dim=-1) > 0
        seq_kl = (
            -per_seq_reward[has_tokens].mean()
            if has_tokens.any()
            else torch.zeros_like(tok_kl)
        )

        reward_vals = reward[mask]
        tok_weight_std = (
            reward_vals.std() if reward_vals.numel() > 1 else torch.zeros_like(tok_kl)
        )
        seq_weight_vals = reverse_cumsum(reward)[mask]
        seq_weight_std = (
            seq_weight_vals.std()
            if seq_weight_vals.numel() > 1
            else torch.zeros_like(tok_kl)
        )

    return {
        "tok_kl": tok_kl,
        "seq_kl": seq_kl,
        "tok_weight_std": tok_weight_std,
        "seq_weight_std": seq_weight_std,
    }
