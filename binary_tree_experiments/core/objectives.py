"""Objective losses of the synthetic OPD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from autoreg_transformer.rollouts import active_token_distributions, sample_student_rollouts


@dataclass(frozen=True)
class Objective:
    name: str
    seed_offset: int


def get_objective_loss(objective_name: str) -> Callable[[object, object, object], torch.Tensor]:
    if objective_name == "seq_reverse_kl":
        return seq_reverse_kl_loss
    if objective_name == "token_sample_reverse_kl":
        return token_sample_reverse_kl_loss
    if objective_name == "token_reverse_kl":
        return partial(_exact_token_divergence_loss, divergence_kind="reverse_kl")
    if objective_name == "token_forward_kl":
        return partial(_exact_token_divergence_loss, divergence_kind="forward_kl")
    if objective_name == "token_js":
        return partial(_exact_token_divergence_loss, divergence_kind="js")
    raise ValueError(f"unknown objective: {objective_name}")


def seq_reverse_kl_loss(policy, cfg, env) -> torch.Tensor:
    rollouts = sample_student_rollouts(policy, cfg, env, cfg.num_rollouts, keep_grad=True)
    active_by_step = rollouts["active_by_step"]
    logp_student_steps = rollouts["logp_student_by_step"]
    logp_teacher_steps = rollouts["logp_teacher_by_step"]

    # [batch, positions]; inactive positions (past a leaf) are masked out.
    active = torch.stack(active_by_step, dim=1)
    logp_student = torch.stack(logp_student_steps, dim=1)
    logp_teacher = torch.stack(logp_teacher_steps, dim=1)
    step_log_ratio = active * (logp_student - logp_teacher)

    # Reward-to-go: the token at each position is weighted by the log ratios of itself and
    # of the subsequent tokens, excluding those of preceding tokens (an unbiased estimate of
    # the gradient of the sequence-level RKL).
    weights = step_log_ratio.detach().flip(1).cumsum(dim=1).flip(1)
    return (active * weights * logp_student).sum(dim=1).mean()


def token_sample_reverse_kl_loss(policy, cfg, env) -> torch.Tensor:
    """Token-level RKL with a gradient estimate that depends only on the sampled tokens (Appendix E.1).

    Each token is weighted by its own log ratio only, which gives an unbiased estimate of the
    gradient of the RKL at its prefix (sequence-level OPD also adds the log ratios of the
    subsequent tokens).
    """
    rollouts = sample_student_rollouts(policy, cfg, env, cfg.num_rollouts, keep_grad=True)
    terms = []
    for logp_student, logp_teacher, active in zip(
            rollouts["logp_student_by_step"],
            rollouts["logp_teacher_by_step"],
            rollouts["active_by_step"],
    ):
        log_ratio = logp_student - logp_teacher
        terms.append(active * log_ratio.detach() * logp_student)
    return torch.stack([term.sum() for term in terms]).sum() / float(cfg.num_rollouts)


def _exact_token_divergence_loss(policy, cfg, env, divergence_kind: str) -> torch.Tensor:
    rollouts = sample_student_rollouts(policy, cfg, env, cfg.num_rollouts, keep_grad=False)
    student_probs, teacher_probs = active_token_distributions(policy, cfg, env, rollouts)
    per_state = exact_divergence(student_probs, teacher_probs, divergence_kind)
    return per_state.sum() / cfg.num_rollouts


def exact_divergence(student_probs: torch.Tensor, teacher_probs: torch.Tensor, type: str) -> torch.Tensor:
    p = student_probs
    q = teacher_probs

    if type == "reverse_kl":
        mask = p > 0
        terms = torch.zeros_like(p)
        terms[mask] = p[mask] * (p[mask].log() - q[mask].log())
        return torch.sum(terms, dim=-1)

    if type == "forward_kl":
        mask = q > 0
        terms = torch.zeros_like(q)
        terms[mask] = q[mask] * (q[mask].log() - p[mask].log())
        return torch.sum(terms, dim=-1)

    if type == "js":
        m = (p + q).mul(0.5)
        p_mask = p > 0
        q_mask = q > 0
        p_terms = torch.zeros_like(p)
        q_terms = torch.zeros_like(q)
        p_terms[p_mask] = p[p_mask] * (p[p_mask].log() - m[p_mask].log())
        q_terms[q_mask] = q[q_mask] * (q[q_mask].log() - m[q_mask].log())
        return 0.5 * torch.sum(p_terms, dim=-1) + 0.5 * torch.sum(q_terms, dim=-1)

    raise ValueError(f"unknown divergence kind: {type}")
