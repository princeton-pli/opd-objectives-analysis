from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from autoreg_transformer.model import AutoregressiveTransformerStudentPolicy
from core.environments import NaryTreeNavigationEnvironment, State


def teacher_distribution_torch(
        states: torch.Tensor,
        cfg,
        env: NaryTreeNavigationEnvironment,
) -> torch.Tensor:
    return _tree_teacher_table(env, cfg.teacher_correct_prob, states.device)[states]


@lru_cache(maxsize=16)
def _tree_teacher_table(
        env: NaryTreeNavigationEnvironment,
        correct_prob: float,
        device: torch.device,
) -> torch.Tensor:
    """Cache the fixed teacher by environment, target configuration, and device."""
    rows = np.asarray([
        env.teacher_distribution_np(state, correct_prob)
        for state in env.all_states()
    ])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def sample_student_rollouts(
        policy: AutoregressiveTransformerStudentPolicy,
        cfg,
        env: NaryTreeNavigationEnvironment,
        num_rollouts: int,
        *,
        keep_grad: bool,
) -> Dict[str, object]:
    """Sample outputs from the student.

    With keep_grad, the log probabilities of the sampled tokens (logp_student_by_step) are
    recomputed in one causal pass that tracks gradients.
    """
    device = next(policy.parameters()).device
    track_grad = keep_grad and torch.is_grad_enabled()
    states_by_step: List[torch.Tensor] = []
    prefixes_by_step: List[torch.Tensor] = []
    prefix_lengths_by_step: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    logp_teacher_by_step: List[torch.Tensor] = []
    active_by_step: List[torch.Tensor] = []
    states = torch.full((num_rollouts,), env.start_state, dtype=torch.long, device=device)
    prefixes = torch.zeros((num_rollouts, env.depth), dtype=torch.long, device=device)
    prefix_lengths = torch.zeros(num_rollouts, dtype=torch.long, device=device)
    first_leaf_state = env.first_state_at_level(env.depth)

    # Every root-to-leaf path has exactly depth actions.
    for step in range(env.depth):
        active = (states < first_leaf_state).float()
        states_by_step.append(states)
        current_prefixes = prefixes[:, :step].clone()
        prefixes_by_step.append(current_prefixes)
        prefix_lengths_by_step.append(prefix_lengths.clone())
        teacher_probs = teacher_distribution_torch(states, cfg, env)

        with torch.no_grad():
            student_logits = policy.logits(current_prefixes, prefix_lengths)
            actions = torch.distributions.Categorical(logits=student_logits).sample()
            selected_teacher_probs = teacher_probs.gather(1, actions[:, None]).squeeze(1).clamp_min(1e-12)
            logp_teacher = selected_teacher_probs.log()

        actions_by_step.append(actions.detach())
        logp_teacher_by_step.append(logp_teacher.detach())
        active_by_step.append(active)

        states = torch.where(active.bool(), env.num_children * states + 1 + actions, states)
        prefixes[:, step] = actions
        prefix_lengths = prefix_lengths + 1

    state_tensor = torch.stack(states_by_step, dim=1)
    rollouts = {
        "states_by_step": state_tensor.T.cpu().tolist(),
        "state_tensor": state_tensor,
        "final_prefixes": prefixes,
        "prefixes_by_step": prefixes_by_step,
        "prefix_lengths_by_step": prefix_lengths_by_step,
        "actions_by_step": actions_by_step,
        "logp_teacher_by_step": logp_teacher_by_step,
        "active_by_step": active_by_step,
    }
    if keep_grad:
        with torch.set_grad_enabled(track_grad):
            logits = trajectory_logits(policy, rollouts)
            actions = torch.stack(actions_by_step, dim=1)
            logp_student = logits.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        rollouts["logp_student_by_step"] = list(logp_student.unbind(dim=1))
    return rollouts


def sample_teacher_rollouts(cfg, env: NaryTreeNavigationEnvironment, num_rollouts: int) -> None:
    """Sample outputs from the teacher and discard them.

    The runs reported in the paper computed metrics on outputs sampled from the teacher at
    every evaluation. These metrics are not needed, but sampling the outputs keeps the random
    number generator, and thus training, identical to those runs.
    """
    states = torch.full((num_rollouts,), env.start_state, dtype=torch.long)
    first_leaf_state = env.first_state_at_level(env.depth)
    for _ in range(env.depth):
        active = (states < first_leaf_state).float()
        teacher_probs = teacher_distribution_torch(states, cfg, env)
        actions = torch.distributions.Categorical(probs=teacher_probs).sample()
        states = torch.where(active.bool(), env.num_children * states + 1 + actions, states)


def trajectory_logits(policy: AutoregressiveTransformerStudentPolicy, rollouts: Dict[str, object]) -> torch.Tensor:
    """Score completed trajectories in one causal pass; return [batch, time, actions]."""
    prefixes = rollouts["final_prefixes"]
    lengths_by_step = rollouts["prefix_lengths_by_step"]
    assert isinstance(prefixes, torch.Tensor)
    assert isinstance(lengths_by_step, list)
    # The final action is not needed to predict any action.
    inputs = prefixes[:, :len(lengths_by_step) - 1]
    logits = policy.all_logits(inputs)
    positions = torch.stack(lengths_by_step, dim=1)
    return logits.gather(1, positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))


def active_token_distributions(
        policy: AutoregressiveTransformerStudentPolicy,
        cfg,
        env: NaryTreeNavigationEnvironment,
        rollouts: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = trajectory_logits(policy, rollouts)
    active = torch.stack(rollouts["active_by_step"]).bool()
    states = rollouts["state_tensor"]
    assert isinstance(states, torch.Tensor)
    # Keep the existing time-major order without expanding any prefixes.
    student_probs = logits.transpose(0, 1)[active].softmax(dim=-1)
    teacher_probs = teacher_distribution_torch(states.T[active], cfg, env)
    return student_probs, teacher_probs


def state_action_prefix(env: NaryTreeNavigationEnvironment, state: State) -> Tuple[int, ...]:
    actions: List[int] = []
    while state != env.start_state:
        parent = env.parent(state)
        actions.append(state - env.first_child(parent))
        state = parent
    return tuple(reversed(actions))


def prefixes_for_states(
        env: NaryTreeNavigationEnvironment,
        states: Sequence[State],
) -> Tuple[torch.Tensor, torch.Tensor]:
    prefixes = [state_action_prefix(env, state) for state in states]
    lengths = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long)
    max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
    padded = torch.zeros((len(prefixes), max_length), dtype=torch.long)
    for row, prefix in enumerate(prefixes):
        if prefix:
            padded[row, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
    return padded, lengths
