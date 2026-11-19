from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

from autoreg_transformer.model import AutoregressiveTransformerStudentPolicy
from autoreg_transformer.rollouts import (
    prefixes_for_states,
    sample_student_rollouts,
    sample_teacher_rollouts,
    teacher_distribution_torch,
)
from core.environments import NaryTreeNavigationConfig, NaryTreeNavigationEnvironment, build_environment
from core.objectives import Objective, exact_divergence, get_objective_loss
from utils.synth_opd_exps_output_utils import (
    History,
    make_output_dir,
    write_config,
    write_histories_csv,
    write_sampled_trajectory_divergence_profile_csv,
)


@dataclass(frozen=True)
class AutoregressiveTransformerPolicyConfig:
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int


@dataclass(frozen=True)
class Config:
    env_config: NaryTreeNavigationConfig
    policy_config: AutoregressiveTransformerPolicyConfig
    teacher_correct_prob: float
    objectives: Tuple[Objective, ...]
    seed: int
    training_steps: int
    num_rollouts: int
    learning_rate: float
    log_every: int
    eval_rollouts: int
    eval_at_steps: Tuple[int, ...]
    output_root: Path


def load_run_config(config_path: Path) -> Config:
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    return Config(
        env_config=NaryTreeNavigationConfig(**config_data.pop("env_config")),
        policy_config=AutoregressiveTransformerPolicyConfig(**config_data.pop("policy_config")),
        objectives=tuple(Objective(**item) for item in config_data.pop("objectives")),
        eval_at_steps=tuple(config_data.pop("eval_at_steps")),
        output_root=Path(config_data.pop("output_root")),
        **config_data,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy(
        config: AutoregressiveTransformerPolicyConfig,
        env: NaryTreeNavigationEnvironment,
) -> AutoregressiveTransformerStudentPolicy:
    return AutoregressiveTransformerStudentPolicy(
        num_actions=env.num_actions,
        max_prefix_length=env.depth,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
    )


@torch.no_grad()
def exact_success_probability(
        policy: AutoregressiveTransformerStudentPolicy,
        env: NaryTreeNavigationEnvironment,
) -> float:
    """Exact student probability of the path to the target leaf, without sampling."""
    goal_path = env.path_to_goal()
    path = env.reference_path()
    goal_actions = torch.tensor(
        [next_state - env.first_child(state) for state, next_state in zip(path, goal_path[1:])],
        dtype=torch.long,
    )
    prefixes, prefix_lengths = prefixes_for_states(env, path)
    student_probs = policy.probs(prefixes, prefix_lengths)
    return float(student_probs.gather(1, goal_actions[:, None]).prod())


def evaluate(
        policy: AutoregressiveTransformerStudentPolicy,
        cfg: Config,
        env: NaryTreeNavigationEnvironment,
) -> Dict[str, float]:
    """Exact success probability and probability of each token at the root."""
    with torch.no_grad():
        sample_student_rollouts(policy, cfg, env, cfg.eval_rollouts, keep_grad=False)
    stats = {"exact_success_probability": exact_success_probability(policy, env)}
    root_probs = policy.probs(torch.zeros((1, 0), dtype=torch.long), torch.zeros(1, dtype=torch.long))
    for child, probability in enumerate(root_probs.squeeze(0).detach().cpu().tolist()):
        stats[f"root_child_{child}_probability"] = float(probability)
    sample_teacher_rollouts(cfg, env, cfg.eval_rollouts)
    return stats


def reverse_kl_profile(
        policy: AutoregressiveTransformerStudentPolicy,
        cfg: Config,
        env: NaryTreeNavigationEnvironment,
) -> Tuple[List[float], List[float]]:
    """Mean and standard error of the RKL at each token position, over outputs sampled from the student.

    RKL is KL(student || teacher) between the next-token distributions given each prefix.
    """
    with torch.no_grad():
        rollouts = sample_student_rollouts(policy, cfg, env, cfg.eval_rollouts, keep_grad=False)

    means: List[float] = []
    stderrs: List[float] = []
    with torch.no_grad():
        for states, prefixes, prefix_lengths, active in zip(
                rollouts["states_by_step"],
                rollouts["prefixes_by_step"],
                rollouts["prefix_lengths_by_step"],
                rollouts["active_by_step"],
        ):
            active_rows = torch.nonzero(active > 0.5, as_tuple=False).squeeze(1)
            active_states = [states[int(row)] for row in active_rows.tolist()]
            active_lengths = prefix_lengths[active_rows]
            max_length = int(active_lengths.max().item()) if active_lengths.numel() > 0 else 0
            active_prefixes = prefixes[active_rows, :max_length]
            state_tensor = torch.tensor(active_states, dtype=torch.long)
            student_probs = policy.probs(active_prefixes, active_lengths)
            teacher_probs = teacher_distribution_torch(state_tensor, cfg, env)
            values = exact_divergence(student_probs, teacher_probs, "reverse_kl")
            means.append(float(values.mean().cpu()))
            if values.numel() > 1:
                stderr = values.std(unbiased=True) / math.sqrt(values.numel())
            else:
                stderr = torch.tensor(0.0, dtype=values.dtype)
            stderrs.append(float(stderr.cpu()))
    return means, stderrs


def train_one_objective(
        cfg: Config,
        env: NaryTreeNavigationEnvironment,
        objective: Objective,
) -> Tuple[History, Dict[int, Tuple[List[float], List[float]]]]:
    loss_fn = get_objective_loss(objective.name)
    set_seed(cfg.seed)
    policy = build_policy(cfg.policy_config, env)
    history: History = {"step": [], "exact_success_probability": []}
    for child in range(env.num_actions):
        history[f"root_child_{child}_probability"] = []
    profiles: Dict[int, Tuple[List[float], List[float]]] = {}
    profile_steps = set(cfg.eval_at_steps) | {0, cfg.training_steps}

    def record(step: int, stats: Dict[str, float]) -> None:
        history["step"].append(step)
        for key, value in stats.items():
            history[key].append(value)
        print(
            f"[{objective.name:16s}] step={step:4d} "
            f"success={stats['exact_success_probability']:.4f} "
            f"root_correct={stats['root_child_0_probability']:.4f}"
        )

    policy.eval()
    record(0, evaluate(policy, cfg, env))
    profiles[0] = reverse_kl_profile(policy, cfg, env)

    set_seed(cfg.seed + objective.seed_offset)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    for step in range(1, cfg.training_steps + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(policy, cfg, env)
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.training_steps:
            policy.eval()
            record(step, evaluate(policy, cfg, env))

        if step in profile_steps:
            policy.eval()
            profiles[step] = reverse_kl_profile(policy, cfg, env)

    return history, profiles


def run_config(cfg: Config) -> None:
    env = build_environment(cfg.env_config)
    out_dir = make_output_dir(cfg.output_root)
    write_config(cfg, out_dir)
    print(f"Saving outputs to: {out_dir}")
    histories: Dict[str, History] = {}
    profiles = {}
    for objective in cfg.objectives:
        print(f"\n=== Training {objective.name} ===")
        histories[objective.name], profiles[objective.name] = train_one_objective(cfg, env, objective)
    write_histories_csv(histories, out_dir)
    write_sampled_trajectory_divergence_profile_csv(profiles, cfg, out_dir)
    print(f"\nSaved metrics.csv and sampled_trajectory_divergence_profiles.csv in {out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML run configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Override the experiment directory prefix (output_root); a timestamp is appended.",
    )
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config)
    if args.output_dir is not None:
        cfg = replace(cfg, output_root=args.output_dir)

    run_config(cfg)


if __name__ == "__main__":
    main()
