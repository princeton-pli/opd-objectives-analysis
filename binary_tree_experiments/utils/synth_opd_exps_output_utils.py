from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

History = Dict[str, List[float]]
Profiles = Dict[str, Dict[int, Tuple[List[float], List[float]]]]


def make_output_dir(output_root: Path) -> Path:
    """Create the experiment directory as <output_root>_<timestamp>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root.with_name(f"{output_root.name}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


def write_config(cfg: Any, out_dir: Path) -> None:
    config_data = asdict(cfg)
    serialized = yaml.safe_dump(_yaml_safe(config_data), sort_keys=False)
    (out_dir / "config.yaml").write_text(serialized, encoding="utf-8")


def write_histories_csv(histories: Dict[str, History], out_dir: Path) -> None:
    """Save the exact success probability and the probability of each token at the root."""
    metrics = [key for key in next(iter(histories.values())) if key != "step"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["objective_name", "step", *metrics])
        for objective_name, history in histories.items():
            for idx, step in enumerate(history["step"]):
                writer.writerow([objective_name, int(step), *[history[metric][idx] for metric in metrics]])


def write_sampled_trajectory_divergence_profile_csv(profiles: Profiles, cfg: Any, out_dir: Path) -> None:
    """Save the mean and standard error of the RKL at each token position (trajectory_step, from 0).

    Means and standard errors are over the eval_rollouts outputs sampled from the student; RKL is in nats.
    """
    with (out_dir / "sampled_trajectory_divergence_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "objective_name",
            "eval_step",
            "trajectory_step",
            "divergence_metric",
            "mean_divergence",
            "stderr",
            "eval_rollouts",
        ])
        for objective_name, by_step in profiles.items():
            for evaluation_step in sorted(by_step):
                values, stderr = by_step[evaluation_step]
                for trajectory_step, (mean_divergence, standard_error) in enumerate(zip(values, stderr)):
                    writer.writerow([
                        objective_name,
                        evaluation_step,
                        trajectory_step,
                        "reverse_kl",
                        mean_divergence,
                        standard_error,
                        cfg.eval_rollouts,
                    ])
