"""
On-Policy Self-Distillation (OPSD) using TRL.

This module implements Self-Distillation Fine-Tuning (SDFT) based on
"Self-Distillation Enables Continual Learning" (https://arxiv.org/abs/2601.19897)

Reference implementation: https://github.com/idanshen/Self-Distillation

DistilTrainer: Full-featured, extends transformers.Trainer (recommended)
"""

from .distil_trainer import DistilTrainer
from .distil_config import DistilConfig

__all__ = [
    "DistilTrainer",
    "DistilConfig",
]
