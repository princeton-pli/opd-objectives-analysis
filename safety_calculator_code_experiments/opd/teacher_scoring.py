from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils import position_ids_from_attention_mask

from .opd_losses import logprobs_at_ids


def check_teacher_eos(
    student_name: str,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> None:
    """Refuse a teacher whose end-of-sequence token differs from the student's."""
    student_eos = student_tokenizer.eos_token_id
    teacher_eos = teacher_tokenizer.eos_token_id
    if student_eos != teacher_eos:
        raise ValueError(
            f"Student {student_name} stops on {student_tokenizer.eos_token!r} (id {student_eos}) "
            f"but the teacher stops on {teacher_tokenizer.eos_token!r} (id {teacher_eos})."
        )


@dataclass
class TeacherScores:
    realized_logprobs: torch.Tensor
    full_logits: torch.Tensor | None = None


class LocalHFTeacherScorer:
    """Score student trajectories with the in-process teacher (exact full-vocabulary logits)."""

    def __init__(
        self,
        teacher_model: torch.nn.Module,
    ):
        self.teacher_model = teacher_model

    def score(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: int,
        prompt_attention_mask: torch.Tensor,
        temperature: float,
        need_full_distribution: bool = False,
    ) -> TeacherScores:
        del prompt_attention_mask

        self.teacher_model.eval()
        teacher_device = getattr(self.teacher_model, "device", input_ids.device)
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids.to(teacher_device),
                attention_mask=attention_mask.to(teacher_device),
                position_ids=position_ids_from_attention_mask(attention_mask).to(teacher_device),
            )
            teacher_logits = outputs.logits[:, prompt_lengths - 1 : -1, :].to(input_ids.device)
            realized_logprobs = logprobs_at_ids(
                teacher_logits,
                input_ids[:, prompt_lengths:].unsqueeze(-1),
                temperature,
            ).squeeze(-1)

        return TeacherScores(
            realized_logprobs=realized_logprobs,
            full_logits=teacher_logits if need_full_distribution else None,
        )
