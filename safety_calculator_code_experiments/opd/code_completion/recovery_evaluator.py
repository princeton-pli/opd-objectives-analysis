from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import gather_object

from opd.code_completion.boundary import Candidate, encode_stable_boundary
from utils import position_ids_from_attention_mask


def _screen_records(
    path: str,
    tokenizer,
    eval_datasets: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("kind") != "new_api_teacher_scores":
        raise ValueError(f"Not a teacher-scores file from build_screen.py: {path}")

    rows = {str(row["example_id"]): row for split in eval_datasets.values() for row in split}
    records = {}
    for candidate_id, scores in payload["candidates"].items():
        if candidate_id not in rows:
            continue
        candidate = Candidate.from_row(rows[candidate_id])
        prompt_ids, continuation_ids = encode_stable_boundary(tokenizer, candidate)
        if continuation_ids[0] != int(scores["token_id"]):
            raise ValueError(f"Teacher scores do not match the current tokenizer for {candidate_id!r}")
        records[candidate_id] = {
            "example_id": candidate_id,
            "api": candidate.api,
            "prompt_ids": prompt_ids,
            "continuation_ids": continuation_ids,
            "teacher_log_probability": scores["teacher_log_probability"],
        }
    if not records:
        raise ValueError(f"None of the prompts in {path} are in the evaluation data")
    return records


class NewAPIRecoveryEvaluator:
    """Score the first token of each API name during periodic evaluation."""

    def __init__(
        self,
        cohort: dict[str, dict[str, Any]],
        tokenizer,
    ) -> None:
        self.cohort = cohort
        self.tokenizer = tokenizer
        self.evaluated_ids: set[str] = set()

    @classmethod
    def from_path(
        cls,
        path: str | None,
        tokenizer,
        eval_datasets: dict[str, Any] | None = None,
    ) -> "NewAPIRecoveryEvaluator":
        cohort = {} if path is None else _screen_records(path, tokenizer, eval_datasets)
        return cls(cohort, tokenizer)

    def reset(self) -> None:
        """Clear the per-evaluation state."""
        self.evaluated_ids.clear()

    def record(self, inputs: dict[str, Any]) -> None:
        if not self.cohort:
            return
        for example_id in inputs.get("example_id", ()):
            key = str(example_id)
            if key in self.cohort:
                self.evaluated_ids.add(key)

    @torch.no_grad()
    def _score(
        self,
        records: list[dict[str, Any]],
        model: torch.nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> list[dict[str, Any]]:
        rows = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            sequences = [
                record["prompt_ids"] + record["continuation_ids"]
                for record in batch
            ]
            width = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(batch), width),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            offsets = []
            for row_index, sequence in enumerate(sequences):
                offset = width - len(sequence)
                offsets.append(offset)
                input_ids[row_index, offset:] = torch.as_tensor(
                    sequence, dtype=torch.long, device=device
                )
                attention_mask[row_index, offset:] = 1

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids_from_attention_mask(attention_mask),
            ).logits

            for row_index, (record, offset) in enumerate(
                zip(batch, offsets, strict=True)
            ):
                position = offset + len(record["prompt_ids"]) - 1
                log_probs = logits[row_index, position].float().log_softmax(-1)
                rows.append(
                    {
                        "candidate_id": record["example_id"],
                        "api": record["api"],
                        "student_log_probability": float(log_probs[record["continuation_ids"][0]]),
                        "teacher_log_probability": record.get("teacher_log_probability"),
                    }
                )
        return rows

    def finish(
        self,
        *,
        model: torch.nn.Module,
        accelerator,
        batch_size: int,
        step: int,
        objective: str,
        metric_key_prefix: str,
        output_dir: str,
        save: Callable[[dict[str, Any], Path], None],
    ) -> None:
        if not self.cohort:
            return
        gathered_ids = gather_object([sorted(self.evaluated_ids)])
        evaluated_ids = {
            example_id for process_ids in gathered_ids for example_id in process_ids
        }
        records = [
            record
            for example_id, record in self.cohort.items()
            if example_id in evaluated_ids
        ]
        if not records:
            return

        rows = self._score(records, model, batch_size, accelerator.device)
        if accelerator.is_main_process:
            safe_prefix = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in metric_key_prefix
            )
            save(
                {
                    "kind": "new_api_recovery_eval",
                    "step": step,
                    "objective": objective,
                    "rows": rows,
                },
                Path(output_dir)
                / "metrics"
                / "new_api_recovery"
                / f"{safe_prefix}_step_{step:08d}.pt",
            )
