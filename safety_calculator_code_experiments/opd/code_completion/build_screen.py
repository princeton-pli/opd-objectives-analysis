from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opd.code_completion.boundary import (
    Candidate,
    encode_stable_boundary,
    shared_tokenizer_fingerprint,
)
from opd.code_completion.recovery_evaluator import NewAPIRecoveryEvaluator
from utils.tokenizer_utils import ensure_distinct_pad_token


def build_records(student_tokenizer, teacher_tokenizer, candidates, shared_vocab_size):
    """Encode each boundary under both tokenizers, reject any disagreement, and return the records to score."""
    if shared_tokenizer_fingerprint(
        student_tokenizer, shared_vocab_size
    ) != shared_tokenizer_fingerprint(teacher_tokenizer, shared_vocab_size):
        raise ValueError(
            f"The first {shared_vocab_size} teacher and student token IDs do not match."
        )
    records = []
    for candidate in candidates:
        prompt_ids, continuation_ids = encode_stable_boundary(student_tokenizer, candidate)
        _, teacher_continuation_ids = encode_stable_boundary(teacher_tokenizer, candidate)
        if continuation_ids != teacher_continuation_ids:
            raise ValueError(f"Continuation token IDs differ for {candidate.candidate_id!r}")
        records.append(
            {
                "example_id": candidate.candidate_id,
                "api": candidate.api,
                "prompt_ids": prompt_ids,
                "continuation_ids": continuation_ids,
            }
        )
    return records


@torch.no_grad()
def score_teacher(args: argparse.Namespace, records: list[dict], tokenizer) -> list[dict]:
    """Score with NewAPIRecoveryEvaluator._score, the code the runs' evaluations use."""
    accelerator = Accelerator(mixed_precision="bf16")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    student_config = AutoConfig.from_pretrained(args.student, cache_dir=args.cache_dir)
    model.resize_token_embeddings(student_config.get_text_config().vocab_size)
    model = accelerator.prepare_model(model, evaluation_mode=True).eval()
    return NewAPIRecoveryEvaluator({}, tokenizer)._score(
        records, model, args.eval_batch_size, accelerator.device
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--shared_vocab_size", type=int, default=151665)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [Candidate.from_row(row) for row in load_from_disk(args.dataset)[args.dataset_split]]
    if not candidates:
        raise SystemExit(f"No rows in {args.dataset} [{args.dataset_split}]")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student, cache_dir=args.cache_dir)
    ensure_distinct_pad_token(student_tokenizer)  # the trainer's processing_class
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, cache_dir=args.cache_dir)
    records = build_records(student_tokenizer, teacher_tokenizer, candidates, args.shared_vocab_size)

    scores = score_teacher(args, records, student_tokenizer)
    payload = {
        "kind": "new_api_teacher_scores",
        "teacher": args.teacher,
        "candidates": {
            record["example_id"]: {
                "token_id": record["continuation_ids"][0],
                "teacher_log_probability": row["student_log_probability"],
            }
            for record, row in zip(records, scores, strict=True)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(candidates)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
