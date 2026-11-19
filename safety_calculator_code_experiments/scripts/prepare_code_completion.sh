#!/bin/bash
# Everything is written to $DATA_DIR/code_completion.

source "$(dirname "$0")/common.sh"

DATASET="$CODE_COMPLETION_DIR/cohort"
RECOVERY_SCREEN="$CODE_COMPLETION_DIR/validation_screen.json"
TEACHER_ROLLOUTS="$CODE_COMPLETION_DIR/teacher_rollouts.json"
mkdir -p "$CODE_COMPLETION_DIR"

python opd/code_completion/build_cohort.py \
    --prompts "$CODE_COMPLETION_PROMPTS" \
    --output_dir "$DATASET"

python -m opd.code_completion.build_screen \
    --student Qwen/Qwen2.5-1.5B \
    --teacher "$CODE_COMPLETION_TEACHER" \
    --dataset "$DATASET" \
    --dataset_split validation \
    --output "$RECOVERY_SCREEN" \
    ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
    --shared_vocab_size 151665 \
    --attn_implementation flash_attention_2 \
    --eval_batch_size 8

python opd/code_completion/teacher_rollout_hits.py \
    --prompts "$CODE_COMPLETION_PROMPTS" \
    --model "$CODE_COMPLETION_TEACHER" \
    --samples 1 \
    --max_tokens 128 \
    --output "$TEACHER_ROLLOUTS"
