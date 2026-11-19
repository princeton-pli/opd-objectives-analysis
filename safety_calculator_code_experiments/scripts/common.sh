#!/bin/bash
# Paths, environment and per-experiment settings shared by the scripts in this folder.
#   DATA_DIR=/data/opd EXPERIMENT=safety_qwen3 OBJECTIVES=tok_rkl SEEDS=42 bash scripts/run_opd.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-datasets}"       # built datasets
MODELS_DIR="${MODELS_DIR:-models}"     # the fine-tuned code completion teacher
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"  # runs and evaluations
CACHE_DIR="${CACHE_DIR:-}"             # Hugging Face cache; empty uses the default one

# ── Environment ────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

# ── Code completion task ───────────────────────────────────────────────────
CODE_COMPLETION_DIR="$DATA_DIR/code_completion"
CODE_COMPLETION_PROMPTS="data/code_completion/prompts.jsonl"
CODE_COMPLETION_TEACHER="$MODELS_DIR/qwen3_8b_base_code_completion"

# ── Experiments of the paper ───────────────────────────────────────────────
load_experiment() {
    : "${EXPERIMENT:?Set EXPERIMENT to one of: safety_qwen3, safety_qwen3_thinking, safety_gemma3, calculator_qwen3, calculator_gemma3, code_completion}"

    ENABLE_THINKING=False MAX_NEW_TOKENS=1024 MAX_LENGTH=1552
    LEARNING_RATE=3e-6 NUM_TRAIN_EPOCHS=4 GRADIENT_ACCUMULATION_STEPS=32
    MAX_TRAIN_SAMPLES=6000 MAX_TRAIN_EVAL_SAMPLES=500 MAX_TEST_EVAL_SAMPLES=500
    EVAL_STEPS=94 EVALS_PER_EPOCH=8 WARMUP_RATIO=0 EVAL_SPLITS=train
    PER_DEVICE_EVAL_BATCH_SIZE= VLLM_GPU_MEM_UTIL= TEACHER_DEVICE= RECOVERY_SCREEN=
    local objectives="seq_rkl tok_rkl tok_fkl tok_js"

    case "$EXPERIMENT" in
        safety_qwen3)
            STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-8B PER_DEVICE_EVAL_BATCH_SIZE=2
            DATASET="$DATA_DIR/wildjailbreak_v4" MAX_PROMPT_TOKENS=439
            ;;
        safety_qwen3_thinking)
            STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-8B PER_DEVICE_EVAL_BATCH_SIZE=2
            DATASET="$DATA_DIR/wildjailbreak_v4" MAX_PROMPT_TOKENS=439
            ENABLE_THINKING=True MAX_NEW_TOKENS=2048 MAX_LENGTH=2576
            ;;
        safety_gemma3)
            STUDENT_MODEL=google/gemma-3-4b-it TEACHER_MODEL=google/gemma-3-27b-it
            TEACHER_DEVICE=cuda:1 VLLM_GPU_MEM_UTIL=0.15
            DATASET="$DATA_DIR/wildjailbreak_v4_gemma" MAX_PROMPT_TOKENS=448
            ;;
        calculator_qwen3)
            STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-8B PER_DEVICE_EVAL_BATCH_SIZE=2
            DATASET="$DATA_DIR/deepmind_arithmetic_calculator_direct" MAX_PROMPT_TOKENS=512
            EVAL_SPLITS="train validation"
            ;;
        calculator_gemma3)
            STUDENT_MODEL=google/gemma-3-4b-it TEACHER_MODEL=google/gemma-3-27b-it
            TEACHER_DEVICE=cuda:1 VLLM_GPU_MEM_UTIL=0.15
            DATASET="$DATA_DIR/deepmind_arithmetic_calculator_direct_conditional" MAX_PROMPT_TOKENS=512
            EVAL_SPLITS="train validation"
            ;;
        code_completion)
            STUDENT_MODEL=Qwen/Qwen2.5-1.5B TEACHER_MODEL="$CODE_COMPLETION_TEACHER"
            DATASET="$CODE_COMPLETION_DIR/cohort" RECOVERY_SCREEN="$CODE_COMPLETION_DIR/validation_screen.json"
            objectives="tok_fkl tok_rkl tok_js tok_skl"
            LEARNING_RATE=1e-5 NUM_TRAIN_EPOCHS=30 GRADIENT_ACCUMULATION_STEPS=8 PER_DEVICE_EVAL_BATCH_SIZE=8
            MAX_NEW_TOKENS=128 MAX_PROMPT_TOKENS=512 MAX_LENGTH=640
            MAX_TRAIN_SAMPLES= MAX_TRAIN_EVAL_SAMPLES=0 MAX_TEST_EVAL_SAMPLES=1024
            EVAL_STEPS=15 EVALS_PER_EPOCH= WARMUP_RATIO= VLLM_GPU_MEM_UTIL=0.4
            ;;
        *)
            echo "Unknown EXPERIMENT '$EXPERIMENT'" >&2
            exit 2
            ;;
    esac
    MAX_RESPONSE_TOKENS="$MAX_NEW_TOKENS"

    OBJECTIVES="${OBJECTIVES:-$objectives}" SEEDS="${SEEDS:-42 43 44}"
    RUN_ROOT="$OUTPUT_ROOT/$EXPERIMENT"
}

check_output_dir() {
    if [[ -d "$1" && -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_EXISTING_OUTPUT:-false}" != "true" ]]; then
        echo "Refusing to overwrite non-empty output directory: $1 (set ALLOW_EXISTING_OUTPUT=true to overwrite)" >&2
        exit 2
    fi
    mkdir -p "$1"
}
