#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

# ── Environment ────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false

# ── Models of the paper ────────────────────────────────────────────────────
student_model() {
    case "$1" in
        0.6B|1.7B) echo "Qwen/Qwen3-$1-Base" ;;
        *) echo "Unknown student '$1': use 0.6B or 1.7B" >&2; return 2 ;;
    esac
}
teacher_model() {
    case "$1" in
        4B|8B|14B|32B) echo "Qwen/Qwen3-$1" ;;
        *) echo "Unknown teacher '$1': use 4B, 8B, 14B or 32B" >&2; return 2 ;;
    esac
}

teacher_gpus() { if [[ "$1" == 32B ]]; then echo 2; else echo 1; fi; }

check_output_dir() {
    if [[ -d "$1" && -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_EXISTING_OUTPUT:-false}" != "true" ]]; then
        echo "Refusing to overwrite non-empty output directory: $1 (set ALLOW_EXISTING_OUTPUT=true to overwrite)" >&2
        exit 2
    fi
    mkdir -p "$1"
}

evaluate() {
    local model=$1 dataset=$2 output_dir=$3 num_gpus=$4
    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] evaluation | $model | $dataset | $output_dir"
    python run_generate.py --model_name "$model" --dataset "$dataset" --output_dir "$output_dir/$dataset" \
        --max_tokens 8192 --temperature 1.0 --top_p 0.8 --num_generation 8 \
        --enable_thinking false --batch_size 64 --num_gpus "$num_gpus"
    python run_eval.py --file_path "$output_dir/$dataset/$dataset-temp_1.0-top_p_0.8.jsonl" --regrade \
        | tail -8 | tee "$output_dir/$dataset.summary.txt"
}
