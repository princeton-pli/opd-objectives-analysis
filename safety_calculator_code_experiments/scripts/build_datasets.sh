#!/bin/bash
# Builds the safety and calculator datasets under $DATA_DIR (default: datasets). CPU only.
# The code completion dataset ships with the repository (data/code_completion/prompts.jsonl).
#
#   bash scripts/build_datasets.sh                     # both tasks
#   DATASETS=calculator bash scripts/build_datasets.sh # one task: safety or calculator


source "$(dirname "$0")/common.sh"

DATASETS="${DATASETS:-safety calculator}"
mkdir -p "$DATA_DIR"

build_safety() {
    # The two datasets hold the same requests; only the prompt template around them differs.
    local pair template name
    for pair in "qwen3:wildjailbreak_v4" "gemma3:wildjailbreak_v4_gemma"; do
        template="${pair%%:*}"
        name="${pair#*:}"
        python -m utils.data.wildjailbreak_abstain \
            --template "$template" \
            --validation_examples_per_type 250 \
            --train_examples_per_type 1500 \
            --seed 42 \
            --output_dir "$DATA_DIR/$name" \
            ${CACHE_DIR:+--cache_dir "$CACHE_DIR"}
    done
}

build_calculator() {
    # The two datasets hold the same problems; only the routing sentence of the system prompt differs.
    local pair preference name
    for pair in "direct:deepmind_arithmetic_calculator_direct" "direct_conditional:deepmind_arithmetic_calculator_direct_conditional"; do
        preference="${pair%%:*}"
        name="${pair#*:}"
        python -m utils.data.deepmind_calculator \
            --route_preference "$preference" \
            --seed 42 \
            --train_examples_per_module 1000 \
            --validation_examples_per_module 1000 \
            --output_dir "$DATA_DIR/$name"
    done
}

for dataset in $DATASETS; do
    case "$dataset" in
        safety) build_safety ;;
        calculator) build_calculator ;;
        *) echo "Unknown dataset group '$dataset': use safety or calculator" >&2; exit 2 ;;
    esac
done
