#!/bin/bash
#   EXPERIMENT=safety_qwen3 bash scripts/run_initial_eval.sh
# Each split writes to $OUTPUT_ROOT/$EXPERIMENT/initial_eval_<split>.

source "$(dirname "$0")/common.sh"
load_experiment

for split in $EVAL_SPLITS; do
    output_dir="$RUN_ROOT/initial_eval_$split"
    check_output_dir "$output_dir"

    args=(
        # model
        --model_name_or_path "$STUDENT_MODEL"
        --teacher_model_name_or_path "$TEACHER_MODEL"
        --dtype bfloat16
        --attn_implementation flash_attention_2

        --dataset_name "$DATASET"
        --eval_split "$split"
        --data_seed 42
        --max_prompt_tokens "$MAX_PROMPT_TOKENS"
        --max_response_tokens "$MAX_RESPONSE_TOKENS"
        --n_eval_samples 500
        --eval_seed 42
        --seed 42
        --eval_batch_size 4

        --max_length "$MAX_LENGTH"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --temperature 1.0
        --top_p 1.0
        --top_k 0
        --enable_thinking "$ENABLE_THINKING"
        --use_vllm true
        --vllm_gpu_memory_utilization 0.9

        --num_completions_to_print 10
        --print_seed 0
        --save_tokenized_generations true
        --output_dir "$output_dir"
    )
    if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then args+=(--max_train_samples "$MAX_TRAIN_SAMPLES"); fi
    if [[ -n "$CACHE_DIR" ]]; then args+=(--cache_dir "$CACHE_DIR"); fi

    echo "[$(date '+%F %T')] $EXPERIMENT | initial evaluation on $split | student $STUDENT_MODEL | teacher $TEACHER_MODEL | $output_dir"
    python eval_pair.py "${args[@]}" 2>&1 | tee "$output_dir/eval.log"
done
