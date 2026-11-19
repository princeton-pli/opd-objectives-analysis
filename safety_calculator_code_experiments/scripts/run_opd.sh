#!/bin/bash
#   EXPERIMENT=safety_qwen3 bash scripts/run_opd.sh                                 # all objectives and seeds
#   EXPERIMENT=safety_qwen3 OBJECTIVES=tok_rkl SEEDS=42 bash scripts/run_opd.sh     # one run
#
# EXPERIMENT       safety_qwen3 | safety_qwen3_thinking | safety_gemma3 | calculator_qwen3 | calculator_gemma3 | code_completion
# OBJECTIVES       any of: seq_rkl tok_rkl tok_fkl tok_js tok_skl (default: those of the experiment in the paper)
# SEEDS            training seeds (default: 42 43 44)
# The models, dataset and settings of each experiment are in scripts/common.sh.
# Each run writes to $OUTPUT_ROOT/$EXPERIMENT/<objective>_seed_<seed>.

source "$(dirname "$0")/common.sh"
load_experiment

run_one() {
    local objective=$1 seed=$2
    case "$objective" in
        seq_rkl|tok_rkl|tok_fkl|tok_js|tok_skl) ;;
        *) echo "Unknown objective '$objective': use seq_rkl, tok_rkl, tok_fkl, tok_js or tok_skl" >&2; exit 2 ;;
    esac

    local data_seed=42
    if [[ "$EXPERIMENT" == code_completion ]]; then data_seed="$seed"; fi

    local vllm_util="$VLLM_GPU_MEM_UTIL" eval_batch_size="$PER_DEVICE_EVAL_BATCH_SIZE"
    if [[ -z "$vllm_util" ]]; then
        if [[ "$objective" == seq_rkl ]]; then vllm_util=0.25; else vllm_util=0.3; fi
    fi
    if [[ -z "$eval_batch_size" ]]; then
        if [[ "$objective" == tok_js ]]; then eval_batch_size=2; else eval_batch_size=4; fi
    fi

    local output_dir="$RUN_ROOT/${objective}_seed_${seed}"
    check_output_dir "$output_dir"

    local args=(
        # model
        --model_name_or_path "$STUDENT_MODEL"
        --teacher_model_name_or_path "$TEACHER_MODEL"
        --dtype bfloat16
        --attn_implementation flash_attention_2

        # dataset
        --dataset_name "$DATASET"
        --dataset_test_split validation
        --max_train_eval_samples "$MAX_TRAIN_EVAL_SAMPLES"
        --max_test_eval_samples "$MAX_TEST_EVAL_SAMPLES"
        --max_prompt_tokens "$MAX_PROMPT_TOKENS"
        --max_response_tokens "$MAX_RESPONSE_TOKENS"

        # seeding
        --seed "$seed"
        --data_seed "$data_seed"

        # OPD
        --opd_loss_type "$objective"
        --lmbda 1.0

        # training
        --num_train_epochs "$NUM_TRAIN_EPOCHS"
        --per_device_train_batch_size 2
        --per_device_eval_batch_size "$eval_batch_size"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --learning_rate "$LEARNING_RATE"
        --max_grad_norm 1
        --lr_scheduler_type constant
        --loss_reduction trajectory_mean
        --max_length "$MAX_LENGTH"
        --gradient_checkpointing
        --bf16 true

        # generation
        --temperature 1.0
        --top_p 1.0
        --top_k 0
        --max_completion_length "$MAX_NEW_TOKENS"
        --enable_thinking "$ENABLE_THINKING"

        # student rollouts: vLLM engine colocated on the student's GPU
        --use_vllm true
        --vllm_mode colocate
        --vllm_gpu_memory_utilization "$vllm_util"
        --vllm_enable_sleep_mode true

        # logging and evaluation
        --logging_steps 1
        --report_to tensorboard
        --log_rollout_metrics true
        --save_eval_rollout_records true
        --log_completions True
        --log_completions_steps 25
        --num_completions_to_print 5
        --eval_strategy steps
        --eval_steps "$EVAL_STEPS"
        --eval_on_start true

        # output
        --output_dir "$output_dir"
        --save_strategy no
    )
    if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then args+=(--max_train_samples "$MAX_TRAIN_SAMPLES"); fi
    if [[ -n "$EVALS_PER_EPOCH" ]]; then args+=(--evals_per_epoch "$EVALS_PER_EPOCH"); fi
    if [[ -n "$WARMUP_RATIO" ]]; then args+=(--warmup_ratio "$WARMUP_RATIO"); fi
    if [[ "$objective" == tok_js ]]; then args+=(--beta 0.5); fi  # weight of the generalized Jensen-Shannon divergence
    if [[ -n "$TEACHER_DEVICE" ]]; then args+=(--teacher_device "$TEACHER_DEVICE"); fi
    if [[ -n "$RECOVERY_SCREEN" ]]; then
        args+=(--new_api_recovery_screen_path "$RECOVERY_SCREEN")
    fi
    if [[ -n "$CACHE_DIR" ]]; then args+=(--cache_dir "$CACHE_DIR"); fi

    echo "[$(date '+%F %T')] $EXPERIMENT | $objective | seed $seed | student $STUDENT_MODEL | teacher $TEACHER_MODEL | $output_dir"
    python opd_train.py "${args[@]}" 2>&1 | tee "$output_dir/train.log"
}

for seed in $SEEDS; do
    for objective in $OBJECTIVES; do
        run_one "$objective" "$seed"
    done
done
