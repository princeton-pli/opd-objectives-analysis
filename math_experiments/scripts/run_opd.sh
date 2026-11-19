#!/bin/bash
#   STUDENT=0.6B TEACHER=4B bash scripts/run_opd.sh                            # FKL and RKL, three seeds each
#   STUDENT=0.6B TEACHER=4B OBJECTIVES=rkl SEEDS=42 bash scripts/run_opd.sh    # one run
#
# Each run writes the trained student, its training log and its evaluation to $OUTPUT_ROOT/student_<STUDENT>_teacher_<TEACHER>/<objective>_seed_<seed>.

source "$(dirname "$0")/common.sh"

: "${STUDENT:?Set STUDENT to 0.6B or 1.7B}"
: "${TEACHER:?Set TEACHER to 4B, 8B, 14B or 32B}"
OBJECTIVES="${OBJECTIVES:-fkl rkl}"
SEEDS="${SEEDS:-42 43 44}"
STUDENT_MODEL=$(student_model "$STUDENT")
TEACHER_MODEL=$(teacher_model "$TEACHER")

run_one() {
    local objective=$1 seed=$2 alpha
    case "$objective" in
        fkl) alpha=0 ;;
        rkl) alpha=1 ;;
        *) echo "Unknown objective '$objective': use fkl or rkl" >&2; exit 2 ;;
    esac

    local output_dir="$OUTPUT_ROOT/student_${STUDENT}_teacher_${TEACHER}/${objective}_seed_${seed}"
    check_output_dir "$output_dir"

    local args=(
        # models and data; the teacher receives the same prompt as the student
        --model_name "$STUDENT_MODEL"
        --teacher_model_name "$TEACHER_MODEL"
        --train_file_path datasets/math_train.parquet
        --enable_thinking false
        --max_prompt_length 2048

        --alpha "$alpha"
        --teacher_eos_remap 151643:151645

        --seed "$seed"
        --max_steps 400
        --per_device_train_batch_size 2
        --gradient_accumulation_steps 8
        --num_generations 1
        --learning_rate 1e-6
        --lr_scheduler_type constant
        --warmup_ratio 0.0
        --weight_decay 0.01
        --max_grad_norm 1.0
        --bf16 true
        --gradient_checkpointing true
        --fsdp "full_shard auto_wrap"
        --fsdp_config configs/fsdp_config_qwen3_opsd.json

        --temperature 1.0
        --top_p 1.0
        --max_completion_length 4096

        --logging_steps 1
        --report_to tensorboard
        --save_strategy no
        --output_dir "$output_dir"
    )

    echo "[$(date '+%F %T')] student $STUDENT_MODEL | teacher $TEACHER_MODEL | $objective | seed $seed | $output_dir"
    python -m torch.distributed.run --nproc_per_node=4 --standalone run_opsd_trl.py "${args[@]}" 2>&1 \
        | tee "$output_dir/train.log"

    evaluate "$output_dir" math500_inst "$output_dir/eval" 1
}

for seed in $SEEDS; do
    for objective in $OBJECTIVES; do
        run_one "$objective" "$seed"
    done
done
