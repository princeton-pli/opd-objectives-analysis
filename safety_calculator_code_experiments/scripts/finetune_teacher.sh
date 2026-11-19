#!/bin/bash
# The teacher is saved to $MODELS_DIR/qwen3_8b_base_code_completion.

source "$(dirname "$0")/common.sh"

SFT_DATASET="$CODE_COMPLETION_DIR/sft"

check_output_dir "$CODE_COMPLETION_TEACHER"
mkdir -p "$CODE_COMPLETION_DIR"

python opd/code_completion/build_sft_dataset.py --prompts "$CODE_COMPLETION_PROMPTS" --output_dir "$SFT_DATASET"

accelerate launch --config_file configs/accelerate/deepspeed_zero3_offload_optimizer.yaml --num_processes 1 \
    opd_train.py \
    --algorithm sft \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --dataset_name "$SFT_DATASET" \
    --dataset_train_split train \
    --output_dir "$CODE_COMPLETION_TEACHER" \
    ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
    --dtype bfloat16 \
    --bf16 true \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 512 \
    --packing false \
    --eval_strategy no \
    --use_vllm false \
    --log_rollout_metrics false \
    --save_strategy no \
    --save_final_model true \
    --logging_steps 1 \
    --report_to tensorboard \
    --seed 42 \
    --data_seed 42 \
    --optim adamw_torch
