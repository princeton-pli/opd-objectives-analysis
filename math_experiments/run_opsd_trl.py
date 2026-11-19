#!/usr/bin/env python3
"""
On-Policy Self-Distillation (TRL-based) Training Script.

Based on "Self-Distillation Enables Continual Learning" - https://arxiv.org/abs/2601.19897
Original implementation: https://github.com/idanshen/Self-Distillation
"""

import os
import sys

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

import argparse
import logging
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

from opsd_trl.distil_trainer import DistilTrainer
from opsd_trl.distil_config import DistilConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_bool(v):
    """Parse boolean string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def parse_args():
    parser = argparse.ArgumentParser(description="On-Policy Self-Distillation Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--teacher_model_name", type=str, required=True, help="Teacher model name or path")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name (defaults to output_dir basename)")
    
    # Data arguments
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to training dataset")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0, total number of optimizer steps to run (overrides num_train_epochs)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Distillation arguments
    parser.add_argument("--alpha", type=float, default=0.0, choices=[0.0, 1.0], help="KL type: 0=forward, 1=reverse")
    parser.add_argument("--teacher_eos_remap", type=str, default=None,
                        help="'SRC:DST' token ids. Show the teacher DST wherever the student's completion "
                             "has its first SRC, and read the teacher's DST column in the SRC slot. Use "
                             "151643:151645 for a Base student with a post-trained Qwen3 teacher.")
    parser.add_argument("--enable_thinking", type=str, default="default",
                        choices=["true", "false", "default"],
                        help="Control Qwen3 thinking mode in chat template during training. "
                             "'true'=enable, 'false'=disable, 'default'=don't pass (tokenizer default)")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of student rollouts per prompt (K) to average during training")

    # Generation arguments
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    
    # vLLM arguments
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3, help="vLLM GPU memory utilization")
    parser.add_argument("--vllm_enable_sleep_mode", type=parse_bool, default=True, help="Enable vLLM sleep mode")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--save_strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report to (wandb, none)")
    
    # Precision and optimization
    parser.add_argument("--bf16", type=parse_bool, default=True, help="Use bf16")
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True, help="Use gradient checkpointing")
    
    # FSDP arguments (default to FSDP with qwen config)
    parser.add_argument("--fsdp", type=str, default="full_shard auto_wrap", help="FSDP configuration string")
    parser.add_argument("--fsdp_config", type=str, default="configs/fsdp_config_qwen.json", help="Path to FSDP config file")
    
    return parser.parse_args()


def prepare_distil_dataset(
    dataset: Dataset,
    seed: int = 42,
) -> Dataset:
    """
    Prepare dataset for distillation training.

    Creates the 'prompt' (student input) column. The teacher receives the same prompt.
    """

    def format_example(example):
        prompt = example.get("prompt", "")

        result = {
            "prompt": [{"role": "user", "content": prompt}],
        }

        return result

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def load_training_dataset(args):
    """Load the training dataset (parquet file)."""
    logging.info(f"Loading dataset from {args.train_file_path}")
    raw_dataset = load_dataset("parquet", data_files=args.train_file_path, split="train")
    return raw_dataset


if __name__ == "__main__":
    args = parse_args()
    
    if args.run_name is None:
        args.run_name = os.path.basename(args.output_dir)
    
    logging.info(f"Loading student model from {args.model_name}")
    logging.info(f"Loading teacher model from {args.teacher_model_name}")

    model_kwargs = {
        "dtype": torch.bfloat16,
        "use_cache": False,
    }

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_name, **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    raw_dataset = load_training_dataset(args)
    train_dataset = prepare_distil_dataset(
        raw_dataset,
        seed=args.seed,
    )

    logging.info(f"Dataset prepared with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logging.info(f"Sample student prompt: {str(sample['prompt'])[:200]}...")
    
    config_kwargs = {
        "seed": args.seed,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        
        # Training hyperparameters
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        
        # Distillation settings
        "alpha": args.alpha,
        "teacher_eos_remap": args.teacher_eos_remap,
        "enable_thinking": args.enable_thinking,
        
        # Generation settings
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,

        # vLLM settings
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
        
        # Logging and saving
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "report_to": args.report_to,
        
        # Precision
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
    if args.fsdp:
        config_kwargs["fsdp"] = args.fsdp
    if args.fsdp_config:
        config_kwargs["fsdp_config"] = args.fsdp_config
    
    config = DistilConfig(**config_kwargs)
    
    logging.info("=" * 60)
    logging.info("DistilTrainer Settings:")
    logging.info(f"  Student Model: {args.model_name}")
    logging.info(f"  Teacher Model: {args.teacher_model_name}")
    logging.info(f"  Num Student Rollouts (K): {args.num_generations}")
    logging.info(f"  Output: {args.output_dir}")
    logging.info(f"  Alpha (KL type): {args.alpha} (0=forward, 1=reverse)")
    logging.info(f"  Max prompt length: {args.max_prompt_length}")
    logging.info(f"  Max completion length: {args.max_completion_length}")
    logging.info(f"  Batch size: {args.per_device_train_batch_size}")
    logging.info(f"  Gradient accum: {args.gradient_accumulation_steps}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info("=" * 60)
    
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    logging.info("Starting on-policy self-distillation training...")
    trainer.train()
    
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(output_dir=args.output_dir)
    
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        logging.info(f"Model and tokenizer saved to {args.output_dir}")
    
    trainer.accelerator.wait_for_everyone()
    _rank = trainer.accelerator.process_index
    logging.info(f"Training complete on rank {_rank}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

    logging.info(f"Exiting rank {_rank} with status 0 (teardown skipped)")
    logging.shutdown()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
