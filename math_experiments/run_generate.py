import os
import json
import argparse
import glob
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from tasks import get_task

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--enable_thinking", type=str, default="default",
                        help="Enable thinking mode in chat template: 'true', 'false', or 'default' (model decides)")

    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--merge_only", action="store_true")
    
    return parser.parse_args()

def run_merge(args, final_filename):
    """Merges all shard files into the final JSONL."""
    search_pattern = f"{final_filename}.shard_*"
    shard_files = [f for f in glob.glob(search_pattern) if not f.endswith(".partial")]
    
    if not shard_files:
        print("No completed shards found to merge.")
        return

    print(f"Supervisor: Merging {len(shard_files)} files...")
    all_data = []
    for s_file in shard_files:
        with open(s_file, 'r') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    all_data.sort(key=lambda x: x.get("question_id", 0))

    print(f"Writing {len(all_data)} records to {final_filename}...")
    with open(final_filename, 'w') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
    
    print("Merge complete.")

def main():
    args = parse_args()
    print("Arguments:", args, flush=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_short = args.dataset #.split(",")[-1].split("/")[-1]
    base_filename = f"{dataset_short}-temp_{args.temperature}-top_p_{args.top_p}.jsonl"
    final_output_path = os.path.join(args.output_dir, base_filename)

    if args.merge_only:
        run_merge(args, final_output_path)
        return

    shard_file_path = f"{final_output_path}.shard_{args.shard_id}"
    temp_shard_path = f"{shard_file_path}.partial"

    if os.path.exists(shard_file_path):
        print(f"Shard {args.shard_id}: Final file exists. Skipping inference.")
    else:
        print(f"Shard {args.shard_id}: Loading datasets...")
        d_name = args.dataset
        all_items = []
        global_counter = 0

        print(f"  - Loading {d_name}...")
        
        task_handler = get_task(d_name)
        ds = task_handler.load_data()
        if not ds:
            print(f"    WARNING: No data found for {d_name}. Skipping.")
            # return
        print(f"    Loaded {len(ds)} items.")
        
        for item in ds:
            all_items.append({
                "dataset": d_name,
                "raw_item": item,
                "global_id": global_counter,
                "task_handler": task_handler
            })
            global_counter += 1
            
        existing_ids = set()
        if os.path.exists(temp_shard_path):
            print(f"Shard {args.shard_id}: Resuming from {temp_shard_path}...")
            with open(temp_shard_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        existing_ids.add(data["question_id"])
                    except: continue

        items_to_process = []
        for i, item in enumerate(all_items):
            if i % args.num_shards == args.shard_id:
                if item["global_id"] not in existing_ids:
                    items_to_process.append(item)

        if items_to_process:
            print(f"Shard {args.shard_id}: Processing {len(items_to_process)} items...")
            
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.num_gpus,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
                trust_remote_code=True
            )
            
            sampling_params = SamplingParams(
                n=args.num_generation,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                skip_special_tokens=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

            formatted_prompts = []
            for item in items_to_process:
                handler = item["task_handler"]
                prompt_data = handler.format_prompt(item["raw_item"])
                if isinstance(prompt_data, str):
                    text = prompt_data
                else:
                    chat_template_kwargs = {}
                    if args.enable_thinking != "default":
                        chat_template_kwargs["enable_thinking"] = (args.enable_thinking == "true")
                    text = tokenizer.apply_chat_template(
                        prompt_data,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs
                    )
                    #formatted_prompts.append(text)

                formatted_prompts.append(text)
                
            with open(temp_shard_path, 'a') as f_out:
                for i in tqdm(range(0, len(items_to_process), args.batch_size)):
                    batch_prompts = formatted_prompts[i : i + args.batch_size]
                    batch_items   = items_to_process[i : i + args.batch_size]
                    
                    outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
                    
                    for j, output in enumerate(outputs):
                        item = batch_items[j]
                        handler = item["task_handler"]
                        
                        for k, output_obj in enumerate(output.outputs):
                            resp_text = output_obj.text
                            
                            # --- DELEGATED GRADING ---
                            gold = handler.get_gold(item["raw_item"])
                            score_dict = handler.grade(resp_text, gold, item["raw_item"])

                            # --- OUTPUT SCHEMA ---
                            result_entry = item["raw_item"].copy()
                            if "datapoint" in result_entry: del result_entry["datapoint"]
                            
                            result_entry.update({
                                "question_id": item["global_id"],
                                "generation_id": k,
                                "dataset": item["dataset"],
                                "prompt": batch_prompts[j],
                                "response": resp_text,
                                "gold_answer": str(gold),
                                #"is_correct": is_correct,
                                "model_name": args.model_name,
                                **score_dict
                            })
                            
                            f_out.write(json.dumps(result_entry) + "\n")
                    
                    f_out.flush()
                    os.fsync(f_out.fileno())

            os.rename(temp_shard_path, shard_file_path)
            print(f"Shard {args.shard_id}: Done -> {shard_file_path}")
        
        else:
            if os.path.exists(temp_shard_path):
                os.rename(temp_shard_path, shard_file_path)

    if args.shard_id == 0:
        print("Supervisor: Checking shards...")
        expected_files = [f"{final_output_path}.shard_{i}" for i in range(args.num_shards)]
        
        wait_counts = 0
        while True:
            if all(os.path.exists(f) for f in expected_files):
                print("Supervisor: All shards ready. Merging...")
                time.sleep(2)
                run_merge(args, final_output_path)
                break
            
            if wait_counts > 1000:
                print("Supervisor: Timeout waiting for shards.")
                break

            time.sleep(20)
            wait_counts += 1

if __name__ == "__main__":
    main()
