import argparse, json, os
import pandas as pd
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data_parquet", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "math_train.parquet"))
ap.add_argument("--n_eval", type=int, default=400); ap.add_argument("--n_sampled", type=int, default=1)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--max_tokens", type=int, default=4096); ap.add_argument("--max_model_len", type=int, default=5120); ap.add_argument("--tp", type=int, default=1)
a = ap.parse_args()

df = pd.read_parquet(a.data_parquet).head(a.n_eval)
llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.85, tensor_parallel_size=a.tp)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False) for p in df["prompt"]]
outs = llm.generate(prompts, SamplingParams(n=a.n_sampled, temperature=a.temperature, top_p=a.top_p,
                                            max_tokens=a.max_tokens))
rows = [{"gold_answer": str(g), "level": str(l), "type": str(t), "samples": [o.text for o in r.outputs]}
        for g, l, t, r in zip(df["answer"], df["level"], df["type"], outs)]
json.dump(rows, open(a.out, "w"))
print(f"wrote {len(rows)} x {a.n_sampled} to {a.out}")
