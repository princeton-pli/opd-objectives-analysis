from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from vllm import LLM, SamplingParams

from boundary import continuation_token_span, read_jsonl
from hit_rates import classify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.prompts)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    forced = llm.generate([r["prompt"] + r["continuation"] for r in rows],
                          SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    expected = []
    for r, sc in zip(rows, forced):
        start, full_ids = continuation_token_span(tokenizer, r["prompt"], r["continuation"])
        assert start is not None, r["task_name"]
        logp = sum(sc.prompt_logprobs[pos][full_ids[pos]].logprob for pos in range(start, len(full_ids)))
        expected.append(math.exp(logp))

    gens = llm.generate([r["prompt"] for r in rows],
                        SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                       max_tokens=args.max_tokens, n=args.samples, seed=args.seed))
    per_prompt = []
    totals = {"name": 0}
    n_total = 0
    for r, g in zip(rows, gens):
        counts = {"name": 0}
        for out in g.outputs:
            counts["name"] += classify(out.text, r["continuation"], r["call_suffix"])
        n = len(g.outputs); n_total += n
        for k in totals:
            totals[k] += counts[k]
        per_prompt.append({"task_name": r["task_name"], "n": n, **counts,
                           "expected_name_p": expected[len(per_prompt)],
                           "completions": [out.text for out in g.outputs]})

    rates = {k: v / n_total for k, v in totals.items()}
    se = {k: math.sqrt(rates[k] * (1 - rates[k]) / n_total) for k in rates}
    result = {
        "model": args.model, "prompts": str(args.prompts), "n_prompts": len(rows),
        "samples_per_prompt": args.samples, "n_samples": n_total,
        "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
        "rates": rates, "standard_errors": se,
        "expected_name_rate": statistics.fmean(expected),
        "per_prompt": per_prompt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.model} on {len(rows)} prompts x {args.samples} samples (T={args.temperature}):")
    for k in rates:
        print(f"  {k:10s} {rates[k]:.3f} +- {se[k]:.3f}")
    print(f"  expected name rate (mean P(name|prompt)) {result['expected_name_rate']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
