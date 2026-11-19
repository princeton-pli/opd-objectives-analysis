import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SEQ = 300
POS_PER_SEQ = 6
SEED = 0
TRAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "math_train.parquet")


def build_texts(tokenizer, parquet, gen_json):
    df = pd.read_parquet(parquet).head(400)
    gen = json.load(open(gen_json))
    assert len(gen) == 400
    texts, meta = [], []
    for i in range(N_SEQ):
        prompt, rec = df.iloc[i]["prompt"], gen[i]
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        completion = rec["samples"][0]
        texts.append(chat + completion)
        meta.append({"i": i, "prompt_chars": len(chat)})
    return texts, meta


def probe_positions(tokenizer, texts, meta, rng):
    """Token positions strictly inside the completion; the position INDEXES the
    prefix ending there, i.e. we probe the next-token distribution at that point."""
    out = []
    for text, m in zip(texts, meta):
        enc = tokenizer(text, return_tensors="np")
        ids = enc["input_ids"][0]
        prompt_len = len(tokenizer(text[:m["prompt_chars"]], return_tensors="np")["input_ids"][0])
        lo, hi = prompt_len + 5, len(ids) - 2
        if hi <= lo:
            continue
        pos = sorted(rng.choice(np.arange(lo, hi),
                                size=min(POS_PER_SEQ, hi - lo), replace=False).tolist())
        out.append({"ids": ids, "pos": pos, "i": m["i"]})
    return out


@torch.no_grad()
def gather_logprobs(model_path, seqs, device="cuda"):
    """Return per (seq, pos) full-vocab log-probs (fp32 numpy)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    rows = []
    for s in seqs:
        ids = torch.tensor(s["ids"][None], device=device)
        logits = model(ids).logits[0].float()          # (T, V)
        lp = torch.log_softmax(logits, dim=-1)
        for j, p in enumerate(s["pos"]):
            row = lp[p]
            rows.append(((s["i"], p), row.cpu().numpy().astype(np.float32)))
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher_outputs", required=True, help="JSON file written by generate_math.py for the teacher")
    ap.add_argument("--initial_student", required=True, help="Initial student (model name or path)")
    ap.add_argument("--trained_student", required=True, help="Trained student (path of the run)")
    ap.add_argument("--output_file", required=True, help="Output .npz file")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.initial_student)
    rng = np.random.default_rng(SEED)
    texts, meta = build_texts(tok, TRAIN_FILE, a.teacher_outputs)
    seqs = probe_positions(tok, texts, meta, rng)

    base = dict(gather_logprobs(a.initial_student, seqs))
    K = sorted(base)
    q0 = np.exp(np.stack([base[k] for k in K]).astype(np.float32))
    del base
    final = dict(gather_logprobs(a.trained_student, seqs))
    q = np.exp(np.stack([final[k] for k in K]).astype(np.float32))
    del final

    r = np.arange(len(K))
    t = (q - q0).argmax(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.output_file)), exist_ok=True)
    np.savez(a.output_file, problem=np.array([k[0] for k in K]), position=np.array([k[1] for k in K]),
             token=t, q0=q0[r, t], q_final=q[r, t])
    print(f"{len(K)} positions; initial probability of the most increased token: median {np.median(q0[r, t]):.3f}, "
          f"below 0.05: {np.mean(q0[r, t] < 0.05):.3f}; saved {a.output_file}")


if __name__ == "__main__":
    main()
