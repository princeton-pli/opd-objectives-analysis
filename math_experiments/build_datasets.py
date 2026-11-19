import os

import numpy as np
import pandas as pd
from datasets import load_dataset

from utils import extract_answer_math

REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def main():
    src = pd.concat([load_dataset("EleutherAI/hendrycks_math", s, split="train", revision=REVISION).to_pandas()
                     for s in SUBJECTS], ignore_index=True)
    answers = src["solution"].map(extract_answer_math)
    train = pd.DataFrame({"prompt": src["problem"] + INSTRUCTION, "answer": answers, "level": src["level"],
                          "type": src["type"], "source": "hendrycks_math"})
    idx = np.sort(np.random.default_rng(0).choice(len(src), 500, replace=False))
    train500 = pd.DataFrame({"problem": src["problem"].values[idx], "answer": answers.values[idx],
                             "level": src["level"].values[idx], "type": src["type"].values[idx], "train_index": idx})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, df in (("math_train.parquet", train), ("math_train500_eval.parquet", train500)):
        df.to_parquet(os.path.join(OUTPUT_DIR, name), index=False)
        print(f"Saved {len(df)} problems to {os.path.join(OUTPUT_DIR, name)}")


if __name__ == "__main__":
    main()
