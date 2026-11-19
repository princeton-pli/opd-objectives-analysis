# Math Experiments

See the [main README](../README.md) for the paper, the other experiments, and the citation.

## Table of Contents

- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Math Experiments (Section 4.2.2)](#math-experiments-section-422)
  - [1. Creating the Datasets](#1-creating-the-datasets)
  - [2. Initial Evaluation](#2-initial-evaluation)
  - [3. On-Policy Distillation](#3-on-policy-distillation)
  - [4. Tokens With the Largest Probability Increase](#4-tokens-with-the-largest-probability-increase)

## Installation

1. Go to this folder from the repository root, then create and activate a Conda environment:
```sh
cd math_experiments
conda create -n opd_math python=3.10 -y
conda activate opd_math
```
2. Install the requirements, pinned to the versions used for the paper's runs:
```sh
pip install -r requirements.txt
```

## Running the Experiments

All experiments are launched from this folder (`cd math_experiments` from the repository root) through the bash scripts in [`scripts/`](./scripts).

| Script | Purpose |
|---|---|
| [`build_datasets.sh`](./scripts/build_datasets.sh) | Builds the training set and the 500 training problems used for evaluation |
| [`run_initial_eval.sh`](./scripts/run_initial_eval.sh) | Evaluates the initial students and the teachers |
| [`run_opd.sh`](./scripts/run_opd.sh) | Runs on-policy distillation and evaluates the trained student on MATH500 |
| [`run_probe.sh`](./scripts/run_probe.sh) | Finds the tokens whose probability increased the most during training |

The `STUDENT` and `TEACHER` variables select a pair of models of the paper:

| Variable | Values | Models |
|---|---|---|
| `STUDENT` | `0.6B`, `1.7B` | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) |
| `TEACHER` | `4B`, `8B`, `14B`, `32B` | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) (thinking disabled) |

`OBJECTIVES` (`fkl`, `rkl`) and `SEEDS` (`42`, `43`, `44`) select a subset of the runs of a pair, for example `STUDENT=0.6B TEACHER=4B OBJECTIVES=rkl SEEDS=42 bash scripts/run_opd.sh`.
Results are saved under `outputs/` (set `OUTPUT_ROOT` to save them elsewhere).

## Math Experiments (Section 4.2.2)

We train [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) students using [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) teachers, via token-level OPD with FKL and RKL, on the 7,500 problems of the [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) training set, and evaluate them on [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

### 1. Creating the Datasets

<details>
<summary><strong>Command for creating the datasets (CPU):</strong></summary>

```sh
bash scripts/build_datasets.sh
```
The datasets will be saved under `datasets/`: `math_train.parquet` (the training set) and `math_train500_eval.parquet` (500 training problems drawn uniformly at random, used to evaluate the initial students).
</details>

### 2. Initial Evaluation

Evaluates the initial students on MATH500 and on the 500 training problems, and the teachers on MATH500.
For each problem, eight outputs are sampled with temperature 1.0 and top-p 0.8 (at most 8,192 tokens), and an output is correct if its final answer matches the reference answer.

<details>
<summary><strong>Command (1 GPU, 2 GPUs for the 32B teacher):</strong></summary>

```sh
bash scripts/run_initial_eval.sh
```
Set `MODELS` to evaluate a subset, for example `MODELS="0.6B 32B"`. The results will be saved under `outputs/initial_eval/<student|teacher>_<size>/`: `<dataset>.summary.txt` holds the accuracy (the fraction of correct outputs, `accuracy`) and Pass@8 (`pass@8`), where `<dataset>` is `math500_inst` (MATH500) or `mathtrain500_inst` (training problems).
</details>

### 3. On-Policy Distillation

Runs FKL and RKL for three random seeds, each for 400 steps, then evaluates the trained student on MATH500 as in the initial evaluation.

<details>
<summary><strong>Command for the 0.6B student and the 4B teacher (4 GPUs):</strong></summary>

```sh
STUDENT=0.6B TEACHER=4B bash scripts/run_opd.sh
```
Replace `0.6B` and `4B` by the student and teacher of interest. Each run will be saved under `outputs/student_<STUDENT>_teacher_<TEACHER>/<objective>_seed_<seed>/`, which holds the trained student, the training log (`train.log`), TensorBoard logs (`runs/`), and its evaluation (`eval/math500_inst.summary.txt`).
</details>

### 4. Tokens With the Largest Probability Increase

For each of the first 300 training problems, samples one output from the teacher and takes six of its tokens uniformly at random.
At each of the resulting 1,800 positions, finds the token whose probability under the student increased the most during training, and records its probability under the initial student.
Requires the runs of [On-Policy Distillation](#3-on-policy-distillation) for the same student and teacher.

<details>
<summary><strong>Command for the 0.6B student and the 4B teacher (1 GPU, 2 GPUs for the 32B teacher):</strong></summary>

```sh
STUDENT=0.6B TEACHER=4B bash scripts/run_probe.sh
```
The teacher's outputs will be saved to `outputs/probe/teacher_<TEACHER>_outputs.json`, and the results of each run to `outputs/probe/student_<STUDENT>_teacher_<TEACHER>/<objective>_seed_<seed>.npz`: `q0` holds the initial student probability of the most increased token at each position.
</details>
