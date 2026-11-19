# Safety, Calculator, and Code Completion Experiments

See the [main README](../README.md) for the paper, the other experiments, and the citation.

## Table of Contents

- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Safety Experiments (Section 3.3)](#safety-experiments-section-33)
  - [1. Creating the Datasets](#1-creating-the-datasets)
  - [2. Initial Evaluation](#2-initial-evaluation)
  - [3. On-Policy Distillation](#3-on-policy-distillation)
  - [4. Classifying Refusals With an LLM Judge](#4-classifying-refusals-with-an-llm-judge)
- [Arithmetic Task With a Calculator Tool (Appendix A.2)](#arithmetic-task-with-a-calculator-tool-appendix-a2)
  - [1. Creating the Datasets](#1-creating-the-datasets-1)
  - [2. Initial Evaluation](#2-initial-evaluation-1)
  - [3. On-Policy Distillation](#3-on-policy-distillation-1)
- [Code Completion Experiments (Section 4.2.1)](#code-completion-experiments-section-421)
  - [Dataset](#dataset)
  - [1. Supervised Fine-Tuning of the Teacher](#1-supervised-fine-tuning-of-the-teacher)
  - [2. Preparing the Training Dataset and Evaluation](#2-preparing-the-training-dataset-and-evaluation)
  - [3. On-Policy Distillation](#3-on-policy-distillation-2)

## Installation

1. Go to this folder from the repository root, then create and activate the conda environment:
```sh
cd safety_calculator_code_experiments
conda env create -f environment.yml
conda activate opd
```
2. Install the remaining dependencies:
```sh
MAX_JOBS=4 python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```
3. The refusal judge of the safety experiments uses the OpenAI API, so it requires the `OPENAI_API_KEY` environment variable to be set.

## Running the Experiments

All experiments are launched from this folder (`cd safety_calculator_code_experiments` from the repository root) through the bash scripts in [`scripts/`](./scripts).

| Script | Purpose |
|---|---|
| [`build_datasets.sh`](./scripts/build_datasets.sh) | Builds the safety and calculator datasets |
| [`run_initial_eval.sh`](./scripts/run_initial_eval.sh) | Evaluates the untrained student and the teacher |
| [`run_opd.sh`](./scripts/run_opd.sh) | Runs on-policy distillation |
| [`run_judge.sh`](./scripts/run_judge.sh) | Classifies the outputs of the safety experiments with an LLM judge |
| [`finetune_teacher.sh`](./scripts/finetune_teacher.sh) | Fine-tunes the teacher of the code completion experiments |
| [`prepare_code_completion.sh`](./scripts/prepare_code_completion.sh) | Builds the training dataset and evaluation of the code completion experiments |

The `EXPERIMENT` variable selects the setting of an experiment in the paper:

| `EXPERIMENT` | Student | Teacher | Paper |
|---|---|---|---|
| `safety_qwen3` | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Section 3.3 |
| `safety_qwen3_thinking` | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Appendix D.2.1 (thinking enabled) |
| `safety_gemma3` | [Gemma-3-4B-it](https://huggingface.co/google/gemma-3-4b-it) | [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it) | Appendix D.2.1 |
| `calculator_qwen3` | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Appendix A.2 |
| `calculator_gemma3` | [Gemma-3-4B-it](https://huggingface.co/google/gemma-3-4b-it) | [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it) | Appendix D.2.2 |
| `code_completion` | [Qwen2.5-1.5B-Base](https://huggingface.co/Qwen/Qwen2.5-1.5B) | [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) (fine-tuned) | Section 4.2.1 |


`OBJECTIVES` (`seq_rkl`, `tok_rkl`, `tok_fkl`, `tok_js`, `tok_skl`; listed in the On-Policy Distillation step of each experiment) and `SEEDS` (`42`, `43`, `44`) select a subset of the runs of an experiment, for example `EXPERIMENT=safety_qwen3 OBJECTIVES=tok_rkl SEEDS=42 bash scripts/run_opd.sh`.
Each run is saved under `outputs/<EXPERIMENT>/<objective>_seed_<seed>/`, which holds the configuration, TensorBoard logs, and the student's sampled outputs at each evaluation (`responses/eval/`).
Calculator and code completion runs also save metrics under `metrics/`.

## Safety Experiments (Section 3.3)

We train [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and [Gemma-3-4B-it](https://huggingface.co/google/gemma-3-4b-it) students using [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it) teachers, respectively, on prompts from [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak), which contains harmful and benign requests.

### 1. Creating the Datasets

We sample 1,500 requests of each of the four WildJailbreak request types for training and 250 of each type for testing, and wrap them in a prompt template that asks the model to refuse unsafe requests.

<details>
<summary><strong>Command for creating the datasets (CPU):</strong></summary>

```sh
DATASETS=safety bash scripts/build_datasets.sh
```
The datasets will be saved under `datasets/`: `wildjailbreak_v4` for the Qwen3 models and `wildjailbreak_v4_gemma` for the Gemma-3 models.
</details>

### 2. Initial Evaluation

Before training, evaluates the student and the teacher on the training prompts used for evaluation during training.

<details>
<summary><strong>Command for Qwen3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=safety_qwen3 bash scripts/run_initial_eval.sh
```
The results will be saved under `outputs/safety_qwen3/initial_eval_train`.
</details>

<details>
<summary><strong>Command for Qwen3 with thinking enabled (1 GPU):</strong></summary>

```sh
EXPERIMENT=safety_qwen3_thinking bash scripts/run_initial_eval.sh
```
The results will be saved under `outputs/safety_qwen3_thinking/initial_eval_train`.
</details>

<details>
<summary><strong>Command for Gemma 3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=safety_gemma3 bash scripts/run_initial_eval.sh
```
The results will be saved under `outputs/safety_gemma3/initial_eval_train`.
</details>

### 3. On-Policy Distillation

Runs all objectives (`seq_rkl`, `tok_rkl`, `tok_fkl`, `tok_js`) for three random seeds. See [Running the Experiments](#running-the-experiments) for launching a subset of these.

<details>
<summary><strong>Command for Qwen3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=safety_qwen3 bash scripts/run_opd.sh
```
The runs will be saved under `outputs/safety_qwen3`.
</details>

<details>
<summary><strong>Command for Qwen3 with thinking enabled (1 GPU):</strong></summary>

```sh
EXPERIMENT=safety_qwen3_thinking bash scripts/run_opd.sh
```
The runs will be saved under `outputs/safety_qwen3_thinking`. The paper reports the first 329 steps, since not all runs completed four epochs within their allowed time budget.
</details>

<details>
<summary><strong>Command for Gemma 3 (2 GPUs):</strong></summary>

```sh
EXPERIMENT=safety_gemma3 bash scripts/run_opd.sh
```
The runs will be saved under `outputs/safety_gemma3`.
</details>

### 4. Classifying Refusals With an LLM Judge

Classifies the saved outputs of the initial evaluation and of each run as refusals or answers, using gpt-5.6-luna through the OpenAI API.
Replace `safety_qwen3` by the experiment of interest.

<details>
<summary><strong>Command (CPU):</strong></summary>

```sh
EXPERIMENT=safety_qwen3 bash scripts/run_judge.sh
```
The classifications will be saved under the `judge/` directory of each run, with the resulting metrics in `judge/summary.json`.
As in the paper, the metrics are computed on the first 100 evaluated prompts of each request type (400 prompts).
Outputs that the OpenAI API declines to classify are listed in `judge/rejected.jsonl`; we labeled those manually.
</details>

## Arithmetic Task With a Calculator Tool (Appendix A.2)

We train [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and [Gemma-3-4B-it](https://huggingface.co/google/gemma-3-4b-it) students using [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it) teachers, respectively, on arithmetic problems that the student can solve directly or by calling a calculator tool.

### 1. Creating the Datasets

We generate problems using the official generator of the [Mathematics dataset](https://github.com/google-deepmind/mathematics_dataset), with 1,000 training and 1,000 test problems for each of six arithmetic modules (the test problems form the datasets' `validation` split).

<details>
<summary><strong>Command for creating the datasets (CPU):</strong></summary>

```sh
DATASETS=calculator bash scripts/build_datasets.sh
```
The datasets will be saved under `datasets/`: `deepmind_arithmetic_calculator_direct` for the Qwen3 models and `deepmind_arithmetic_calculator_direct_conditional` for the Gemma-3 models.
</details>

### 2. Initial Evaluation

Evaluates the student and the teacher before training, on both training and test prompts.

<details>
<summary><strong>Command for Qwen3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=calculator_qwen3 bash scripts/run_initial_eval.sh
```
The results will be saved under `outputs/calculator_qwen3/initial_eval_train` and `outputs/calculator_qwen3/initial_eval_validation`.
</details>

<details>
<summary><strong>Command for Gemma 3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=calculator_gemma3 bash scripts/run_initial_eval.sh
```
The results will be saved under `outputs/calculator_gemma3/initial_eval_train` and `outputs/calculator_gemma3/initial_eval_validation`.
</details>

### 3. On-Policy Distillation

Runs all objectives (`seq_rkl`, `tok_rkl`, `tok_fkl`, `tok_js`) for three random seeds.
At each evaluation, the student's calculator usage rate and accuracy on a fixed random subset of 500 training prompts are logged as `eval_train_rollout/calculator_rate` and `eval_train_rollout/accuracy` (`eval_test_rollout/*` for 500 test prompts); the teacher's are in the `results.json` of the initial evaluation.
Outputs are scored by [`calculator_metrics.py`](./utils/calculator_metrics.py).

<details>
<summary><strong>Command for Qwen3 (1 GPU):</strong></summary>

```sh
EXPERIMENT=calculator_qwen3 bash scripts/run_opd.sh
```
The runs will be saved under `outputs/calculator_qwen3`.
</details>

<details>
<summary><strong>Command for Gemma 3 (2 GPUs):</strong></summary>

```sh
EXPERIMENT=calculator_gemma3 bash scripts/run_opd.sh
```
The runs will be saved under `outputs/calculator_gemma3`.
</details>

## Code Completion Experiments (Section 4.2.1)

We train a [Qwen2.5-1.5B-Base](https://huggingface.co/Qwen/Qwen2.5-1.5B) student using a [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) teacher, fine-tuned in [step 1](#1-supervised-fine-tuning-of-the-teacher), to complete programs that call Python APIs first released between October 2024 and April 2025.

### Dataset

The dataset is [`data/code_completion/prompts.jsonl`](./data/code_completion/prompts.jsonl), and consists of 96 APIs from 14 libraries, with five programs per API (480 prompts).
Each row is one program, with fields including `prompt` (the text to complete), `continuation` (the API name), `call_suffix` (the code that follows it), `api`, `lib`, and `task_name`.

### 1. Supervised Fine-Tuning of the Teacher

Fine-tunes Qwen3-8B-Base on the 480 prompts and their completions.

<details>
<summary><strong>Command for fine-tuning (SFT) the teacher (1 GPU):</strong></summary>

```sh
bash scripts/finetune_teacher.sh
```
The fine-tuned teacher will be saved under `models/qwen3_8b_base_code_completion`.
</details>

### 2. Preparing the Training Dataset and Evaluation

Builds the training dataset and the evaluation used during training, and measures the success rate of the teacher.

<details>
<summary><strong>Command for preparing the training dataset and evaluation (1 GPU):</strong></summary>

```sh
bash scripts/prepare_code_completion.sh
```
The outputs will be saved under `datasets/code_completion/`.
</details>

### 3. On-Policy Distillation

Runs all objectives (`tok_rkl`, `tok_fkl`, `tok_js`, `tok_skl`) for three random seeds.
`tok_skl` is the equal mixture of the FKL and RKL objectives in the paper, computed as `0.5 * (FKL + RKL)`.

<details>
<summary><strong>Command (1 GPU):</strong></summary>

```sh
EXPERIMENT=code_completion bash scripts/run_opd.sh
```
The runs will be saved under `outputs/code_completion`.
The log-probabilities that the student and the teacher assign to the first token of each API name are saved under `metrics/new_api_recovery/` in each run. Whether a sampled output names the correct API is checked by `classify` in [`hit_rates.py`](./opd/code_completion/hit_rates.py).
</details>
