# Binary Tree Navigation Experiments

See the [main README](../README.md) for the paper, the other experiments, and the citation.

## Table of Contents

- [Installation](#installation)
- [Binary Tree Navigation Experiments](#binary-tree-navigation-experiments)
  - [Sequence-Level vs. Token-Level OPD (Section 3.2)](#sequence-level-vs-token-level-opd-section-32)
  - [Token-Level OPD With RKL, FKL, and JS (Appendix D.1)](#token-level-opd-with-rkl-fkl-and-js-appendix-d1)
  - [Gradient Estimate That Depends Only on the Sampled Tokens (Appendix E.1)](#gradient-estimate-that-depends-only-on-the-sampled-tokens-appendix-e1)
  - [Outputs](#outputs)

## Installation

The experiments run on CPU and require only PyTorch, NumPy, and PyYAML.

1. Go to this folder from the repository root, then create and activate a Conda environment:
```sh
cd binary_tree_experiments
conda create -n opd_synth python=3.12 -y
conda activate opd_synth
```
2. Since a GPU is not needed, you can install the smaller CPU-only build of PyTorch before installing the requirements:
```sh
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```
3. Install the requirements:
```sh
pip install -r requirements.txt
```

## Binary Tree Navigation Experiments

All commands are run from this folder (`cd binary_tree_experiments` from the repository root).

We train a Transformer student via OPD on the binary tree navigation task of Section 3.2, where the goal is to navigate from the root of a binary tree to its leftmost leaf.
The teacher assigns probability $1 - \delta$ to the left child at the root and at every node in its left subtree, and is uniform over children if the initial token is incorrect.

Each command below trains the student with several objectives, each starting from the same initial student:
- `seq_reverse_kl`: sequence-level OPD.
- `token_reverse_kl`, `token_forward_kl`, `token_js`: token-level OPD with RKL, FKL, and JS, computed exactly over the full vocabulary.
- `token_sample_reverse_kl`: token-level OPD with RKL, using a gradient estimate that depends only on the sampled tokens.

### Sequence-Level vs. Token-Level OPD (Section 3.2)

<details>
<summary><strong>Command (CPU):</strong></summary>

```sh
python run_autoreg_transformer_synthetic_opd_experiments.py --config run_configs/sequence_vs_token_opd.yaml
```
Trains `seq_reverse_kl` and `token_reverse_kl`. The results will be saved under `outputs/sequence_vs_token_opd_<TIMESTAMP>/`.
</details>

### Token-Level OPD With RKL, FKL, and JS (Appendix D.1)

<details>
<summary><strong>Command (CPU):</strong></summary>

```sh
python run_autoreg_transformer_synthetic_opd_experiments.py --config run_configs/token_level_objectives.yaml
```
Trains `seq_reverse_kl`, `token_reverse_kl`, `token_forward_kl`, and `token_js`. The results will be saved under `outputs/token_level_objectives_<TIMESTAMP>/`.
</details>

### Gradient Estimate That Depends Only on the Sampled Tokens (Appendix E.1)

Appendix E.1 reports that, for token-level OPD with RKL, replacing the exact computation over the full vocabulary with a gradient estimate that depends only on the sampled tokens leads to similar results.

<details>
<summary><strong>Command (CPU):</strong></summary>

```sh
python run_autoreg_transformer_synthetic_opd_experiments.py --config run_configs/sampled_token_rkl.yaml
```
Trains `token_reverse_kl` and `token_sample_reverse_kl`. The results will be saved under `outputs/sampled_token_rkl_<TIMESTAMP>/`.
</details>

### Outputs

Each command takes one to two minutes and saves the following files (pass `--output_dir <DIR>` to save them elsewhere):
- `metrics.csv`: for each objective, the student's success probability (`exact_success_probability`) and probability of the correct token at the root (`root_child_0_probability`) at training steps 0, 1, 10, 20, ..., 300. 
- `sampled_trajectory_divergence_profiles.csv`: for each objective, the RKL between the student and teacher next-token distributions at each token position (`trajectory_step`, counted from 0), averaged over 512 outputs sampled from the student (`mean_divergence`, with standard error `stderr`), at the training steps in `eval_at_steps`. 
- `config.yaml`: the configuration of the run.
