# Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2611.23397-b31b1b.svg)](https://arxiv.org/abs/2611.23397)

</div>

Official implementation based on the [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [Hugging Face TRL](https://huggingface.co/docs/trl/en/index), and [vLLM](https://github.com/vllm-project/vllm) libraries, as well as the code releases of [Zhao et al. (2026)](https://github.com/siyan-zhao/OPSD) and [Kaur et al. (2026)](https://github.com/princeton-pli/rethinking-opsd-for-thinking-models).

## Repository Structure

The code of each set of experiments is in its own folder, with its own environment and instructions.
Run the commands of an experiment from its folder.

| Folder | Experiments | Paper |
|---|---|---|
| [`binary_tree_experiments`](./binary_tree_experiments) | Synthetic experiments using a binary tree navigation task | Section 3.2, Appendix D.1 |
| [`safety_calculator_code_experiments`](./safety_calculator_code_experiments) | Safety, arithmetic task with a calculator tool, and code completion | Sections 3.3 and 4.2.1, Appendices A.2, D.2.1, and D.2.2 |
| [`math_experiments`](./math_experiments) | Math | Section 4.2.2, Appendix D.3.1 |

## Citation

For citing the paper you can use:
```bibtex
@article{strauss2026which,
  title   = {Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives},
  author  = {Strauss, Hubert and Fowl, Liam and Razin, Noam},
  journal = {arXiv preprint arXiv:2611.23397},
  year    = {2026}
}
```
