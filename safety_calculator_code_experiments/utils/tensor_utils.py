from __future__ import annotations

import torch


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])
