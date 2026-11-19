from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveTransformerStudentPolicy(nn.Module):
    """Decoder-only Transformer policy over child-action token prefixes.

    PyTorch exposes the standard decoder-only/GPT-style stack most directly as
    TransformerEncoder layers with PyTorch's causal self-attention mask. The
    learned start embedding supplies the root context without adding a BOS token
    to the action vocabulary, so a binary tree still has vocabulary size 2.
    """

    def __init__(
            self,
            num_actions: int,
            max_prefix_length: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.max_prefix_length = max_prefix_length
        self.token_embedding = nn.Embedding(num_actions, d_model)
        self.start_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Embedding(max_prefix_length + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, num_actions)

    def _hidden_states(
            self, prefixes: torch.Tensor, prefix_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefixes.dim() != 2:
            raise ValueError("prefixes must have shape [batch, max_prefix_length]")
        batch_size, max_observed_prefix_length = prefixes.shape
        if max_observed_prefix_length > self.max_prefix_length:
            raise ValueError(
                f"prefix length {max_observed_prefix_length} exceeds configured maximum "
                f"{self.max_prefix_length}"
            )
        if prefix_lengths is None:
            prefix_lengths = torch.full(
                (batch_size,),
                max_observed_prefix_length,
                dtype=torch.long,
                device=prefixes.device,
            )
        else:
            prefix_lengths = prefix_lengths.to(device=prefixes.device, dtype=torch.long)
        if torch.any(prefix_lengths < 0) or torch.any(prefix_lengths > max_observed_prefix_length):
            raise ValueError("prefix_lengths must be between 0 and prefixes.shape[1]")

        token_embeddings = self.token_embedding(prefixes) if max_observed_prefix_length > 0 else torch.empty(
            batch_size,
            0,
            self.start_embedding.shape[-1],
            device=prefixes.device,
            dtype=self.start_embedding.dtype,
        )
        start_embeddings = self.start_embedding.expand(batch_size, 1, -1)
        x = torch.cat([start_embeddings, token_embeddings], dim=1)
        sequence_length = x.shape[1]
        positions = torch.arange(sequence_length, device=prefixes.device)
        x = x + self.position_embedding(positions)[None, :, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length,
            device=prefixes.device,
        )
        key_padding_mask = None
        if max_observed_prefix_length > 0:
            token_positions = torch.arange(max_observed_prefix_length, device=prefixes.device)[None, :]
            padded_tokens = token_positions >= prefix_lengths[:, None]
            bool_padding_mask = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=prefixes.device), padded_tokens],
                dim=1,
            )
            key_padding_mask = torch.zeros(
                (batch_size, sequence_length),
                dtype=causal_mask.dtype,
                device=prefixes.device,
            )
            key_padding_mask = key_padding_mask.masked_fill(bool_padding_mask, float("-inf"))

        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return hidden, prefix_lengths

    def all_logits(self, prefixes: torch.Tensor, prefix_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return next-action logits at every causal position, including START."""
        hidden, _ = self._hidden_states(prefixes, prefix_lengths)
        return self.output(hidden)

    def logits(self, prefixes: torch.Tensor, prefix_lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden, prefix_lengths = self._hidden_states(prefixes, prefix_lengths)
        next_token_positions = prefix_lengths
        next_hidden = hidden[torch.arange(prefixes.shape[0], device=prefixes.device), next_token_positions]
        return self.output(next_hidden)

    def probs(self, prefixes: torch.Tensor, prefix_lengths: torch.Tensor | None = None) -> torch.Tensor:
        return F.softmax(self.logits(prefixes, prefix_lengths), dim=-1)
