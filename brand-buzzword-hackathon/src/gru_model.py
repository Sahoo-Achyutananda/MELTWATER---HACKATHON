"""Char-level BiGRU + self-attention, same masked-language-model objective
as bilstm_model.py: randomly mask letters in a word, predict the true
letter at each masked position from bidirectional context. A GRU has fewer
gates than an LSTM (2 vs 3, no separate cell state) -- fewer parameters,
typically faster to train, comparable quality on short sequences like
these.

Mirrors approach/bilstm-attention's change on top of approach/gru: a
self-attention layer between the recurrent output and the classification
head, so every position gets a direct, weighted view of every other
position instead of only what survives the recurrence.

Inference: identical aggregation as the other agents -- sum per-position
softmax probability across all blanks, zero out already-guessed letters,
take the argmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_IDX = {c: i for i, c in enumerate(ALPHABET)}  # target classes 0-25
MASK_TOKEN = 0
LETTER_TOKEN_OFFSET = 1  # input vocab: 0=MASK, 1..26='a'..'z'
VOCAB_SIZE = 27


def word_to_input_ids(word: str) -> torch.Tensor:
    return torch.tensor([LETTER_IDX[c] + LETTER_TOKEN_OFFSET for c in word], dtype=torch.long)


def pattern_to_input_ids(pattern: str) -> torch.Tensor:
    ids = [MASK_TOKEN if c == "_" else LETTER_IDX[c] + LETTER_TOKEN_OFFSET for c in pattern]
    return torch.tensor(ids, dtype=torch.long)


class BiGRUMasker(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden: int = 128, num_layers: int = 1,
                 dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.gru = nn.GRU(
            emb_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden * 2, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Linear(hidden * 2, 26)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids -> (batch, seq_len, 26) logits."""
        e = self.emb(x)
        gru_out, _ = self.gru(e)  # (batch, seq_len, hidden*2)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out, need_weights=False)
        out = self.norm(gru_out + attn_out)  # residual + layernorm
        return self.head(out)
