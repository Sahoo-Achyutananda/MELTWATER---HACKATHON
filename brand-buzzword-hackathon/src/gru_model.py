"""Char-level BiGRU, same masked-language-model objective as bilstm_model.py:
randomly mask letters in a word, predict the true letter at each masked
position from bidirectional context. A GRU has fewer gates than an LSTM
(2 vs 3, no separate cell state) -- fewer parameters, typically faster to
train, comparable quality on short sequences like these. Worth comparing
directly against the BiLSTM branch's validated 47.7% win rate.

Inference: identical aggregation as the BiLSTM agent -- sum per-position
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
    def __init__(self, emb_dim: int = 32, hidden: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.gru = nn.GRU(
            emb_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, 26)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids -> (batch, seq_len, 26) logits."""
        e = self.emb(x)
        out, _ = self.gru(e)
        return self.head(out)
