"""Char-level Transformer encoder (BERT-style), trained with a masked-
language-model objective, plus the same guessed-wrong-letters/remaining-
guesses features used on the BiLSTM branches.

Why a pure Transformer instead of another recurrent variant: a BiLSTM
processes the sequence through recurrence, so information from distant
positions has to survive many recurrent steps to influence a far-away
blank; adding attention on top (approach/bilstm-attention) patches that,
but the recurrence backbone is still there. A Transformer encoder has no
recurrence at all -- every position attends directly to every other
position from layer one, with no compression bottleneck to design around.

The tradeoff recurrence gave for free and attention doesn't: sequence
order. A BiLSTM inherently knows position because it walks the sequence;
pure self-attention treats every position identically unless told
otherwise, so this model adds a learned positional embedding on top of
the character embedding -- not optional, without it the model can't tell
position 2 from position 7.

Both game-state features are computed from (pattern, guessed_letters)
alone -- no interface change needed:
  - guessed_wrong = guessed_letters - set(revealed letters in pattern)
  - remaining = MAX_WRONG - len(guessed_wrong), normalized to [0, 1]

Training: take a word, randomly mask a subset of positions (blank token),
predict the true letter at each masked position from full bidirectional
self-attention. Alongside the masking, synthesize a plausible guessed-
wrong set and remaining-guess count per example, same as the BiLSTM
branches.

Inference: feed the real board mask + guessed-wrong/remaining features
through the trained model, sum the softmax probability mass per letter
across all blanks, zero out already-guessed letters, rank what's left.
"""
from __future__ import annotations

from typing import Set

import torch
import torch.nn as nn

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_IDX = {c: i for i, c in enumerate(ALPHABET)}  # target classes 0-25
MASK_TOKEN = 0
LETTER_TOKEN_OFFSET = 1  # input vocab: 0=MASK, 1..26='a'..'z'
VOCAB_SIZE = 27
MAX_WRONG = 6
MAX_LEN = 32  # longest train.txt/test.txt word is 29; leaves headroom


def word_to_input_ids(word: str) -> torch.Tensor:
    return torch.tensor([LETTER_IDX[c] + LETTER_TOKEN_OFFSET for c in word], dtype=torch.long)


def pattern_to_input_ids(pattern: str) -> torch.Tensor:
    ids = [MASK_TOKEN if c == "_" else LETTER_IDX[c] + LETTER_TOKEN_OFFSET for c in pattern]
    return torch.tensor(ids, dtype=torch.long)


def guessed_wrong_vector(pattern: str, guessed_letters: Set[str]) -> torch.Tensor:
    """26-dim binary vector: 1 where that letter has been guessed and is
    confirmed NOT in the word (a miss), 0 otherwise."""
    revealed = {c for c in pattern if c != "_"}
    wrong = guessed_letters - revealed
    return torch.tensor([1.0 if c in wrong else 0.0 for c in ALPHABET], dtype=torch.float32)


def remaining_feature(pattern: str, guessed_letters: Set[str]) -> torch.Tensor:
    """Scalar in [0, 1]: fraction of the 6-wrong-guess budget still left."""
    revealed = {c for c in pattern if c != "_"}
    n_wrong = len(guessed_letters - revealed)
    remaining = max(0, MAX_WRONG - n_wrong)
    return torch.tensor([remaining / MAX_WRONG], dtype=torch.float32)


class TransformerMasker(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 3, num_heads: int = 4,
                 ff_dim: int = 256, dropout: float = 0.2, max_len: int = MAX_LEN,
                 global_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # game-state features: 26 (guessed-wrong binary) + 1 (remaining, in
        # [0,1]) -> a small global vector, broadcast onto every position
        # and concatenated before the final per-position prediction.
        self.global_encoder = nn.Sequential(
            nn.Linear(26 + 1, global_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_model + global_dim, 26)

    def forward(self, x: torch.Tensor, guessed_wrong: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids.
        guessed_wrong: (batch, 26) float. remaining: (batch, 1) float.
        -> (batch, seq_len, 26) logits."""
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        e = self.emb(x) + self.pos_emb(positions)  # (batch, seq_len, d_model)

        out = self.encoder(e)  # (batch, seq_len, d_model)

        global_feat = torch.cat([guessed_wrong, remaining], dim=-1)  # (batch, 27)
        global_enc = self.global_encoder(global_feat)  # (batch, global_dim)
        global_enc = global_enc.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, global_dim)

        combined = torch.cat([out, global_enc], dim=-1)
        return self.head(combined)
