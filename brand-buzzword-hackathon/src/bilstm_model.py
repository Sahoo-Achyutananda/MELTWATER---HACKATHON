"""BiLSTM + self-attention + explicit game-state features, trained with a
masked-language-model objective, char-level. Forked from
approach/bilstm-attention with one addition: the model no longer sees only
the board pattern -- it's also fed which letters have already been
confirmed wrong, and how many wrong guesses remain.

Why this matters: the board pattern alone tells the model which positions
are revealed, but says nothing about *failed* guesses -- a wrong guess
doesn't correspond to any position at all, so there was previously no way
to tell the model "e, a, t, o are NOT in this word." That's real evidence
about what kind of word it is (unusual spelling, foreign-origin, etc.),
and until now it was thrown away after being used once to avoid a repeat
guess. Likewise the model had no notion of the guess budget -- it played
identically whether it had 5 wrong guesses left or 1, when a model that
knew the budget could learn to play safer as it runs out.

Both features are computed from (pattern, guessed_letters) alone -- no
interface change needed:
  - guessed_wrong = guessed_letters - set(revealed letters in pattern)
    (a letter that was guessed and IS revealed was a hit, not a miss)
  - remaining = MAX_WRONG - len(guessed_wrong), normalized to [0, 1]

Training: take a word, randomly mask a subset of positions (blank token),
predict the true letter at each masked position from bidirectional
context -- exactly BERT-style MLM, applied to characters instead of
subword tokens. Alongside the masking, also synthesize a plausible
guessed-wrong set (random letters absent from the word) and remaining-
guess count per example, decoupled from the masking itself, so the model
learns to condition on both signals independently.

Inference: feed the real board mask + the real guessed-wrong/remaining
features through the trained model, get a softmax distribution per blank
position, sum the probability mass per letter across all blanks, zero out
already-guessed letters, and rank what's left.
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


class BiLSTMMasker(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden: int = 128, num_layers: int = 1,
                 dropout: float = 0.1, num_heads: int = 4, global_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # self-attention over the BiLSTM outputs: query=key=value=the same
        # sequence, so every position gets a direct, weighted view of every
        # other position instead of only what survives the recurrence.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden * 2, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)

        # game-state features: 26 (guessed-wrong binary) + 1 (remaining, in
        # [0,1]) -> a small global vector, broadcast onto every position
        # and concatenated before the final per-position prediction.
        self.global_encoder = nn.Sequential(
            nn.Linear(26 + 1, global_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden * 2 + global_dim, 26)

    def forward(self, x: torch.Tensor, guessed_wrong: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids.
        guessed_wrong: (batch, 26) float. remaining: (batch, 1) float.
        -> (batch, seq_len, 26) logits."""
        e = self.emb(x)
        lstm_out, _ = self.lstm(e)  # (batch, seq_len, hidden*2)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, need_weights=False)
        out = self.norm(lstm_out + attn_out)  # (batch, seq_len, hidden*2)

        global_feat = torch.cat([guessed_wrong, remaining], dim=-1)  # (batch, 27)
        global_enc = self.global_encoder(global_feat)  # (batch, global_dim)
        global_enc = global_enc.unsqueeze(1).expand(-1, out.size(1), -1)  # (batch, seq_len, global_dim)

        combined = torch.cat([out, global_enc], dim=-1)
        return self.head(combined)
