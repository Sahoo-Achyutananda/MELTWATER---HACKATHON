"""BiLSTM + self-attention, trained with a masked-language-model objective,
char-level. Same training/inference story as the plain BiLSTM branch this
was forked from; the only architectural change is a self-attention layer
inserted between the BiLSTM output and the classification head.

Training: take a word, randomly mask a subset of positions (blank token),
predict the true letter at each masked position from bidirectional context
-- exactly BERT-style MLM, applied to characters instead of subword tokens.
This maps directly onto Hangman: a blank in the game *is* a masked position,
and "which letter is most likely under here" is exactly what we need to
rank guesses by.

Why add attention on top of the BiLSTM: the recurrence compresses
long-range context through a fixed-size hidden state, so information from
distant positions can get diluted by the time it reaches a far-away blank
(e.g. in a 15-letter word, what position 1 "knows" has passed through many
recurrent steps before it can influence position 14). Self-attention lets
every position look directly at every other position's BiLSTM output,
weighted by relevance, with no such bottleneck -- cheap to add (one
nn.MultiheadAttention layer) and often a real accuracy bump on top of a
working recurrent baseline.

Inference: feed the real board mask (blanks = unrevealed) through the
trained model, get a softmax distribution per blank position, sum the
probability mass per letter across all blanks (a letter guessed once
reveals every occurrence, so we want its probability of appearing
*anywhere*, not just at one position), zero out already-guessed letters,
and rank what's left.
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


class BiLSTMMasker(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden: int = 128, num_layers: int = 1,
                 dropout: float = 0.1, num_heads: int = 4):
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
        self.head = nn.Linear(hidden * 2, 26)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids -> (batch, seq_len, 26) logits."""
        e = self.emb(x)
        lstm_out, _ = self.lstm(e)  # (batch, seq_len, hidden*2)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, need_weights=False)
        out = self.norm(lstm_out + attn_out)  # residual + layernorm
        return self.head(out)
