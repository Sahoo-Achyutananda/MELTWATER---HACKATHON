"""BiLSTM trained with a masked-language-model objective, char-level.

Training: take a word, randomly mask a subset of positions (blank token),
predict the true letter at each masked position from bidirectional context
-- exactly BERT-style MLM, applied to characters instead of subword tokens.
This maps directly onto Hangman: a blank in the game *is* a masked position,
and "which letter is most likely under here" is exactly what we need to
rank guesses by.

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
    def __init__(self, emb_dim: int = 32, hidden: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, 26)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) input token ids -> (batch, seq_len, 26) logits."""
        e = self.emb(x)
        out, _ = self.lstm(e)
        return self.head(out)
