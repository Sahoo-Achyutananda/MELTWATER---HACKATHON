"""Raw feature encoding for the catboost-raw approach -- deliberately NOT
reusing candidate/ngram/neural signal scores the way approach/catboost-meta
does. CatBoost sees only low-level positional/pattern features and has to
learn letter-presence patterns itself, via 26 separate per-letter
classifiers.

Encoding, one fixed-length vector per game state:
  - pattern: MAX_LEN positions, each 0 (padding past the word's actual
    length), 1 (blank '_'), or 2-27 (that position's known letter, a=2..z=27)
  - guessed: 26-dim binary vector, every letter guessed so far (hit or miss)
  - length: the word's actual length
  - n_wrong: wrong guesses used so far

60 features total (32 + 26 + 1 + 1). Same vector regardless of which
letter's presence you're asking about -- that's the point of the 26-
separate-classifier design: each classifier specializes on one letter
rather than the model being told which letter to score.
"""
from __future__ import annotations

from typing import Set

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_IDX = {c: i for i, c in enumerate(ALPHABET)}
MAX_LEN = 32

N_FEATURES = MAX_LEN + 26 + 1 + 1


def encode_state(pattern: str, guessed_letters: Set[str]) -> np.ndarray:
    x = np.zeros(N_FEATURES, dtype=np.float32)

    for j, c in enumerate(pattern[:MAX_LEN]):
        x[j] = 1.0 if c == "_" else (2.0 + LETTER_IDX[c])

    guessed_offset = MAX_LEN
    for c in guessed_letters:
        x[guessed_offset + LETTER_IDX[c]] = 1.0

    revealed = {c for c in pattern if c != "_"}
    x[MAX_LEN + 26] = float(len(pattern))
    x[MAX_LEN + 26 + 1] = float(len(guessed_letters - revealed))

    return x
