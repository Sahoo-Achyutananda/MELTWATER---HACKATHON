"""Hangman agent: dictionary candidate-filtering as the primary strategy,
falling back to the n-gram model when no dictionary word of the right
length is consistent with the revealed pattern + wrong guesses so far
(the expected case for brand names, which won't literally be in train.txt).

Candidate matching is vectorized with numpy: a pure-Python scan over a
30k-word length bucket on every single turn is far too slow to be usable
(the naive version was killed after 5 minutes on a 3k-word sample). Here
each length bucket is a fixed-width (n_words, L) int8 letter-code matrix,
and filtering/counting are done with array ops instead of per-word loops.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set

import numpy as np

from ngram_model import ALPHABET, NgramFallback

LETTER_IDX = {c: i for i, c in enumerate(ALPHABET)}


class CandidateAgent:
    def __init__(self, dictionary_words: Sequence[str], ngram_order: int = 4):
        by_length_words: Dict[int, List[str]] = defaultdict(list)
        for w in dictionary_words:
            by_length_words[len(w)].append(w)

        # (n_words, L) int8 matrix of letter codes, per length bucket
        self.by_length_matrix: Dict[int, np.ndarray] = {}
        for L, words in by_length_words.items():
            mat = np.empty((len(words), L), dtype=np.int8)
            for i, w in enumerate(words):
                for j, c in enumerate(w):
                    mat[i, j] = LETTER_IDX[c]
            self.by_length_matrix[L] = mat

        self.ngram = NgramFallback(dictionary_words, max_order=ngram_order)

        # global letter-inclusion frequency by length, used as a fallback tiebreak
        self.length_prior: Dict[int, Dict[str, float]] = {}
        for L, words in by_length_words.items():
            counts = Counter()
            for w in words:
                counts.update(set(w))
            total = len(words) or 1
            self.length_prior[L] = {c: counts.get(c, 0) / total for c in ALPHABET}

    def _matching_candidates(self, pattern: str, guessed_letters: Set[str]) -> np.ndarray:
        mat = self.by_length_matrix.get(len(pattern))
        if mat is None or mat.shape[0] == 0:
            return mat if mat is not None else np.empty((0, len(pattern)), dtype=np.int8)

        mask = np.ones(mat.shape[0], dtype=bool)
        guessed_codes = [LETTER_IDX[c] for c in guessed_letters]

        for j, pc in enumerate(pattern):
            if pc != "_":
                mask &= mat[:, j] == LETTER_IDX[pc]
            else:
                for code in guessed_codes:
                    mask &= mat[:, j] != code
            if not mask.any():
                break

        return mat[mask]

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        candidates = self._matching_candidates(pattern, guessed_letters)

        if candidates.shape[0] > 0:
            best_letter, best_count = None, -1
            for c in ALPHABET:
                if c in guessed_letters:
                    continue
                count = int((candidates == LETTER_IDX[c]).any(axis=1).sum())
                if count > best_count:
                    best_letter, best_count = c, count
            if best_count > 0:
                return best_letter

        # fallback: n-gram positional model, tie-broken by length prior
        scores = self.ngram.letter_scores(pattern)
        prior = self.length_prior.get(len(pattern), {})
        best_letter, best_score = None, -1.0
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            s = scores.get(c, 0.0) + 0.05 * prior.get(c, 0.0)
            if s > best_score:
                best_letter, best_score = c, s
        return best_letter
