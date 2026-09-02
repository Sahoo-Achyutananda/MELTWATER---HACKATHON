"""Two algorithms, hard switch: dictionary frequency-matching as the
primary algorithm, 26 separate CatBoost classifiers (raw pattern
features, no pre-computed signals) as the fallback for words that don't
match anything in the dictionary.

Hard switch, not a blend: if any dictionary word (of the right length,
consistent with the board + wrong guesses) still matches, guess the
letter most common among those matches. Otherwise, ask each of the 26
per-letter classifiers for P(letter present) and take the argmax among
unguessed letters.
"""
from __future__ import annotations

import os
from typing import Set

import numpy as np
from catboost import CatBoostClassifier

from candidate_agent import CandidateAgent, ALPHABET, LETTER_IDX
from catboost_raw_features import encode_state

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "catboost_raw_models")


class CatBoostRawAgent:
    def __init__(self, dictionary_words, models_dir: str = DEFAULT_MODELS_DIR):
        self.candidate = CandidateAgent(dictionary_words)
        self.models = {}
        for c in ALPHABET:
            clf = CatBoostClassifier()
            clf.load_model(os.path.join(models_dir, f"{c}.cbm"))
            self.models[c] = clf

    def _dictionary_guess(self, pattern: str, guessed_letters: Set[str]):
        candidates = self.candidate._matching_candidates(pattern, guessed_letters)
        if candidates.shape[0] == 0:
            return None
        best_letter, best_count = None, -1
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            count = int((candidates == LETTER_IDX[c]).any(axis=1).sum())
            if count > best_count:
                best_letter, best_count = c, count
        return best_letter if best_count > 0 else None

    def _catboost_guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        x = encode_state(pattern, guessed_letters).reshape(1, -1)
        best_letter, best_prob = None, -1.0
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            prob = self.models[c].predict_proba(x)[0, 1]
            if prob > best_prob:
                best_letter, best_prob = c, prob
        return best_letter

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        letter = self._dictionary_guess(pattern, guessed_letters)
        if letter is not None:
            return letter
        return self._catboost_guess(pattern, guessed_letters)
