"""Wraps CombinedAgent's three raw signals (candidate entropy, n-gram,
neural) plus the trained CatBoost meta-classifier into the standard
guess(pattern, guessed_letters) interface. Replaces combined_agent.py's
hand-tuned blend (candidate-trust formula + 30/70 ngram/neural split)
with a learned combination.
"""
from __future__ import annotations

import os
from typing import Set

import numpy as np
from catboost import CatBoostClassifier

from combined_agent import CombinedAgent, ALPHABET

VOWELS = set("aeiou")
DEFAULT_CATBOOST_PATH = os.path.join(os.path.dirname(__file__), "catboost_meta.cbm")


class CatBoostAgent:
    def __init__(self, dictionary_words, bilstm_model_path: str, device: str = "cpu",
                 catboost_path: str = DEFAULT_CATBOOST_PATH):
        self.base = CombinedAgent(dictionary_words, model_path=bilstm_model_path, device=device)
        self.clf = CatBoostClassifier()
        self.clf.load_model(catboost_path)

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        cand_scores, n_cand = self.base._candidate_scores(pattern, guessed_letters)
        cand_scores = cand_scores or {c: 0.0 for c in ALPHABET}
        ngram_scores = self.base._ngram_scores(pattern)
        neural_scores = self.base._neural_scores(pattern, guessed_letters)

        revealed = [c for c in pattern if c != "_"]
        vowel_ratio = (sum(1 for c in revealed if c in VOWELS) / len(revealed)) if revealed else 0.0
        L = len(pattern)
        n_blanks = pattern.count("_")
        n_wrong = len(guessed_letters - set(revealed))

        candidates_c = [c for c in ALPHABET if c not in guessed_letters]
        X = np.array([[
            cand_scores.get(c, 0.0), ngram_scores.get(c, 0.0), neural_scores.get(c, 0.0),
            float(L), float(n_blanks), float(n_wrong), vowel_ratio, 1.0 if c in VOWELS else 0.0,
        ] for c in candidates_c], dtype=np.float64)

        probs = self.clf.predict_proba(X)[:, 1]
        best_i = int(np.argmax(probs))
        return candidates_c[best_i]
