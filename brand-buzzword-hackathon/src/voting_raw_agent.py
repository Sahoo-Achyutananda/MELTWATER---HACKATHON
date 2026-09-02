"""Same two-algorithm hard switch as catboost_raw_agent.py (dictionary
frequency-matching first, classifier fallback when nothing in the
dictionary matches), except the fallback is now a soft-voting ensemble
across every classifier family that's actually been trained (whichever
of catboost/xgboost/lightgbm/random_forest/logreg have a
<family>_raw_models/ directory present) -- not just one.

Soft voting: average each family's P(letter present) rather than having
each family cast one all-or-nothing vote for its single best letter.
Averaging probabilities lets a family that's 90% confident outweigh one
that's a coin flip, instead of every family counting equally regardless
of how sure it is -- the standard tradeoff hard voting gives up.
Auto-detects which families are available, so this still works with
only 1 family trained (falls back to using just that one) all the way
up to all 5.
"""
from __future__ import annotations

import os
from typing import List, Set

import numpy as np

from candidate_agent import CandidateAgent, ALPHABET, LETTER_IDX
from catboost_raw_features import encode_state
from raw_classifier_io import FAMILIES, file_ext, load_classifier, model_dir, predict_proba_positive

DEFAULT_MODELS_ROOT = os.path.dirname(__file__)


class VotingRawAgent:
    def __init__(self, dictionary_words, models_root: str = DEFAULT_MODELS_ROOT,
                 families: List[str] = None):
        self.candidate = CandidateAgent(dictionary_words)

        available = families or FAMILIES
        self.families = []
        # {family: {letter: classifier}}
        self.models = {}
        for family in available:
            d = model_dir(models_root, family)
            ext = file_ext(family)
            if not os.path.isdir(d):
                continue
            letter_models = {}
            for c in ALPHABET:
                path = os.path.join(d, f"{c}.{ext}")
                if not os.path.exists(path):
                    break
                letter_models[c] = load_classifier(family, path)
            if len(letter_models) == 26:
                self.models[family] = letter_models
                self.families.append(family)

        if not self.families:
            raise RuntimeError(
                f"no trained classifier families found under {models_root} "
                f"(looked for: {available}) -- run train_raw_classifier.py first"
            )
        print(f"VotingRawAgent: using families {self.families}")

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

    def _voting_guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        x = encode_state(pattern, guessed_letters).reshape(1, -1)
        unguessed = [c for c in ALPHABET if c not in guessed_letters]

        avg_probs = np.zeros(len(unguessed), dtype=np.float64)
        for family in self.families:
            for i, c in enumerate(unguessed):
                clf = self.models[family][c]
                avg_probs[i] += predict_proba_positive(family, clf, x)[0]
        avg_probs /= len(self.families)

        best_i = int(np.argmax(avg_probs))
        return unguessed[best_i]

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        letter = self._dictionary_guess(pattern, guessed_letters)
        if letter is not None:
            return letter
        return self._voting_guess(pattern, guessed_letters)
