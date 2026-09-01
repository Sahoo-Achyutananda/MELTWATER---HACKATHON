"""Four-signal combined agent: dictionary candidate-filtering + character
n-gram fallback + the trained BiLSTM (attention + guessed-wrong/remaining
features) + a vowel-ratio guard rail. Mirrors the reference repo's NLP
write-up (pattern-matching dictionary + vowel-ratio heuristic + n-grams)
plus our own BiLSTM branch, combined into one agent instead of kept
separate.

Signals:
  - candidate: fraction of dictionary words (of the right length, still
    consistent with the board + wrong guesses) containing each letter.
    Sharp once the matching pool is small; weak/absent early-game or for
    words that aren't literal dictionary entries.
  - ngram: CandidateAgent's own forward/backward character n-gram model
    (reused directly, not rebuilt) -- a statistical signal that doesn't
    need an exact dictionary match, useful whenever candidate-filtering's
    pool is large or empty.
  - neural: the trained BiLSTM's aggregated per-letter probability,
    conditioned on the board AND the guessed-wrong-letters/remaining-
    guesses features -- generalizes past both the dictionary and simple
    n-gram statistics.
  - vowel guard: once more than half of the currently REVEALED letters are
    vowels, stop guessing further vowels (most words aren't majority-
    vowel, so continued vowel guesses are low-value at that point) --
    applied as a final override on the blended score, not a fourth
    weighted signal.

All three numeric signals are normalized to proper distributions over the
unguessed letters before blending (the ensemble branch's scale-mismatch
bug taught us why this matters -- an unnormalized signal can numerically
dominate regardless of what its blend weight says to trust).

Blend weight: candidate-filtering gets w = K/(K+n_candidates) of the
vote (more candidates -> less trust, same confidence handoff as
approach/ensemble); the remaining weight is split 30/70 between ngram and
neural, favoring the model that generalizes furthest past the dictionary.
"""
from __future__ import annotations

import os
from typing import Set

import torch
import torch.nn.functional as F

from candidate_agent import CandidateAgent, ALPHABET, LETTER_IDX
from bilstm_model import BiLSTMMasker, pattern_to_input_ids, guessed_wrong_vector, remaining_feature

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "bilstm_conv_attn_feat_masker.pt")

CANDIDATE_TRUST_K = 50
NGRAM_WEIGHT_OF_REST = 0.3   # of the weight not given to candidate-filtering
NEURAL_WEIGHT_OF_REST = 0.7
VOWELS = set("aeiou")
VOWEL_RATIO_THRESHOLD = 0.5


class CombinedAgent:
    def __init__(self, dictionary_words, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.candidate = CandidateAgent(dictionary_words)  # also builds its own n-gram fallback
        self.device = torch.device(device)
        self.model = BiLSTMMasker().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def _neural_scores(self, pattern: str, guessed_letters: Set[str]) -> dict:
        ids = pattern_to_input_ids(pattern).unsqueeze(0).to(self.device)
        wrong_vec = guessed_wrong_vector(pattern, guessed_letters).unsqueeze(0).to(self.device)
        remaining = remaining_feature(pattern, guessed_letters).unsqueeze(0).to(self.device)
        logits = self.model(ids, wrong_vec, remaining)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        blank_mask = torch.tensor([c == "_" for c in pattern], device=self.device)
        agg = probs[blank_mask].sum(dim=0) if blank_mask.any() else probs.sum(dim=0)
        return {c: agg[i].item() for i, c in enumerate(ALPHABET)}

    def _candidate_scores(self, pattern: str, guessed_letters: Set[str]):
        candidates = self.candidate._matching_candidates(pattern, guessed_letters)
        n = candidates.shape[0]
        if n == 0:
            return None, 0
        scores = {c: float((candidates == LETTER_IDX[c]).any(axis=1).sum()) / n for c in ALPHABET}
        return scores, n

    def _ngram_scores(self, pattern: str) -> dict:
        return self.candidate.ngram.letter_scores(pattern)

    @staticmethod
    def _normalize(scores: dict, guessed_letters: Set[str]) -> dict:
        total = sum(v for c, v in scores.items() if c not in guessed_letters)
        if total <= 0:
            return {c: 0.0 for c in scores}
        return {c: v / total for c, v in scores.items()}

    @staticmethod
    def _apply_vowel_guard(scores: dict, pattern: str, guessed_letters: Set[str]) -> dict:
        revealed = [c for c in pattern if c != "_"]
        if not revealed:
            return scores
        vowel_ratio = sum(1 for c in revealed if c in VOWELS) / len(revealed)
        if vowel_ratio <= VOWEL_RATIO_THRESHOLD:
            return scores
        return {c: (0.0 if c in VOWELS and c not in guessed_letters else v) for c, v in scores.items()}

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        neural = self._normalize(self._neural_scores(pattern, guessed_letters), guessed_letters)
        ngram = self._normalize(self._ngram_scores(pattern), guessed_letters)
        cand_scores, n_candidates = self._candidate_scores(pattern, guessed_letters)

        if cand_scores is None:
            w_cand = 0.0
            cand_norm = {c: 0.0 for c in ALPHABET}
        else:
            cand_norm = self._normalize(cand_scores, guessed_letters)
            w_cand = CANDIDATE_TRUST_K / (CANDIDATE_TRUST_K + n_candidates)

        rest = 1.0 - w_cand
        blended = {
            c: w_cand * cand_norm[c] + rest * (NGRAM_WEIGHT_OF_REST * ngram[c] + NEURAL_WEIGHT_OF_REST * neural[c])
            for c in ALPHABET
        }
        blended = self._apply_vowel_guard(blended, pattern, guessed_letters)

        best_letter, best_score = None, -1.0
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            if blended[c] > best_score:
                best_letter, best_score = c, blended[c]
        return best_letter
