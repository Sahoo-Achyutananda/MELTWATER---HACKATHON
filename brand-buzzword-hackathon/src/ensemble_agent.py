"""Ensemble of dictionary candidate-filtering (approach/candidate-ngram,
39-40% win rate alone) and the trained BiLSTM (approach/bilstm, 47.7% win
rate alone). Neither dominates the other across the whole game:

- Candidate-filtering is extremely sharp once the matching pool has
  narrowed to a handful of words (can often nail the exact word), but its
  signal is weak/noisy early-game when thousands of words still match, and
  it has *nothing* to say when zero dictionary words match at all (the
  expected case for brand names not literally in train.txt).
- The BiLSTM has no such cliff -- it always produces a distribution, and
  it generalizes to words that stump the dictionary -- but it doesn't
  exploit exact dictionary consistency the way candidate-filtering can
  once the pool is small.

So: blend the two, with the blend weight shifting toward candidate-
filtering as its matching pool shrinks (more confident), and toward the
BiLSTM as the pool grows or vanishes (candidate-filtering's signal is
weak or nonexistent). This is a smooth confidence-based handoff rather
than a hard threshold.
"""
from __future__ import annotations

import os
from typing import Set

import numpy as np
import torch
import torch.nn.functional as F

from candidate_agent import CandidateAgent, ALPHABET, LETTER_IDX
from bilstm_model import BiLSTMMasker, pattern_to_input_ids

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "bilstm_masker.pt")

# candidate-filtering trust half-life: at CANDIDATE_TRUST_K remaining
# candidates, the blend is 50/50; fewer candidates -> trust it more,
# more candidates -> trust the neural model more.
CANDIDATE_TRUST_K = 50


class EnsembleAgent:
    def __init__(self, dictionary_words, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.candidate = CandidateAgent(dictionary_words)
        self.device = torch.device(device)
        self.model = BiLSTMMasker().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def _neural_scores(self, pattern: str) -> dict:
        ids = pattern_to_input_ids(pattern).unsqueeze(0).to(self.device)
        logits = self.model(ids)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (L, 26)
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

    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        neural = self._neural_scores(pattern)
        cand_scores, n_candidates = self._candidate_scores(pattern, guessed_letters)

        if cand_scores is None:
            blended = neural
        else:
            w = CANDIDATE_TRUST_K / (CANDIDATE_TRUST_K + n_candidates)
            blended = {c: w * cand_scores[c] + (1 - w) * neural[c] for c in ALPHABET}

        best_letter, best_score = None, -1.0
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            if blended[c] > best_score:
                best_letter, best_score = c, blended[c]
        return best_letter
