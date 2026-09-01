"""Wraps the trained BiLSTM masked-letter predictor into the standard
guess(pattern, guessed_letters) interface used by hangman_sim.py, so it
drops into the same validation harness as CandidateAgent.

Inference: run the model on the current board mask, take the softmax
distribution at every blank position, sum the probability mass per letter
across all blanks (guessing a letter reveals every occurrence at once, so
we want P(letter appears somewhere), not P(letter at one specific spot)),
zero out already-guessed letters, return the argmax.
"""
from __future__ import annotations

import os
from typing import Set

import torch
import torch.nn.functional as F

from bilstm_model import ALPHABET, BiLSTMMasker, pattern_to_input_ids

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "bilstm_attn_masker.pt")


class BiLSTMAgent:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = BiLSTMMasker().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def guess(self, pattern: str, guessed_letters: Set[str]) -> str:
        ids = pattern_to_input_ids(pattern).unsqueeze(0).to(self.device)  # (1, L)
        logits = self.model(ids)  # (1, L, 26)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (L, 26)

        blank_mask = torch.tensor([c == "_" for c in pattern], device=self.device)
        if blank_mask.any():
            agg = probs[blank_mask].sum(dim=0)  # (26,)
        else:
            agg = probs.sum(dim=0)

        best_letter, best_score = None, -1.0
        for i, c in enumerate(ALPHABET):
            if c in guessed_letters:
                continue
            score = agg[i].item()
            if score > best_score:
                best_letter, best_score = c, score
        return best_letter
