"""Vectorized (numpy) local-search refinement of a per-length guess order.

The greedy order in policy.py is a fast approximation. Here we hill-climb
on top of it: try swapping pairs of positions within the first `top_k` slots
(later slots rarely affect win rate, since most words resolve within ~15
guesses under a 6-miss budget) and keep any swap that improves win rate,
measured on the same training pool the order was built from.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_IDX = {c: i for i, c in enumerate(ALPHABET)}
MAX_WRONG = 6


def words_to_matrix(words: Sequence[str]) -> np.ndarray:
    """(n, 26) boolean matrix: contains[i, j] = word i has letter j."""
    n = len(words)
    mat = np.zeros((n, 26), dtype=bool)
    for i, w in enumerate(words):
        for c in set(w):
            mat[i, LETTER_IDX[c]] = True
    return mat


def simulate(order: Sequence[str], contains: np.ndarray):
    """Returns (won: bool array, misses: int array) after playing `order`
    against every word until it's solved or hits MAX_WRONG misses."""
    n = contains.shape[0]
    needed_count = contains.sum(axis=1)
    revealed = np.zeros(n, dtype=np.int32)
    misses = np.zeros(n, dtype=np.int32)
    alive = np.ones(n, dtype=bool)
    won = np.zeros(n, dtype=bool)

    for c in order:
        if not alive.any():
            break
        col = LETTER_IDX[c]
        is_hit = contains[:, col]
        active_hit = alive & is_hit
        revealed[active_hit] += 1
        active_miss = alive & ~is_hit
        misses[active_miss] += 1

        won_now = alive & (revealed >= needed_count)
        lost_now = alive & (misses >= MAX_WRONG)
        won |= won_now
        alive &= ~won_now & ~lost_now

    return won, misses


def win_rate(order: Sequence[str], contains: np.ndarray) -> float:
    if contains.shape[0] == 0:
        return 0.0
    won, _ = simulate(order, contains)
    return won.mean()


def composite_score(order: Sequence[str], contains: np.ndarray) -> float:
    """Leaderboard-aligned objective: whole part = wins (dominant), decimal
    part = efficiency (fewer misses, including on words you don't win).
    Wins are weighted far above any possible efficiency swing so the search
    never trades a win away for efficiency -- it only breaks ties with it."""
    n = contains.shape[0]
    if n == 0:
        return 0.0
    won, misses = simulate(order, contains)
    efficiency = (1.0 - misses / MAX_WRONG).mean()  # in [0, 1]
    return won.sum() * 10.0 + efficiency


def local_search(
    order: List[str],
    words: Sequence[str],
    top_k: int = 15,
    max_passes: int = 5,
    sample_cap: int = 6000,
    seed: int = 0,
) -> List[str]:
    if not words:
        return order
    sample = words
    if len(sample) > sample_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sample), size=sample_cap, replace=False)
        sample = [sample[i] for i in idx]

    contains = words_to_matrix(sample)
    best_order = list(order)
    best_score = composite_score(best_order, contains)

    for _ in range(max_passes):
        improved = False
        k = min(top_k, len(best_order))
        for i in range(k):
            for j in range(i + 1, k):
                cand = list(best_order)
                cand[i], cand[j] = cand[j], cand[i]
                score = composite_score(cand, contains)
                if score > best_score:
                    best_order, best_score = cand, score
                    improved = True
        if not improved:
            break

    return best_order
