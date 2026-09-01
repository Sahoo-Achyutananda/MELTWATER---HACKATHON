"""Validate a policy (per-length guess order) by simulating the open-loop
Hangman game against held-out words, measuring win rate."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

MAX_WRONG = 6


def simulate_word(order: Sequence[str], word: str) -> Tuple[bool, int]:
    """Apply `order` to `word` until solved or 6 wrong guesses. Returns (won, misses)."""
    needed = set(word)
    misses = 0
    for c in order:
        if not needed:
            break
        if c in needed:
            needed.discard(c)
        else:
            misses += 1
            if misses >= MAX_WRONG:
                return False, misses
    return (len(needed) == 0), misses


def evaluate_policy(policies: Dict[int, List[str]], words: Sequence[str]) -> dict:
    wins = 0
    total_misses = 0
    total_efficiency = 0.0
    for w in words:
        order = policies.get(len(w))
        if order is None:
            continue
        won, misses = simulate_word(order, w)
        wins += int(won)
        total_misses += misses
        total_efficiency += 1.0 - misses / MAX_WRONG
    n = len(words)
    return {
        "n": n,
        "win_rate": wins / n if n else 0.0,
        "avg_misses": total_misses / n if n else 0.0,
        # Leaderboard-style score on this sample: whole part = wins,
        # decimal = mean efficiency (fewer misses raises it).
        "leaderboard_score": wins + (total_efficiency / n if n else 0.0),
    }
