"""
Core policy-building logic for the open-loop Hangman submission format:
each word_id gets ONE fixed 26-letter guess order (no feedback loop).

Strategy per word length L:
  1. Pool training words of length L (widening to nearby lengths if the
     bucket is too small to be statistically reliable).
  2. Greedily build a guess order by simulating ALL pooled words at once:
     at each step, guess the letter that is still needed by the largest
     number of words that are still "alive" (not yet fully solved, not
     yet at 6 misses). This directly optimizes the actual win-rate
     objective, unlike plain letter frequency.
  3. Any letters never chosen (can happen for tiny/degenerate buckets)
     are appended in global-frequency order as a safety tail.

Falls back to the global letter-frequency order for lengths with zero
training coverage.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_WRONG = 6


def load_words(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def words_by_length(words: Sequence[str]) -> Dict[int, List[str]]:
    buckets: Dict[int, List[str]] = {}
    for w in words:
        buckets.setdefault(len(w), []).append(w)
    return buckets


def global_frequency_order(words: Sequence[str]) -> List[str]:
    """Letters ranked by fraction of words containing them (the '1-gram' fallback)."""
    counts = Counter()
    for w in words:
        counts.update(set(w))
    return sorted(ALPHABET, key=lambda c: -counts.get(c, 0))


def pooled_bucket(
    target_len: int,
    buckets: Dict[int, List[str]],
    min_size: int = 500,
    max_radius: int = 6,
) -> List[str]:
    """Widen the length window around target_len until we have enough words."""
    pool: List[str] = list(buckets.get(target_len, []))
    radius = 0
    while len(pool) < min_size and radius < max_radius:
        radius += 1
        for L in (target_len - radius, target_len + radius):
            pool.extend(buckets.get(L, []))
    return pool


def greedy_order(words: Sequence[str], fallback_order: Sequence[str]) -> List[str]:
    """Simulate all `words` simultaneously; at each step pick the letter that
    helps the most still-alive words. Returns a full 26-letter order."""
    if not words:
        return list(fallback_order)

    needed = [set(w) for w in words]  # unique letters not yet guessed, per word
    misses = [0] * len(words)
    alive = set(range(len(words)))
    remaining_letters = set(ALPHABET)
    order: List[str] = []

    for _ in range(26):
        if not remaining_letters or not alive:
            break
        # pick letter maximizing count of alive words that still need it
        best_letter, best_count = None, -1
        for c in remaining_letters:
            cnt = sum(1 for i in alive if c in needed[i])
            if cnt > best_count:
                best_letter, best_count = c, cnt
        order.append(best_letter)
        remaining_letters.discard(best_letter)

        new_alive = set()
        for i in alive:
            if best_letter in needed[i]:
                needed[i].discard(best_letter)
                if needed[i]:
                    new_alive.add(i)
                # else: word fully solved -> drops out (win)
            else:
                misses[i] += 1
                if misses[i] < MAX_WRONG:
                    new_alive.add(i)
                # else: word failed -> drops out (loss)
        alive = new_alive

    # Any letters never picked (rare, tiny buckets) -> append via fallback order
    for c in fallback_order:
        if c in remaining_letters:
            order.append(c)
            remaining_letters.discard(c)
    return order


def build_policies(
    train_words: Sequence[str],
    lengths_needed: Sequence[int],
    min_bucket_size: int = 500,
) -> Dict[int, List[str]]:
    buckets = words_by_length(train_words)
    fallback = global_frequency_order(train_words)
    policies: Dict[int, List[str]] = {}
    for L in lengths_needed:
        pool = pooled_bucket(L, buckets, min_size=min_bucket_size)
        policies[L] = greedy_order(pool, fallback) if pool else fallback
    return policies
