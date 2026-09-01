"""Character-level n-gram language model, used as a fallback signal when
no dictionary word matches the current revealed pattern (the common case
for brand names, which won't be literal dictionary entries).

Two directional models are kept (forward: predict next char from preceding
context; backward: predict previous char from following context) so a
blank can be scored from whichever neighbor(s) happen to be revealed.
Backoff: try the highest order whose context was seen in training, else
drop to a lower order, else fall back to unigram letter frequency.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
START = "^"
END = "$"


class DirectionalNgram:
    def __init__(self, words: Sequence[str], max_order: int = 4, reverse: bool = False):
        self.max_order = max_order
        # counts[order][context] -> Counter(next_letter -> count)
        self.counts: Dict[int, Dict[str, Counter]] = {o: defaultdict(Counter) for o in range(1, max_order + 1)}
        self.unigram = Counter()

        for w in words:
            seq = w[::-1] if reverse else w
            padded = START * (max_order - 1) + seq + END
            for i in range(max_order - 1, len(padded)):
                nxt = padded[i]
                if nxt == END:
                    continue
                self.unigram[nxt] += 1
                for order in range(1, max_order + 1):
                    if i - order + 1 < 0:
                        continue
                    ctx = padded[i - order + 1:i]
                    self.counts[order][ctx][nxt] += 1

    def scores(self, context: str) -> Dict[str, float]:
        """context: up to max_order-1 known preceding (or following, if this
        is the reverse model) characters, right-aligned to the blank; use
        START for unknown/word-boundary positions. Returns a probability-like
        score per letter (not necessarily normalized across all 26)."""
        context = context[-(self.max_order - 1):] if context else ""
        for order in range(min(self.max_order, len(context) + 1), 0, -1):
            ctx = context[-(order - 1):] if order > 1 else ""
            table = self.counts[order].get(ctx)
            if table:
                total = sum(table.values())
                return {c: table.get(c, 0) / total for c in ALPHABET}
        total = sum(self.unigram.values()) or 1
        return {c: self.unigram.get(c, 0) / total for c in ALPHABET}


class NgramFallback:
    def __init__(self, words: Sequence[str], max_order: int = 4):
        self.forward = DirectionalNgram(words, max_order, reverse=False)
        self.backward = DirectionalNgram(words, max_order, reverse=True)
        self.max_order = max_order

    def letter_scores(self, pattern: str) -> Dict[str, float]:
        """pattern: e.g. 'ap_l_'. Returns an aggregate score per letter,
        summed across all blank positions (higher = more likely to appear
        somewhere in the word)."""
        n = len(pattern)
        agg: Dict[str, float] = {c: 0.0 for c in ALPHABET}
        k = self.max_order - 1

        for i, ch in enumerate(pattern):
            if ch != "_":
                continue
            left = pattern[max(0, i - k):i]
            left_ctx = "".join(c if c != "_" else START for c in left)
            left_ctx = (START * (k - len(left_ctx)) + left_ctx) if len(left_ctx) < k else left_ctx

            right = pattern[i + 1:i + 1 + k]
            right_ctx_raw = "".join(c if c != "_" else START for c in right)
            # backward model was trained on reversed words, so feed it the
            # right-hand context reversed (nearest neighbor first)
            right_ctx = right_ctx_raw[::-1]

            fwd = self.forward.scores(left_ctx)
            bwd = self.backward.scores(right_ctx)
            for c in ALPHABET:
                agg[c] += 0.5 * fwd[c] + 0.5 * bwd[c]

        return agg
