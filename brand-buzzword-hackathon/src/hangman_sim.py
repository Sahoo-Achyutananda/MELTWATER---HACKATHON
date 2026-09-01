"""Local simulator for the REAL adaptive game: guess letters one at a time,
see the revealed pattern, up to 6 wrong guesses. This replaces the earlier
static-order simulator, which assumed a non-adaptive submission format that
the competition overview has since ruled out.
"""
from __future__ import annotations

from typing import Callable, Set, Tuple

MAX_WRONG = 6
BLANK = "_"


def play(word: str, guess_fn: Callable[[str, Set[str]], str]) -> Tuple[bool, int, int]:
    """Runs one game. guess_fn(pattern, guessed_letters) -> next letter to guess.
    Returns (won, wrong_guesses, turns_taken)."""
    guessed: Set[str] = set()
    wrong = 0
    pattern = [BLANK] * len(word)
    turns = 0

    while wrong < MAX_WRONG and BLANK in pattern:
        letter = guess_fn("".join(pattern), set(guessed))
        if letter is None or letter in guessed:
            # defensive: agent misbehaved, treat as a wasted wrong guess
            wrong += 1
            turns += 1
            continue
        guessed.add(letter)
        turns += 1
        if letter in word:
            for i, c in enumerate(word):
                if c == letter:
                    pattern[i] = letter
        else:
            wrong += 1

    won = BLANK not in pattern
    return won, wrong, turns
