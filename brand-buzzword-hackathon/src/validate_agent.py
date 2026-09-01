"""Validate the CandidateAgent under the REAL adaptive rules: build the
dictionary from a train split, then play full interactive games (via
hangman_sim.play) against held-out words the agent never saw. This is the
metric that actually matters now, replacing the earlier static-order proxy.

Also runs a second validation pass excluding held-out words from the
dictionary entirely by length-bucket sampling of *novel* letter patterns
is not needed -- holding words out of `dictionary_words` already forces
every val word through the n-gram fallback path whenever no other train
word shares its exact spelling, which is the honest brand-name proxy.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from candidate_agent import CandidateAgent
from hangman_sim import play

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")

SEED = 42
VAL_FRAC = 0.1
VAL_SAMPLE = 3000  # cap for runtime; full val set is ~22k words


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def evaluate(agent, words):
    wins = 0
    total_wrong = 0
    for w in words:
        won, wrong, _ = play(w, agent.guess)
        wins += int(won)
        total_wrong += wrong
    n = len(words)
    return wins / n, total_wrong / n


def main():
    random.seed(SEED)
    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    val_words = all_words[:n_val]
    train_words = all_words[n_val:]

    sample = val_words[:VAL_SAMPLE]
    print(f"train={len(train_words)} val_sample={len(sample)} (of {len(val_words)})")

    agent = CandidateAgent(train_words)
    win_rate, avg_wrong = evaluate(agent, sample)
    print(f"[candidate + n-gram fallback] win_rate={win_rate:.4f} avg_wrong={avg_wrong:.2f}")


if __name__ == "__main__":
    main()
