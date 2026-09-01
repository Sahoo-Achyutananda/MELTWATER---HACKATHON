"""Validate the trained BiGRU agent under the real adaptive rules, same
methodology as validate_bilstm.py / validate_agent.py: hold out 10% of
train.txt, play full interactive games (hangman_sim.play) against words
the model never trained on. Run after train_gru.py has produced
gru_masker.pt.
"""
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from gru_agent import GRUAgent, DEFAULT_MODEL_PATH
from hangman_sim import play

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")

SEED = 42
VAL_FRAC = 0.1
VAL_SAMPLE = 3000  # same sample size as the other approaches, for a fair comparison


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
    sample = val_words[:VAL_SAMPLE]
    print(f"val_sample={len(sample)} (of {len(val_words)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = GRUAgent(model_path=DEFAULT_MODEL_PATH, device=device)

    t0 = time.time()
    win_rate, avg_wrong = evaluate(agent, sample)
    print(f"[gru] win_rate={win_rate:.4f} avg_wrong={avg_wrong:.2f} "
          f"time={time.time()-t0:.0f}s device={device}")


if __name__ == "__main__":
    main()
