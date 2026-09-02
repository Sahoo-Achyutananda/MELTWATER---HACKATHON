"""Validate the CatBoost meta-learner agent under the real adaptive rules,
same held-out-train.txt methodology as every other approach. Run after
train_bilstm.py has produced bilstm_conv_attn_feat_masker.pt AND
train_catboost_meta.py has produced catboost_meta.cbm.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from catboost_agent import CatBoostAgent, DEFAULT_CATBOOST_PATH
from combined_agent import DEFAULT_MODEL_PATH
from hangman_sim import play

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")

SEED = 42
VAL_FRAC = 0.1
VAL_SAMPLE = 3000


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def evaluate(agent, words, progress_every=2000):
    wins, total_wrong = 0, 0
    t0 = time.time()
    for i, w in enumerate(words):
        won, wrong, _ = play(w, agent.guess)
        wins += int(won)
        total_wrong += wrong
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(words) - (i + 1)) / rate
            print(f"  {i+1}/{len(words)}  win_rate_so_far={wins/(i+1):.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    n = len(words)
    return wins / n, total_wrong / n


def main(full=False):
    random.seed(SEED)
    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    val_words = all_words[:n_val]
    train_words = all_words[n_val:]
    sample = val_words if full else val_words[:VAL_SAMPLE]
    print(f"train={len(train_words)} val_sample={len(sample)} (of {len(val_words)}) full={full}")

    agent = CatBoostAgent(train_words, bilstm_model_path=DEFAULT_MODEL_PATH,
                           device="cpu", catboost_path=DEFAULT_CATBOOST_PATH)

    t0 = time.time()
    win_rate, avg_wrong = evaluate(agent, sample)
    print(f"[catboost-meta] win_rate={win_rate:.4f} avg_wrong={avg_wrong:.2f} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    main(full=args.full)
