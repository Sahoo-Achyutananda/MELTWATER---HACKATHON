"""Train 26 separate CatBoost binary classifiers (one per letter) on raw
positional/pattern features: each classifier learns "is letter X present
in the word" directly from the masked pattern + guessed letters, with no
pre-computed candidate/ngram/neural signals involved (contrast with
approach/catboost-meta, which deliberately reuses those signals instead).

train.txt only contains whole words, not partial-reveal game states, so
there's no way around simulating them -- but every word used is a real
word straight from the competition's own train.txt, nothing substituted
or fabricated. For each of N random (word, mask-fraction, synthetic
wrong-guesses) draws, one shared 60-dim raw feature vector is built in
memory; for each letter not yet guessed in that state, it contributes one
training example to that letter's own classifier. Nothing is written to
a separate dataset file -- features are generated and consumed directly
in this one script.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from catboost_raw_features import ALPHABET, LETTER_IDX, encode_state, N_FEATURES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "catboost_raw_models")

SEED = 42
VAL_FRAC = 0.1
N_STATES = 60000
MAX_WRONG = 6


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def make_synthetic_state(word, rng):
    L = len(word)
    mask_frac = rng.uniform(0.15, 0.85)
    revealed_idx = [i for i in range(L) if rng.random() >= mask_frac]
    if not revealed_idx and L > 1:
        revealed_idx = [rng.randrange(L)]
    pattern = ["_"] * L
    for i in revealed_idx:
        pattern[i] = word[i]
    revealed_letters = set(pattern) - {"_"}

    absent_letters = [c for c in ALPHABET if c not in word]
    n_wrong = min(rng.randint(0, MAX_WRONG - 1), len(absent_letters))
    wrong_letters = set(rng.sample(absent_letters, n_wrong)) if n_wrong else set()

    return "".join(pattern), revealed_letters | wrong_letters


def generate_dataset(words, n_states, seed=SEED):
    rng = random.Random(seed)
    X = np.empty((n_states, N_FEATURES), dtype=np.float32)
    # label[i, k] = 1 if letter k is in the word for state i, else 0
    labels = np.zeros((n_states, 26), dtype=np.int8)
    # eligible[i, k] = True if letter k was NOT yet guessed in state i
    # (only eligible rows are used to train letter k's classifier)
    eligible = np.zeros((n_states, 26), dtype=bool)

    t0 = time.time()
    kept = 0
    for i in range(n_states):
        word = rng.choice(words)
        pattern, guessed_letters = make_synthetic_state(word, rng)
        if pattern.count("_") == 0:
            continue
        X[kept] = encode_state(pattern, guessed_letters)
        for c in ALPHABET:
            k = LETTER_IDX[c]
            if c in guessed_letters:
                continue
            eligible[kept, k] = True
            labels[kept, k] = 1 if c in word else 0
        kept += 1
        if kept % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {kept}/{n_states} states  elapsed={elapsed:.0f}s")

    print(f"dataset: {kept} states, {time.time()-t0:.0f}s")
    return X[:kept], labels[:kept], eligible[:kept]


def main(n_states=N_STATES):
    random.seed(SEED)
    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    train_words = all_words[n_val:]

    print(f"generating {n_states} synthetic states...")
    X, labels, eligible = generate_dataset(train_words, n_states)

    os.makedirs(MODELS_DIR, exist_ok=True)
    for c in ALPHABET:
        k = LETTER_IDX[c]
        mask = eligible[:, k]
        Xc, yc = X[mask], labels[mask, k]
        pos_rate = yc.mean() if len(yc) else 0.0
        print(f"letter '{c}': {mask.sum()} training rows, positive_rate={pos_rate:.3f}")

        clf = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            loss_function="Logloss", random_seed=SEED, verbose=False,
        )
        t0 = time.time()
        clf.fit(Xc, yc)
        clf.save_model(os.path.join(MODELS_DIR, f"{c}.cbm"))
        print(f"  trained+saved in {time.time()-t0:.0f}s")

    print(f"all 26 models saved to {MODELS_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-states", type=int, default=N_STATES)
    args = p.parse_args()
    main(n_states=args.n_states)
