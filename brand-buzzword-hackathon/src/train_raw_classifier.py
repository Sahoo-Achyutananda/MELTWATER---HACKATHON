"""Train 26 separate per-letter classifiers on raw positional/pattern
features, for whichever classifier family you pass via --model
(catboost, xgboost, lightgbm, random_forest, logreg -- see
raw_classifier_io.py). Same data-generation recipe as train_catboost_raw.py
(kept there too, for the CatBoost-only path); this version just swaps in
whichever library raw_classifier_io.build_classifier() hands back.

Run once per family you want to try, e.g.:
    python src/train_raw_classifier.py --model xgboost
    python src/train_raw_classifier.py --model lightgbm
    python src/train_raw_classifier.py --model random_forest
    python src/train_raw_classifier.py --model logreg
Each run only needs its own library installed -- doesn't touch or
require the others.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from catboost_raw_features import ALPHABET, LETTER_IDX, encode_state, N_FEATURES
from raw_classifier_io import build_classifier, save_classifier, file_ext, model_dir

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
MODELS_ROOT = os.path.dirname(__file__)

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
    labels = np.zeros((n_states, 26), dtype=np.int8)
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
            print(f"  {kept}/{n_states} states  elapsed={time.time()-t0:.0f}s")

    print(f"dataset: {kept} states, {time.time()-t0:.0f}s")
    return X[:kept], labels[:kept], eligible[:kept]


def main(family: str, n_states=N_STATES):
    random.seed(SEED)
    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    train_words = all_words[n_val:]

    print(f"model family: {family}")
    print(f"generating {n_states} synthetic states...")
    X, labels, eligible = generate_dataset(train_words, n_states)

    out_dir = model_dir(MODELS_ROOT, family)
    os.makedirs(out_dir, exist_ok=True)
    ext = file_ext(family)

    for c in ALPHABET:
        k = LETTER_IDX[c]
        mask = eligible[:, k]
        Xc, yc = X[mask], labels[mask, k]
        pos_rate = yc.mean() if len(yc) else 0.0
        print(f"letter '{c}': {mask.sum()} training rows, positive_rate={pos_rate:.3f}")

        clf = build_classifier(family, SEED)
        t0 = time.time()
        clf.fit(Xc, yc)
        save_classifier(clf, family, os.path.join(out_dir, f"{c}.{ext}"))
        print(f"  trained+saved in {time.time()-t0:.0f}s")

    print(f"all 26 {family} models saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["catboost", "xgboost", "lightgbm", "random_forest", "logreg"])
    p.add_argument("--n-states", type=int, default=N_STATES)
    args = p.parse_args()
    main(family=args.model, n_states=args.n_states)
