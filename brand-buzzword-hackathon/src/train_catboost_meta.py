"""Train a CatBoost meta-classifier to replace the hand-tuned blend
weights in combined_agent.py's guess() (the candidate-trust formula and
the 30/70 ngram/neural split). Same three raw signals -- candidate
entropy, character n-gram, trained BiLSTM -- but instead of manually
deciding how to combine them, learn the combination from real synthetic
game states.

Mirrors the reference repo's design (github.com/Aditya-dom/Auto_Hangman,
also seen as trexquant_Hangman): CatBoost trained to predict "is this
letter actually in the word" from features describing the current game
state, reported there at 95% in-dictionary / 67% out-of-dictionary
accuracy. Our version differs in what the model sees: instead of raw
positional/pattern features, it's fed our three ALREADY-COMPUTED signal
scores (candidate/ngram/neural) plus a few auxiliary features -- so
CatBoost's job is specifically to learn the *combination* rule, not
rediscover pattern-matching from scratch (which candidate_agent.py
already does exactly, via exhaustive dictionary search rather than
approximating it).

Dataset: for N random synthetic game states (random word, random mask
fraction, random synthetic guessed-wrong set -- same recipe as
train_bilstm.py), compute the three raw signal scores for every unguessed
letter, label = whether that letter is actually in the word. One row per
(state, unguessed letter).

Needs a trained BiLSTM checkpoint already present (bilstm_conv_attn_feat_
masker.pt) since the neural signal is one of the three features -- run
train_bilstm.py first in the same session.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from combined_agent import CombinedAgent, DEFAULT_MODEL_PATH, ALPHABET

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_meta.cbm")

SEED = 42
VAL_FRAC = 0.1
N_STATES = 40000  # synthetic game states -> ~40000 * ~20 unguessed letters/state training rows
MAX_WRONG = 6
VOWELS = set("aeiou")

FEATURE_NAMES = ["cand_score", "ngram_score", "neural_score", "length",
                  "n_blanks", "n_wrong", "vowel_ratio", "is_vowel"]


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


def state_features(agent, pattern, guessed_letters):
    """Returns (dict of per-letter raw feature tuples, n_candidates)."""
    cand_scores, n_cand = agent._candidate_scores(pattern, guessed_letters)
    cand_scores = cand_scores or {c: 0.0 for c in ALPHABET}
    ngram_scores = agent._ngram_scores(pattern)
    neural_scores = agent._neural_scores(pattern, guessed_letters)

    revealed = [c for c in pattern if c != "_"]
    vowel_ratio = (sum(1 for c in revealed if c in VOWELS) / len(revealed)) if revealed else 0.0
    L = len(pattern)
    n_blanks = pattern.count("_")
    n_wrong = len(guessed_letters - set(revealed))

    per_letter = {}
    for c in ALPHABET:
        per_letter[c] = (
            cand_scores.get(c, 0.0), ngram_scores.get(c, 0.0), neural_scores.get(c, 0.0),
            float(L), float(n_blanks), float(n_wrong), vowel_ratio, 1.0 if c in VOWELS else 0.0,
        )
    return per_letter


def generate_dataset(agent, words, n_states, seed=SEED):
    rng = random.Random(seed)
    X_rows, y_rows = [], []
    t0 = time.time()
    for i in range(n_states):
        word = rng.choice(words)
        pattern, guessed_letters = make_synthetic_state(word, rng)
        if pattern.count("_") == 0:
            continue
        per_letter = state_features(agent, pattern, guessed_letters)
        for c in ALPHABET:
            if c in guessed_letters:
                continue
            X_rows.append(per_letter[c])
            y_rows.append(1 if c in word else 0)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_states - (i + 1))
            print(f"  {i+1}/{n_states} states  rows={len(X_rows)}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"dataset: {len(X_rows)} rows from {n_states} states, {time.time()-t0:.0f}s")
    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


def main(n_states=N_STATES):
    random.seed(SEED)
    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    train_words = all_words[n_val:]

    print("building base agent (candidate + ngram + trained BiLSTM)...")
    agent = CombinedAgent(train_words, model_path=DEFAULT_MODEL_PATH, device="cpu")

    print(f"generating {n_states} synthetic game states...")
    X, y = generate_dataset(agent, train_words, n_states)
    print(f"X shape={X.shape}  positive rate={y.mean():.3f}")

    print("training CatBoostClassifier...")
    t0 = time.time()
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, verbose=100,
    )
    clf.fit(X, y)
    print(f"trained in {time.time()-t0:.0f}s")

    importances = clf.get_feature_importance()
    print("feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name:14s} {imp:.2f}")

    clf.save_model(MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-states", type=int, default=N_STATES)
    args = p.parse_args()
    main(n_states=args.n_states)
