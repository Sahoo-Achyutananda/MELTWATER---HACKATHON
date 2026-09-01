"""
1. Split train.txt into train/val.
2. Build the global-frequency baseline and the greedy simultaneous-simulation
   policy from the train split only.
3. Evaluate both on the held-out val split -> win rate + leaderboard-style
   score comparison (whole = wins, decimal = efficiency, per the "Whole
   number = words won; decimal = efficiency score" leaderboard rule).
4. Rebuild the greedy + refined policy on the FULL train.txt (max data) and
   generate submission.csv for every word in test.txt (using only word
   length, never the actual test letters -- avoids leaking the "answer").
"""
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from policy import build_policies, global_frequency_order, words_by_length, pooled_bucket
from evaluate import evaluate_policy
from optimize import local_search

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
TEST_PATH = os.path.join(DATA_DIR, "test.txt")
SUB_PATH = os.path.join(DATA_DIR, "submission.csv")

SEED = 42
VAL_FRAC = 0.1
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def report(name, result, n):
    print(f"[{name:24s}] win_rate={result['win_rate']:.4f} "
          f"avg_misses={result['avg_misses']:.2f} "
          f"score~={result['leaderboard_score']:.1f} / {n}")


def main():
    random.seed(SEED)
    all_train = load_words(TRAIN_PATH)
    random.shuffle(all_train)
    n_val = int(len(all_train) * VAL_FRAC)
    val_words = all_train[:n_val]
    train_words = all_train[n_val:]

    lengths_in_val = sorted(set(len(w) for w in val_words))
    print(f"train={len(train_words)} val={len(val_words)}")

    # --- Baseline: single global frequency order for every length ---
    baseline_order = global_frequency_order(train_words)
    baseline_policy = {L: baseline_order for L in lengths_in_val}
    report("baseline global-freq", evaluate_policy(baseline_policy, val_words), len(val_words))

    # --- Per-length frequency (no greedy simulation) ---
    buckets = words_by_length(train_words)
    freq_policy = {}
    for L in lengths_in_val:
        pool = buckets.get(L, train_words)
        counts = Counter()
        for w in pool:
            counts.update(set(w))
        freq_policy[L] = sorted(ALPHABET, key=lambda c: -counts.get(c, 0))
    report("per-length frequency", evaluate_policy(freq_policy, val_words), len(val_words))

    # --- Greedy simultaneous-simulation policy (win-count objective) ---
    greedy_policy = build_policies(train_words, lengths_in_val)
    report("greedy simulation", evaluate_policy(greedy_policy, val_words), len(val_words))

    # --- Local-search refinement: wins first, misses as tiebreaker ---
    refined_policy = {}
    for L in lengths_in_val:
        pool = pooled_bucket(L, buckets, min_size=500)
        refined_policy[L] = local_search(greedy_policy[L], pool)
    report("local-search refined", evaluate_policy(refined_policy, val_words), len(val_words))

    # --- Rebuild on full train.txt for the actual submission ---
    test_words = load_words(TEST_PATH)  # length only is used below
    lengths_needed = sorted(set(len(w) for w in test_words))
    final_greedy = build_policies(all_train, lengths_needed)
    full_buckets = words_by_length(all_train)
    final_policy = {}
    for L in lengths_needed:
        pool = pooled_bucket(L, full_buckets, min_size=500)
        final_policy[L] = local_search(final_greedy[L], pool)

    with open(SUB_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_id", "guessed_letters_string"])
        for i, w in enumerate(test_words):
            order = final_policy[len(w)]
            writer.writerow([i, "".join(order)])
    print(f"wrote {SUB_PATH} ({len(test_words)} rows)")

    # Proxy score against the actual public test.txt words (evaluation only,
    # never used to build the model -- same leak-safety as before).
    report("final vs test.txt", evaluate_policy(final_policy, test_words), len(test_words))


if __name__ == "__main__":
    main()
