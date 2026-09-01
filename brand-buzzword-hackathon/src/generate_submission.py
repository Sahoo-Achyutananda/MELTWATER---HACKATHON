"""Generate submission.csv by actually playing the adaptive game against
test.txt, per the competition's implementation blueprint: for each word,
run our own local game loop (board mask + guessed letters only -> next
guess), terminating the instant a 6th wrong guess lands or the word is
fully solved, and join the guesses made into one flat string.

This is *not* a precomputed order -- it's a genuine per-word transcript,
correctly stopping at exactly 6 wrong guesses (the blueprint's own sample
code has an off-by-one, allowing a 7th guess; the "Constraints & Edge
Cases" section is authoritative and says the loop terminates at the 6th
wrong guess, so that's what this implements).
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from candidate_agent import CandidateAgent

MAX_WRONG = 6
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
TEST_PATH = os.path.join(DATA_DIR, "test.txt")
SUB_PATH = os.path.join(DATA_DIR, "submission.csv")


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def play_and_record(agent, word):
    board = ["_"] * len(word)
    guessed_seq = []
    guessed_set = set()
    wrong = 0

    while wrong < MAX_WRONG and "_" in board:
        nxt = agent.guess("".join(board), guessed_set)
        guessed_seq.append(nxt)
        guessed_set.add(nxt)
        if nxt in word:
            board = [word[i] if word[i] == nxt else board[i] for i in range(len(word))]
        else:
            wrong += 1

    return "".join(guessed_seq), wrong, ("_" not in board)


def main(limit=None):
    train_words = load_words(TRAIN_PATH)
    test_words = load_words(TEST_PATH)
    if limit:
        test_words = test_words[:limit]

    t0 = time.time()
    agent = CandidateAgent(train_words)
    print(f"agent built on {len(train_words)} words in {time.time()-t0:.1f}s")

    rows = []
    wins = 0
    total_wrong = 0
    t0 = time.time()
    for i, w in enumerate(test_words):
        seq, wrong, won = play_and_record(agent, w)
        wins += int(won)
        total_wrong += wrong
        rows.append((i, seq))
        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_words) - (i + 1)) / rate
            print(f"  {i+1}/{len(test_words)}  win_rate_so_far={wins/(i+1):.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    n = len(test_words)
    print(f"FINAL win_rate={wins/n:.4f} avg_wrong={total_wrong/n:.2f} n={n} "
          f"time={time.time()-t0:.0f}s")

    with open(SUB_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_id", "guessed_letters_string"])
        writer.writerows(rows)
    print(f"wrote {SUB_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(limit=args.limit)
