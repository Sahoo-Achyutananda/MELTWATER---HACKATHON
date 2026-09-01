"""Train the BiLSTM masked-letter predictor (+ game-state features) on
train.txt.

Batches are formed within same-length buckets (no padding needed -- every
word in a batch has identical sequence length). Each word gets a random
mask fraction per batch (sampled per-batch, roughly uniform in [0.15, 0.85])
so the model learns to work from both early-game (mostly masked) and
late-game (mostly revealed) board states, since real games pass through
the whole range.

Alongside masking, each example also gets a synthetic "guessed-wrong"
letter set and a matching "remaining guesses" value: pick a random count
of wrong guesses (0-5), sample that many letters from the alphabet that
are NOT in the word (a real wrong guess couldn't have revealed anything),
build the 26-dim indicator vector, and set remaining = (6 - count) / 6.
This is sampled independently of the masking fraction -- a real game
correlates the two, but decoupling them here still teaches the model to
condition on both signals rather than ignore one.
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from bilstm_model import ALPHABET, BiLSTMMasker, LETTER_IDX, MASK_TOKEN, LETTER_TOKEN_OFFSET, MAX_WRONG

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bilstm_attn_feat_masker.pt")

SEED = 42
VAL_FRAC = 0.1
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
MIN_BUCKET_SIZE = 8  # skip length buckets too small to batch sensibly


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def words_by_length(words):
    buckets = {}
    for w in words:
        buckets.setdefault(len(w), []).append(w)
    return buckets


def make_batch(words_of_len_L, device):
    L = len(words_of_len_L[0])
    n = len(words_of_len_L)
    batch_ids = torch.empty((n, L), dtype=torch.long)
    targets = torch.full((n, L), -100, dtype=torch.long)
    guessed_wrong = torch.zeros((n, 26), dtype=torch.float32)
    remaining = torch.empty((n, 1), dtype=torch.float32)

    mask_frac = random.uniform(0.15, 0.85)
    for i, w in enumerate(words_of_len_L):
        masked_any = False
        for j, c in enumerate(w):
            true_idx = LETTER_IDX[c]
            if random.random() < mask_frac:
                batch_ids[i, j] = MASK_TOKEN
                targets[i, j] = true_idx
                masked_any = True
            else:
                batch_ids[i, j] = true_idx + LETTER_TOKEN_OFFSET
        if not masked_any:
            j = random.randrange(L)
            batch_ids[i, j] = MASK_TOKEN
            targets[i, j] = LETTER_IDX[words_of_len_L[i][j]]

        absent_letters = [c for c in ALPHABET if c not in w]
        n_wrong = min(random.randint(0, MAX_WRONG - 1), len(absent_letters))
        for c in random.sample(absent_letters, n_wrong):
            guessed_wrong[i, LETTER_IDX[c]] = 1.0
        remaining[i, 0] = (MAX_WRONG - n_wrong) / MAX_WRONG

    return batch_ids.to(device), targets.to(device), guessed_wrong.to(device), remaining.to(device)


def iter_batches(buckets, batch_size):
    batches = []
    for L, words in buckets.items():
        if len(words) < MIN_BUCKET_SIZE:
            continue
        shuffled = words[:]
        random.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            chunk = shuffled[i:i + batch_size]
            if len(chunk) >= 2:
                batches.append(chunk)
    random.shuffle(batches)
    return batches


def main(epochs=EPOCHS):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    all_words = load_words(TRAIN_PATH)
    random.shuffle(all_words)
    n_val = int(len(all_words) * VAL_FRAC)
    val_words = all_words[:n_val]
    train_words = all_words[n_val:]
    print(f"train={len(train_words)} val={len(val_words)}")

    buckets = words_by_length(train_words)
    model = BiLSTMMasker().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # cosine decay from LR down to ~0 over the full run -- ties T_max to
    # whatever `epochs` is passed in, so raising epochs later still decays
    # smoothly across the new full length with no further code changes.
    # Earlier runs plateaued at a flat 1e-3 well before the loss stopped
    # improving in absolute terms, which is exactly what this addresses.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        batches = iter_batches(buckets, BATCH_SIZE)
        total_loss = 0.0
        n_batches = 0
        for chunk in batches:
            ids, targets, guessed_wrong, remaining = make_batch(chunk, device)
            logits = model(ids, guessed_wrong, remaining)
            loss = loss_fn(logits.reshape(-1, 26), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        print(f"epoch {epoch}/{epochs}  loss={total_loss/n_batches:.4f}  "
              f"lr={lr_now:.2e}  batches={n_batches}  time={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    main(epochs=args.epochs)
