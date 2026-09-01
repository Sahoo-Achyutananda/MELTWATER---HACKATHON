"""Train the BiLSTM masked-letter predictor on train.txt.

Batches are formed within same-length buckets (no padding needed -- every
word in a batch has identical sequence length). Each word gets a random
mask fraction per batch (sampled per-batch, roughly uniform in [0.15, 0.85])
so the model learns to work from both early-game (mostly masked) and
late-game (mostly revealed) board states, since real games pass through
the whole range.
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from bilstm_model import BiLSTMMasker, LETTER_IDX, MASK_TOKEN, LETTER_TOKEN_OFFSET

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bilstm_masker.pt")

SEED = 42
VAL_FRAC = 0.1
BATCH_SIZE = 256
EPOCHS = 6
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
    batch_ids = torch.empty((len(words_of_len_L), L), dtype=torch.long)
    targets = torch.full((len(words_of_len_L), L), -100, dtype=torch.long)

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

    return batch_ids.to(device), targets.to(device)


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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        batches = iter_batches(buckets, BATCH_SIZE)
        total_loss = 0.0
        n_batches = 0
        for chunk in batches:
            ids, targets = make_batch(chunk, device)
            logits = model(ids)
            loss = loss_fn(logits.reshape(-1, 26), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch}/{epochs}  loss={total_loss/n_batches:.4f}  "
              f"batches={n_batches}  time={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    main(epochs=args.epochs)
