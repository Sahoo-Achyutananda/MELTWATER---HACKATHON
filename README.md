# Brand & Buzzword Hackathon

Play Hangman against a secret test set of brand names: guess letters one at a
time, at most 6 wrong guesses per word. This is a **code competition** — we
submit a Notebook that Kaggle calls interactively during scoring, not a
predictions file. Final ranking (and, per the competition, hiring outcomes)
is decided on the private leaderboard against words we never see.

## Data

`brand-buzzword-hackathon/`
- `train.txt` — 225,300 English dictionary words (no labels), the only
  vocabulary we're given to build a model from.
- `test.txt` — 250,000 more English words. **Not the real target** — the
  actual hidden test set is brand names, which won't literally appear in any
  dictionary. `test.txt` is only useful as extra vocabulary / a public sanity
  check, never as a stand-in for the private set.
- `sample_submission.csv` — leftover from before we confirmed the mechanic;
  not the real submission format (see `approach/static-order-legacy`).

## The mechanic (confirmed from the competition overview)

Adaptive, interactive Hangman — guess a letter, see what's revealed, guess
again, up to 6 wrong guesses. Scoring runs our submitted Notebook's code
directly against the hidden words. This rules out any strategy that commits
to a fixed guess order in advance; the model must react to the revealed
pattern each turn.

Leaderboard score = `wins.efficiency`: whole number is words won, decimal is
an efficiency score that rewards using fewer wrong guesses (so it's worth
optimizing beyond raw win rate once win rate is decent).

## Approach branches

| Branch | Idea | Status | Validated win rate* |
|---|---|---|---|
| `approach/static-order-legacy` | Precompute one fixed 26-letter guess order per word length (no adaptation) | Superseded — built before the mechanic was confirmed adaptive. Kept for reference/comparison only. | 13.7% (wrong metric for this competition) |
| `approach/candidate-ngram` | Dictionary candidate-filtering (guess the letter most common among words still consistent with the pattern) + character n-gram fallback for words that don't match any dictionary entry (the expected case for brand names) | Working baseline | **39.3%** |
| `approach/bilstm` | Bi-LSTM predicting next-letter distribution from masked pattern, trained on train.txt | Not started | — |
| `approach/rl` | Reinforcement learning agent | Not started, low priority (heuristics usually competitive per prior Hangman challenges; time-boxed) | — |

\* Win rate measured by holding out 10% of train.txt, building the model on
the remaining 90%, and playing full interactive games (see
`src/validate_agent.py` on `approach/candidate-ngram`) against held-out
words. This is a proxy for the real private leaderboard — held-out English
words are structurally closer to train.txt than true brand names will be, so
expect the real score to differ, likely downward.

## Working on this

Each branch is self-contained under `brand-buzzword-hackathon/src/`. Switch
to a branch and read its module docstrings for the approach-specific design
notes.
