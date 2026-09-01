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

## The mechanic (confirmed from the competition's Implementation Blueprint)

Adaptive, interactive Hangman — guess a letter, see what's revealed, guess
again, up to 6 wrong guesses. This rules out any strategy that commits to a
fixed guess order in advance; the model must react to the revealed pattern
each turn.

Two separate scoring layers, not one:
- **Sandbox / live leaderboard** (48h dev period only): we run our own game
  loop locally against the public `test.txt`, recording the actual guesses
  made turn by turn, and upload the transcript as `submission.csv`
  (`word_id, guessed_letters_string`). Kaggle already knows `test.txt`'s
  answers, so it scores this by replaying our recorded sequence against
  them — no model execution needed on their end for this part. Any
  character that doesn't reveal a new letter (wrong, duplicate, space,
  invalid symbol) counts as one strike; play stops the instant strike 6
  lands or the word is solved, and anything left over in the string is
  ignored. Score = `wins.efficiency` (whole number = words won, decimal =
  an efficiency term rewarding fewer wrong guesses) — confirmed against the
  live leaderboard: our BiLSTM submission scored ~49.1, matching its ~49%
  validated win rate almost exactly, so "whole number" is a percentage-
  scale win rate, not a raw count out of the full sample.
- **Final judgement (decides hiring outcomes)**: Meltwater re-runs the
  *submitted model/notebook itself* against a completely separate, locked
  private word list we never see. The sandbox CSV is explicitly called a
  dev-only checkpoint — it proves nothing about the private run except that
  the model behaves the way we think it does. Judging also explicitly
  weights **advanced architectures (Transformers, RNNs, Embeddings, Custom
  Neural Networks) above N-gram/frequency baselines**, and disqualifies any
  hardcoding/lookup-leak tactics regardless of sandbox score.

Because of this, every agent's `guess(pattern, guessed_letters)` function is
architecturally blind to the real word — it only ever receives the board
mask and guess history, exactly what a real opponent would see. The *game
loop* around it (which needs the true word to check hits/misses) is kept
strictly separate from the *model*, so a strong local score reflects real
model behavior rather than an artifact of the answers being visible in
`test.txt`.

## Approach branches

| Branch | Idea | Status | Validated win rate* |
|---|---|---|---|
| `approach/static-order-legacy` | Precompute one fixed 26-letter guess order per word length (no adaptation) | Superseded — built before the mechanic was confirmed adaptive. Kept for reference/comparison only. | 13.7% (wrong metric for this competition) |
| `approach/candidate-ngram` | Dictionary candidate-filtering (guess the letter most common among words still consistent with the pattern) + character n-gram fallback for words that don't match any dictionary entry (the expected case for brand names) | Working baseline | **39.3%** |
| `approach/bilstm` | BiLSTM predicting next-letter distribution from masked pattern (MLM objective), trained on train.txt | Validated | **47.7%** |
| `approach/bilstm-attention` | Same BiLSTM + a self-attention layer between the recurrent output and the classification head | Built, training/validation pending | — |
| `approach/gru` | Same MLM approach, GRU instead of LSTM | Built, training/validation pending | — |
| `approach/gru-attention` | GRU + the same self-attention addition | Built, training/validation pending | — |
| `approach/ensemble` | Blends candidate-filtering and the BiLSTM: weight shifts toward candidate-filtering as its matching dictionary pool shrinks (sharp late-game), toward the BiLSTM as it grows or hits zero (robust early-game / off-dictionary words) | Built, training/validation pending | — |
| `approach/rl` | Reinforcement learning agent | Not started, low priority (real risk of high effort for uncertain/negative gain within the remaining time; judging's "advanced architecture" bar is already met by the BiLSTM/GRU branches) | — |

\* Win rate measured by holding out 10% of train.txt, building the model on
the remaining 90%, and playing full interactive games (see each branch's
`src/validate_*.py`) against held-out words. This is a proxy for the real
private leaderboard — held-out English words are structurally closer to
train.txt than true brand names will be, so expect the real score to
differ, likely downward.

## Working on this

Each branch is self-contained under `brand-buzzword-hackathon/src/`. Switch
to a branch and read its module docstrings for the approach-specific design
notes.
