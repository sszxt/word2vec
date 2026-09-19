# word2vec: a from-scratch reproduction

A reproduction of Mikolov, Chen, Corrado & Dean, ["Efficient Estimation of
Word Representations in Vector Space"](paper/1301.3781v3.pdf) (arXiv:1301.3781,
2013) -- the paper that introduced the CBOW and Skip-gram architectures,
better known collectively as word2vec.

This repository implements both architectures from scratch in PyTorch --
vocabulary construction, Huffman-tree hierarchical softmax, training, and the
word-analogy evaluation task -- and trains them on a single GPU on the
public text8 corpus, at a scale small enough to run in minutes rather than
the paper's original multi-day distributed training run.

It is written as a guided, documented reproduction: `docs/` walks through the
paper's ideas in the order they're implemented, including three real training
bugs that came up during this reproduction and how they were diagnosed and
fixed -- one of which was found only after an earlier experiment produced a
confidently wrong conclusion.

## Reading order

| Doc | Covers |
|---|---|
| [`docs/01-background.md`](docs/01-background.md) | Why atomic word representations fail; NNLM/RNNLM and their cost |
| [`docs/02-architectures.md`](docs/02-architectures.md) | CBOW and Skip-gram: what they predict and why |
| [`docs/03-hierarchical-softmax.md`](docs/03-hierarchical-softmax.md) | Huffman-tree softmax: the training-cost trick shared by both models |
| [`docs/04-training.md`](docs/04-training.md) | SGD schedule, batching, three real bugs hit while reproducing this, and CBOW's unspecified backward pass |
| [`docs/05-evaluation-task.md`](docs/05-evaluation-task.md) | The analogy task (3CosAdd) and vocabulary coverage |
| [`docs/06-results.md`](docs/06-results.md) | Full results, compared point-by-point against the paper's tables |
| [`docs/07-negative-sampling.md`](docs/07-negative-sampling.md) | Negative sampling as an alternative loss, and cross-validation against gensim |

## Results at a glance

CBOW vs Skip-gram, dim=100, full text8 (16.7M tokens), 1 epoch, mean ± stdev
over 3 seeds:

| Architecture | Semantic | Syntactic | Total |
|---|---|---|---|
| CBOW | 6.03 ± 0.03 | 8.53 ± 0.44 | 7.49 ± 0.27 |
| Skip-gram | 15.57 ± 0.92 | 20.08 ± 0.19 | 18.21 ± 0.32 |

This reproduces the paper's central qualitative finding -- Skip-gram beats
CBOW by a wide margin on semantic analogies -- at roughly 1/50th of the
paper's smallest data size. See [`docs/06-results.md`](docs/06-results.md)
for the full dimensionality, data-amount and epoch sweeps and a
point-by-point comparison against the paper's tables, including two places
where our results disagree with theirs and why.

## Repository layout

```
paper/            the source paper (PDF)
docs/             concept notes, paired with the implementation, read in order
src/word2vec/     library code
  vocab.py           vocabulary + frequency counting
  huffman.py         Huffman tree for hierarchical softmax
  dataset.py         CBOW / Skip-gram training pair generation
  models.py          CBOW and Skip-gram models; hierarchical softmax and negative sampling losses
  train.py           training loop (SGD, linear LR decay, gradient clipping)
  evaluate.py        analogy evaluation (3CosAdd)
scripts/          CLI entry points (download data, train, evaluate, nearest neighbours, run the full experiment suite)
tests/            unit tests for every module above
results/          results.{json,csv} and results_summary.csv from the experiment suite (checkpoints are gitignored)
```

## Setup

Requires Python >= 3.10 and (recommended) a CUDA-capable GPU -- training
works on CPU too, just slower.

```
pip install -e .
python scripts/download_text8.py
python scripts/build_vocab.py
python scripts/download_analogies.py
```

This downloads text8 (~95MB), builds a frequency-filtered vocabulary
(`min_count=5`, ~71k words), and fetches the word-analogy test set (19,544
questions).

## Training a model

```
python -m word2vec.train --arch cbow --dim 100 --out results/cbow.pt
python -m word2vec.train --arch skipgram --dim 100 --out results/skipgram.pt
```

Key flags: `--dim` (vector size), `--epochs`, `--batch-size`, `--lr`,
`--max-words` (truncate the corpus, for data-scaling experiments),
`--vocab-size` (restrict to the N most frequent words), `--window`. Defaults
match the paper: CBOW uses a symmetric 4-word window (8 words of context
total), Skip-gram uses a dynamic window up to 10, both start at learning
rate 0.025 decayed linearly to (approximately) zero.

Two flags control choices the paper does not specify, both documented in
[`docs/04-training.md`](docs/04-training.md): `--context-grad` selects how
CBOW distributes gradient to its context words (`mean` is the true
derivative, `sum` reproduces reference `word2vec.c` and is what the
experiment suite uses), and `--grad-clip` overrides a clipping threshold
that otherwise scales with batch size.

`--loss {hs,ns}` switches between hierarchical softmax (the paper's own
method, default) and negative sampling (`--negative` sets noise-word count),
documented in [`docs/07-negative-sampling.md`](docs/07-negative-sampling.md)
-- like `--sample`, this is beyond what this paper specifies.

## Evaluating a checkpoint

```
python scripts/evaluate.py results/cbow.pt
```

Prints per-category accuracy plus semantic / syntactic / total rollups on
the analogy task.

## Exploring the vectors

```
python scripts/nearest.py king france water
```

Prints nearest neighbours by cosine similarity, several checkpoints side by
side. Comparing CBOW against Skip-gram on the same query shows the difference
the accuracy tables report.

For an interactive version, open [`docs/explorer.html`](docs/explorer.html) in
a browser -- no server, no network requests. It has neighbour lookup, live
`vec(b) − vec(a) + vec(c)` analogy arithmetic over the real test questions, a
2D projection of the vector space, a Huffman-path visualizer for
`docs/03-hierarchical-softmax.md`, and the result charts. Regenerate it after
training with:

```
python scripts/export_viz.py
```

The committed copy is generated from the checkpoints this repository was
developed against. Note that the interactive tabs search a truncated
vocabulary using quantized vectors, so accuracy seen there runs well above the
real benchmark -- the per-category table in its Results tab is computed over
the full vocabulary and is the number that matters.

## Cross-validating against gensim

```
pip install -e ".[reference]"
python scripts/reference_gensim.py --arch cbow --dim 100 --loss ns
```

text8 is too small for the paper's own numbers to tell a correct
implementation apart from a subtly broken one (they come from 50-400x more
data). `scripts/reference_gensim.py` trains gensim -- a well-tested port of
`word2vec.c` -- with matching hyperparameters on the same corpus and
vocabulary, and scores both models with our own evaluator, isolating whether
a gap comes from our implementation rather than from scale. See
[`docs/07-negative-sampling.md`](docs/07-negative-sampling.md).

## Reproducing the full experiment suite

```
python scripts/run_experiments.py
```

Runs all six experiments described in
[`docs/06-results.md`](docs/06-results.md) -- architecture comparison, CBOW
dimensionality/data-amount grid, Skip-gram dimensionality sweep, epochs vs
data amount, and subsampling and negative-sampling arms marked as
beyond-the-paper -- over 3 seeds each (93 runs, `--seeds` to change; `--only`
to (re)run a subset). Writes per-run `results/results.{json,csv}` and
per-configuration means and standard deviations to
`results/results_summary.csv`.

Takes roughly an hour on a single consumer GPU. Skip-gram's larger pair
count dominates the cost -- see the training-time breakdown in
[`docs/06-results.md`](docs/06-results.md).

## Tests

```
pytest tests/
```

Unit tests cover Huffman tree correctness (prefix-free codes, frequency-depth
ordering), CBOW/Skip-gram pair generation, gradient flow through both models,
and the analogy evaluation logic (including a hand-constructed example where
the correct vector arithmetic answer is known exactly).

## What this does not reproduce

- **Scale.** text8 is ~17M tokens; the paper's main results use 783M-6B.
  Every absolute accuracy number here is well below the paper's, though the
  qualitative trends (more data helps, Skip-gram wins on semantics, etc.)
  hold at this smaller scale too -- see `docs/06-results.md` for exactly
  where the small-scale results agree and disagree with the paper's tables.
- **Negative sampling and frequent-word subsampling** are both from the
  immediate follow-up paper (Mikolov et al., NIPS 2013), not this one, so the
  reproduction itself uses only what Sections 2-3 of *this* paper describe:
  hierarchical softmax over a Huffman tree, `--sample 0`. Both are
  implemented and **off by default** -- subsampling behind `--sample`, to
  measure how much of the remaining gap to reference implementations it
  explains (a lot, see `docs/06-results.md`); negative sampling behind
  `--loss ns`, as a second independent loss implementation cross-validated
  against gensim (`docs/07-negative-sampling.md`).
- **Distributed training (DistBelief).** Section 2.3's multi-replica
  parameter-server setup isn't reproduced; this is single-machine,
  single-GPU minibatch SGD instead (see `docs/04-training.md` for why that's
  still a faithful analog of the paper's own large-scale training approach).

## License

MIT. See [`LICENSE`](LICENSE).
