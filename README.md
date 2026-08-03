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
paper's ideas in the order they're implemented, including two real training
bugs that came up during this reproduction and how they were diagnosed and
fixed.

## Reading order

| Doc | Covers |
|---|---|
| [`docs/01-background.md`](docs/01-background.md) | Why atomic word representations fail; NNLM/RNNLM and their cost |
| [`docs/02-architectures.md`](docs/02-architectures.md) | CBOW and Skip-gram: what they predict and why |
| [`docs/03-hierarchical-softmax.md`](docs/03-hierarchical-softmax.md) | Huffman-tree softmax: the training-cost trick shared by both models |
| [`docs/04-training.md`](docs/04-training.md) | SGD schedule, batching, and two real bugs hit while reproducing this |
| [`docs/05-evaluation-task.md`](docs/05-evaluation-task.md) | The analogy task (3CosAdd) and vocabulary coverage |
| [`docs/06-results.md`](docs/06-results.md) | Full results, compared point-by-point against the paper's tables |

## Results at a glance

CBOW vs Skip-gram, dim=100, full text8 (16.7M tokens), 1 epoch:

| Architecture | Semantic | Syntactic | Total |
|---|---|---|---|
| CBOW | 3.7% | 4.5% | 4.2% |
| Skip-gram | 14.6% | 20.3% | 17.9% |

This reproduces the paper's central qualitative finding -- Skip-gram beats
CBOW by a wide margin on semantic analogies -- at roughly 1/50th of the
paper's smallest data size. See [`docs/06-results.md`](docs/06-results.md)
for the full dimensionality/data-amount sweep and a point-by-point
comparison against the paper's own tables, including where our small-scale
results diverge from theirs and why.

## Repository layout

```
paper/            the source paper (PDF)
docs/             concept notes, paired with the implementation, read in order
src/word2vec/     library code
  vocab.py           vocabulary + frequency counting
  huffman.py         Huffman tree for hierarchical softmax
  dataset.py         CBOW / Skip-gram training pair generation
  models.py          CBOW and Skip-gram models, hierarchical softmax loss
  train.py           training loop (SGD, linear LR decay, gradient clipping)
  evaluate.py        analogy evaluation (3CosAdd)
scripts/          CLI entry points (download data, train, evaluate, run the full experiment suite)
tests/            unit tests for every module above
results/          results.json / results.csv from the experiment suite (checkpoints are gitignored)
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

## Evaluating a checkpoint

```
python scripts/evaluate.py results/cbow.pt
```

Prints per-category accuracy plus semantic / syntactic / total rollups on
the analogy task.

## Reproducing the full experiment suite

```
python scripts/run_experiments.py
```

Runs the architecture comparison, the CBOW dimensionality/data-amount sweep,
and the Skip-gram dimensionality sweep described in
[`docs/06-results.md`](docs/06-results.md), and writes
`results/results.json` / `results/results.csv`. Takes roughly 15-20 minutes
on a single consumer GPU (text8 is small; Skip-gram's larger pair count is
the dominant cost -- see the training-time breakdown in
[`docs/06-results.md`](docs/06-results.md)).

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
- **Negative sampling.** From the immediate follow-up paper (Mikolov et al.,
  NIPS 2013), not this one. The reproduction implements only what Sections
  2-3 of *this* paper describe: hierarchical softmax over a Huffman tree.
- **Frequent-word subsampling** is also from that follow-up paper, so it is
  **off by default** and excluded from every reproduction result. It is
  implemented behind `--sample` purely to measure how much of the remaining
  gap to reference implementations it explains -- which turns out to be a
  lot. See `docs/06-results.md`.
- **Distributed training (DistBelief).** Section 2.3's multi-replica
  parameter-server setup isn't reproduced; this is single-machine,
  single-GPU minibatch SGD instead (see `docs/04-training.md` for why that's
  still a faithful analog of the paper's own large-scale training approach).

## License

MIT. See [`LICENSE`](LICENSE).
