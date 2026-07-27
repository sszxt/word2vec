# Hierarchical softmax

Implementation: `src/word2vec/huffman.py` (tree construction),
`src/word2vec/models.py` (loss computation). Builds on the `H*V` /
`D*log2(V)` discussion in `docs/01-background.md` and `docs/02-architectures.md`.

## The problem it solves

A standard softmax over a vocabulary of `V` words computes a score for every
single word, normalizes, and only then reads off the probability of the one
word you actually care about. That's `O(V)` work per training example just to
get one number. At `V` = 71,290 (our text8 vocabulary) that's 71,290 dot
products per prediction, for every one of the ~17 million training positions,
for every epoch. Hierarchical softmax avoids ever computing that full
distribution.

## Binary tree over the vocabulary

Every word becomes a leaf of a binary tree. Every internal node is a small
trainable binary classifier: "given the input vector, go left or right?".
Predicting a specific word's probability no longer means normalizing over `V`
outputs — it means walking the fixed root-to-leaf path for that word and
multiplying the probabilities of taking the correct turn at each internal node
along the way.

```
P(word | v_input) = product over each internal node n on the path to word of
                     P(correct turn at n | v_input)
```

Each of those per-node probabilities is an ordinary binary logistic
regression: every internal node `n` owns its own trainable vector `v_n`
(completely separate from the word input/output embeddings), and

```
P(go right at n | v_input) = sigmoid(v_input . v_n)
P(go left  at n | v_input) = 1 - sigmoid(v_input . v_n) = sigmoid(-v_input . v_n)
```

so the whole path probability is a product of `len(path)` sigmoids instead of
one `V`-way normalization. Training maximizes the log-probability of the
correct path, i.e. minimizes

```
loss = - sum over nodes n on the path of log(sigmoid(sign_n * v_input . v_n))
```

where `sign_n = +1` if the correct direction at `n` is "right" and `-1` if
"left". This is exactly what `models.py` computes: for each training pair we
already know the target word's path and the correct bit at every node on that
path (precomputed once by `huffman.py`), so the "correct answer" for every one
of those binary classifiers is known in advance and the whole thing trains
with ordinary backpropagation, no explicit normalization term anywhere.

Only two things are learned per word: its input embedding (looked up when the
word is the CBOW context / Skip-gram center) and, if it happens to appear as
an internal-node ancestor of other words, that internal node's classifier
vector. The internal-node vectors are not word vectors — they belong to the
tree, not the vocabulary — and are discarded after training. The word vectors
we actually keep and evaluate are the input embeddings.

## Why Huffman coding specifically, not just any binary tree

Any binary tree with `V` leaves gives `O(log V)` paths in the worst case. A
*balanced* tree gives every word a path of about `log2(V)`. But word
frequencies are extremely skewed (Zipf's law): a handful of words like "the"
account for a large fraction of all training examples, while most vocabulary
words are rare. Huffman coding builds the tree by weight (frequency), which
guarantees frequent words get the shortest paths and rare words get longer
ones — and because it specifically minimizes weighted path length, it
minimizes the *expected* number of sigmoid evaluations per training step, not
just the worst case.

Measured on our text8 vocabulary (71,290 words, `min_count=5`):

| Metric | Value |
|---|---|
| naive `log2(V)` | 16.12 |
| unweighted mean path length over the vocabulary | 19.44 |
| frequency-weighted mean path length (what training actually pays, on average, per step) | **10.65** |
| path length for `"the"` (most frequent) | 4 |
| path length for `"dolophine"` (least frequent kept word) | 21 |

The unweighted mean (19.44) is worse than naive `log2(V)`, because the long
tail of rare words drags it up. But that number is irrelevant to training
cost — what matters is the frequency-weighted mean (10.65), since that's the
expected cost actually paid per training example, and it beats naive
`log2(V)` by a third. This is the paper's claim in Section 2.1 made concrete:
Huffman coding turns `H*V` into something close to
`H*log2(unigram_perplexity(V))`, which is smaller than `H*log2(V)`.

## Practical notes

- We build the tree once, from the vocabulary's frequency counts, before
  training starts (`huffman.py::build_huffman_tree`). It never changes during
  training.
- `paths[word_id]` and `codes[word_id]` are precomputed for every word so the
  training loop never has to walk the tree structure itself — it just looks up
  a list of `(internal_node_id, target_bit)` pairs per target word.
- Batching a hierarchical-softmax loss is slightly more awkward than a flat
  softmax because different words have different path lengths. We handle this
  in `models.py` by padding paths to the batch's max length and masking the
  padded positions out of the loss.
