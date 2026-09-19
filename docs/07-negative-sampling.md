# Negative sampling

Implementation: `src/word2vec/models.py::NegativeSampling`. An alternative to
`docs/03-hierarchical-softmax.md`'s tree-structured loss, selected with
`--loss ns` (default is `--loss hs`, hierarchical softmax, which is what the
paper this repository reproduces actually specifies).

**Not from this paper.** Negative sampling is from the immediate follow-up
(Mikolov et al., "Distributed Representations of Words and Phrases and their
Compositionality", NIPS 2013), not from Mikolov et al. 2013 Section 2-3, which
this repository otherwise reproduces. It is implemented and cross-validated
against gensim here as a correctness exercise and a second, independent loss
implementation to compare against hierarchical softmax on the same corpus --
not as part of the reproduction itself, the same stance this repository
already takes on frequent-word subsampling (`docs/04-training.md`).

## The problem it solves, and what it gives up to solve it differently

Hierarchical softmax turns predicting one word out of `V` into walking a
Huffman path and evaluating one sigmoid per node on that path -- `O(log V)`
work, exact in the sense that every internal node's classifier is trained
consistently with the whole tree structure. Negative sampling throws the tree
away entirely and reframes the same prediction as a much simpler question,
answered directly: "is this actually the target word, or is it noise?" --
one sigmoid for the real target, plus one sigmoid for each of a handful of
sampled "noise" words that are not the target. `--negative` (default 5)
controls how many noise words.

Concretely, for input vector `h` and target word `w`, drawing noise words
`n_1..n_k` from a noise distribution `P_n`:

```
loss = -log(sigmoid(h . v_w)) - sum_i log(sigmoid(-h . v_{n_i}))
```

This is `O(k)` per training example instead of hierarchical softmax's
`O(mean path length)` -- at `k=5` versus this vocabulary's frequency-weighted
mean path length of 10.65 (`docs/03`), negative sampling scores roughly half
as many output rows per example. What it gives up: hierarchical softmax's
tree structure means every prediction is implicitly compared against the
*whole* vocabulary's structure (walking the tree at all commits to every turn
not taken); negative sampling only ever compares the target against a small
random handful of alternatives, so its notion of "far from every other word"
is statistical, established over many training steps, rather than baked into
a fixed structure up front.

## The noise distribution

`v_n`, the vector each noise word owns, is a second, separate output
embedding table -- distinct from hierarchical softmax's internal-node
vectors, and also distinct from the input embeddings we actually keep.
Following word2vec.c, it is zero-initialized, same as the internal-node
vectors it replaces.

Noise words are drawn from the unigram distribution raised to the 3/4 power,
`P_n(w) = count(w)^0.75 / sum(count^0.75)`, word2vec.c's own choice rather
than raw unigram frequency. Raising to a power less than 1 flattens the
distribution: rare words get sampled somewhat more often, common words
somewhat less, than their raw frequency share would give them, which keeps
the noise words from being almost entirely "the", "of", "and" every time.

word2vec.c builds this distribution once into a fixed-size (1e8-entry, by
default) lookup table, because single-threaded C has no vectorized weighted
sampling primitive -- the table turns sampling into an O(1) array index.
`torch.multinomial(noise_dist, k, replacement=True)` draws directly from the
exact same distribution and vectorizes over an entire batch at once, so this
implementation needs no such table.

## Cross-validation against gensim

`scripts/reference_gensim.py --arch {cbow,skipgram} --dim 100 --loss ns
--sample 1e-3` trains both implementations with word2vec.c's own
negative-sampling defaults (5 noise words, subsampling at `t=1e-3`) on the
same corpus and vocabulary, then scores both with our own evaluator:

| | semantic | syntactic | total | train time |
|---|---|---|---|---|
| CBOW, ours | 3.7% | 7.4% | 5.9% | 13.6s |
| CBOW, gensim | 4.0% | 8.1% | 6.4% | 2.2s |
| Skip-gram, ours | 9.4% | 20.2% | 15.7% | 85.7s |
| Skip-gram, gensim | 7.8% | 17.4% | 13.4% | 5.8s |

CBOW lands within half a point of gensim; Skip-gram beats it outright. Single
seed each, not the 3-seed measurement `docs/06-results.md` uses for the
reproduction itself -- this is a correctness check, not a claim about which
implementation is "better", and the question it exists to answer is just
whether either one is broken relative to the other. Neither gap is large
enough to suggest that.

gensim trains 6-15x faster here, which is expected and not a correctness
signal: it's multi-threaded, hand-optimized C reached through Cython, against
a single GPU paying Python-loop and kernel-launch overhead per minibatch on a
comparatively small amount of work per step (`docs/06-results.md`'s training-time
section has the same story for hierarchical softmax).

## Practical notes

- Unlike hierarchical softmax, negative sampling has no tree to build, so
  startup is faster and there's no `docs/03`-style path-length measurement to
  make -- every example costs the same `1 + negative` output rows regardless
  of word frequency.
- A noise word that happens to equal the batch's own target is left in rather
  than resampled, matching word2vec.c: at vocab sizes in the tens of
  thousands the collision rate (`negative / V` per example) is too low to
  measurably change the loss.
- `--sample` (frequent-word subsampling, `docs/04-training.md`) is
  independent of `--loss` and combines with either one; word2vec.c's own
  defaults pair negative sampling with subsampling, so `--loss ns --sample
  1e-3` is the closest match to what a real word2vec.c negative-sampling run
  looks like.
