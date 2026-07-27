# Background: why word2vec exists

Source: Mikolov, Chen, Corrado, Dean. "Efficient Estimation of Word Representations
in Vector Space." arXiv:1301.3781v3, 2013. See `paper/1301.3781v3.pdf`.

## The problem with atomic word representations

Before this line of work, most NLP systems treated words as atomic symbols: an index
into a vocabulary, e.g. `dog = 4592`. Under this representation there is no notion
of similarity between words — `dog` and `cat` are as unrelated as `dog` and `bicycle`,
because indices carry no information about meaning. N-gram language models are the
classic example: they scale extremely well (trillions of words of training data are
usable) precisely because they make no attempt to generalize across similar words.

That scaling stops helping in domains where training data is inherently limited —
the paper cites speech recognition (millions of words of transcribed speech) and
machine translation (a few billion words for many language pairs) as examples. When
you cannot simply add more data, you need a model that generalizes better per word
of training data, and that means giving the model some notion that `dog` and `cat`
are related.

## Distributed representations

The fix is to represent each word as a dense, low-dimensional real-valued vector
(e.g. 50-1000 dimensions) instead of a one-hot index into a vocabulary of possibly
millions of words. If trained well, words used in similar contexts end up with
similar vectors, which lets the model share statistical strength across related
words instead of treating every word as an isolated symbol. This idea predates the
paper by decades (Hinton et al., 1986), but became practical for large vocabularies
and large corpora only with neural network language models in the 2000s-2010s.

## Prior architectures the paper positions itself against

The paper frames its contribution against two earlier neural language model
architectures, and defines a shared cost model to compare them.

### Training cost model

For every architecture in the paper, training cost is modeled as

```
O = E * T * Q
```

where `E` is the number of training epochs (typically 3-50), `T` is the number of
words in the training corpus (up to ~1 billion in the paper), and `Q` is the cost of
processing *one* training example, which depends on the architecture. Because `E`
and `T` are shared knobs, the entire comparison between architectures reduces to
comparing their per-example cost `Q` — this is the number to watch in every section
below.

### Feedforward NNLM (Bengio et al., 2003)

Input: the previous `N` words, one-hot encoded over a vocabulary of size `V`.
Layers: input -> projection -> hidden (nonlinear, `tanh`) -> output (softmax over
`V`).

```
Q = N*D + N*D*H + H*V
```

- `N*D`: build the projection layer by concatenating N word vectors of dimension D.
  Cheap — only N vectors are looked up.
- `N*D*H`: the projection layer (dense, size `N*D`) is fully connected to a hidden
  layer of size `H`. This is a dense matrix multiply and dominates cost for
  practical `N`, `D`, `H`.
- `H*V`: the hidden layer feeds a softmax over the entire vocabulary to produce a
  probability distribution over the next word. Naively this is proportional to `V`,
  which is catastrophic once `V` reaches hundreds of thousands to millions of words.

The `H*V` term can be reduced using **hierarchical softmax**: instead of a flat
softmax over `V` outputs, represent the vocabulary as leaves of a binary tree and
turn "pick the right word out of V" into "make ~log2(V) binary decisions to walk
from the root to the right leaf". A Huffman tree (built from word frequencies, so
frequent words get short codes) reduces the expected number of decisions further,
to roughly `log2(unigram_perplexity(V))` rather than `log2(V)`. We implement this
mechanism ourselves in `docs/03-hierarchical-softmax.md` — it is the one piece of
machinery shared by every model in this repository.

Even with that fix, `N*D*H` remains: NNLM pays for a dense, nonlinear hidden layer
on every training example, which is the cost this paper's models set out to
eliminate entirely.

### RNNLM (Mikolov et al., 2010)

Recurrent architecture: no fixed context window `N` and no projection layer. Instead
a hidden layer is connected to itself through a recurrent weight matrix, letting it
in principle retain information from arbitrarily far back in the sequence.

```
Q = H*H + H*V
```

`H*V` is reducible the same way as above via hierarchical softmax, leaving `H*H` —
the recurrent matrix multiply — as the dominant, unavoidable cost per example.

### The shared bottleneck

Both prior architectures pay for a dense, nonlinear hidden layer on every single
training example (`N*D*H` for NNLM, `H*H` for RNNLM). That hidden layer is exactly
what gives these models their representational power over simpler n-gram models —
but it's also what makes them expensive to scale to billion-word corpora.

The paper's core bet, developed in `docs/02-architectures.md`, is: drop the
nonlinear hidden layer entirely, accept a less expressive per-example model, and
spend the compute you saved on training over far more data instead. Table 5 in the
paper is the empirical justification — a simpler model trained for one epoch on 2x
the data beats the same model trained for three epochs on the smaller set, at lower
total cost.
