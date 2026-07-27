# The two architectures: CBOW and Skip-gram

Builds on `docs/01-background.md`. Both models remove the nonlinear hidden layer
that made NNLM/RNNLM expensive, leaving a "log-linear" model: a linear projection
followed directly by a (hierarchical) softmax, with no nonlinearity in between.
Removing the hidden layer is what makes billion-word training tractable — it is
also why these models are less individually expressive than NNLM/RNNLM per
training example. The paper's wager is that far more training data more than
compensates.

Both architectures are, at their core, a *fill-in-the-blank* task read off a
sliding window over raw unlabeled text — no manual labeling is needed. A word
vector is the byproduct of getting good at that task: to succeed at it, the model
is forced to place words that appear in similar contexts near each other in vector
space. The vectors are what we keep; the fill-in-the-blank predictions themselves
are discarded after training.

## Continuous Bag-of-Words (CBOW)

Task: given the words surrounding a position (some window of history and future
words), predict the word that actually occurs there.

```
context:  w(t-2)  w(t-1)  [ ? ]  w(t+1)  w(t+2)
target:                   w(t)
```

Mechanics:
1. Look up the vector for each of the `N` context words (shared embedding matrix,
   same weights regardless of position — a word's vector doesn't depend on whether
   it appeared two positions before or one position after the target).
2. Average (sum, in the paper's phrasing) those `N` vectors into a single
   projection vector. This is the "bag-of-words" part: order within the context
   window is thrown away, only *which* words are present matters.
3. Feed that single averaged vector into a (hierarchical) softmax over the
   vocabulary, and train to maximize the probability of the true center word.

```
Q = N*D + D*log2(V)
```

`N*D` to look up and average `N` context vectors of dimension `D`; `D*log2(V)` for
one hierarchical-softmax prediction from that single averaged vector. No hidden
layer, no `H` term anywhere — this is the entire cost.

The paper's best CBOW configuration uses 4 history words and 4 future words
(`N=8` total).

## Continuous Skip-gram

The mirror image of CBOW. Task: given a single word, predict each of the words
surrounding it.

```
context:  w(t-2)  w(t-1)  w(t+1)  w(t+2)
input:                     w(t)
```

Mechanics:
1. Look up the vector for the single current word `w(t)`.
2. For each position within a window of size `C` around `w(t)`, independently
   predict that context word via a (hierarchical) softmax fed only by `w(t)`'s
   vector.

Because more distant context words are usually less related to `w(t)` than nearby
ones, the paper does not use a fixed-size window for every example. Instead, for
each training word it samples a random window size `R` uniformly from `1..C`, then
uses only the `R` words before and `R` words after as training pairs. This gives
distant words a systematically lower chance of ever being sampled, without having
to hand-tune a distance-based weighting scheme. The paper uses `C = 10`.

```
Q = C * (D + D*log2(V))
```

For each of up to `2*R` context words in the window (bounded by `C`), pay `D` to
read the input word's vector plus `D*log2(V)` for one hierarchical-softmax
prediction. Skip-gram does strictly more softmax predictions per input word than
CBOW does per context window, which is why the paper's Table 5 shows Skip-gram
taking roughly 3x longer to train than CBOW on the same data.

## Why bother with two architectures instead of one

Table 3 in the paper (reproduced in our results once we run the equivalent
experiment) shows they are not interchangeable:

| Architecture | Semantic accuracy | Syntactic accuracy |
|---|---|---|
| CBOW    | 24% | 64% |
| Skip-gram | 55% | 59% |

CBOW is better at syntactic regularities (`quick -> quickest` style relationships,
verb tense, plurals), Skip-gram is much better at semantic ones (`France -> Paris`
style relationships, capitals, currencies). Intuitively: CBOW smooths several
context words into one averaged prediction target, which acts like extra training
signal for word *form* patterns that show up consistently across many contexts,
while Skip-gram treats every individual context word as its own training example,
giving rarer/more specific semantic co-occurrences more chances to be learned
individually rather than averaged away.

## What "quality" means here, concretely

The paper's central empirical claim is that simple vector arithmetic on the
resulting word vectors recovers relationships, e.g.

```
vector("king") - vector("man") + vector("woman") ~= vector("queen")
vector("Paris") - vector("France") + vector("Italy") ~= vector("Rome")
vector("biggest") - vector("big") + vector("small") ~= vector("smallest")
```

To find the answer word given the other three, compute the target vector `X` from
the arithmetic above, then search the vocabulary for the word whose vector has the
highest cosine similarity to `X` (excluding the three input words). This procedure
— called 3CosAdd in later literature — is exactly the analogy task we implement in
`docs/04-evaluation-task.md` and use to score every model we train.
