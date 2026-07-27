# Evaluation: the analogy task

Implementation: `src/word2vec/evaluate.py`. Builds on the vector-arithmetic
idea introduced in `docs/02-architectures.md`.

## Test set

The paper defines 5 semantic and 9 syntactic analogy categories (Table 1),
19544 questions total (8869 semantic + 10675 syntactic), and links to a
hosted copy of the test file. We use the same file as distributed with the
original author's released C tool
(`data/questions-words.txt`, fetched by `scripts/download_analogies.py`),
which has identical category names and question count to what the paper
reports:

```
capital-common-countries, capital-world, currency, city-in-state, family        (semantic)
gram1-adjective-to-adverb, gram2-opposite, gram3-comparative, gram4-superlative,
gram5-present-participle, gram6-nationality-adjective, gram7-past-tense,
gram8-plural, gram9-plural-verbs                                                (syntactic)
```

## 3CosAdd

For a question "a is to b as c is to d" (e.g. `Athens Greece Oslo Norway`
means "Athens is to Greece as Oslo is to Norway"), compute

```
X = vec(b) - vec(a) + vec(c)
```

and predict the vocabulary word — excluding a, b, and c themselves — whose
vector has the highest cosine similarity to `X`. Correct only on an exact
match to `d`; a synonym counts as wrong (Section 4.1's stated scoring rule).
`evaluate_analogies` implements this fully batched: all word vectors are
L2-normalized once, and cosine similarity against the whole vocabulary
becomes a single matrix multiply (`target @ normed_vectors.T`) per batch of
questions, rather than a per-question nearest-neighbor search.

One implementation detail: `X` itself is never renormalized before the
similarity matmul. Since every vocabulary vector is already unit-norm,
`cosine(X, v) = (X . v) / |X|` for any `v`, and `|X|` is a per-question
constant that doesn't affect which `v` maximizes the similarity — so skipping
that normalization changes nothing about the predicted answer, only the
(unused) numeric similarity score, and saves an operation per batch.

## Vocabulary coverage on text8

text8 is lowercase-only, whereas the analogy file uses normal capitalization
(`Athens`, not `athens`), so every question is lowercased before vocabulary
lookup — without that step nearly every question would be marked
out-of-vocabulary (text8 has zero uppercase characters). Even after
lowercasing, a question is skipped entirely if any of its 4 words isn't in
the trained vocabulary (a real constraint, not a bug to fix: the paper notes
100% accuracy is unreachable in principle since the model has no morphology
awareness, and an incomplete vocabulary is an additional, expected source of
unanswerable questions at small-corpus scale).

Measured against our text8 vocabulary (`min_count=5`, 71,290 words):

| | count |
|---|---|
| total questions | 19,544 |
| skipped (word not in vocabulary) | 1,717 |
| evaluated | 17,827 |
| coverage | 91.2% |

All accuracy numbers we report are over the 17,827 evaluated questions, same
convention as the paper (Section 4.1: "we evaluate the overall accuracy for
all question types").
