"""Training pair generation for CBOW and Skip-gram, as described in Section 3.

Both generators are fully vectorized with numpy: at text8 scale (~17M token
ids) a plain Python loop over every position would dominate wall-clock time,
so we express both generation procedures as array operations instead. Neither
function crosses the boundaries of the corpus array (positions closer to the
start/end than `window` are skipped), which discards a negligible fraction of
positions relative to a 17M-token corpus.
"""

from __future__ import annotations

import numpy as np


def subsample_frequent(
    token_ids: np.ndarray, sample: float, rng: np.random.Generator
) -> np.ndarray:
    """Randomly discard frequent words, keeping the corpus in order.

    NOT from the paper this repository reproduces. Subsampling comes from the
    immediate follow-up (Mikolov et al., "Distributed Representations of Words
    and Phrases and their Compositionality", NIPS 2013) and is off by default;
    it is included only to measure how much of the remaining gap to reference
    implementations it accounts for.

    Uses the keep-probability from the reference `word2vec.c`:

        P_keep(w) = sqrt(t / f(w)) + t / f(w)

    where f(w) is the word's corpus frequency and `t` is `sample`. The paper's
    own text states the simpler `P_discard = 1 - sqrt(t/f)`, i.e. without the
    trailing `t/f` term; the code adds it, which keeps slightly more of the
    mid-frequency words. Words rarer than `t` are always kept.

    Discarding happens before pair generation, so removing a token also pulls
    distant words into each other's windows -- the widened context is part of
    the effect, not a side effect.
    """
    if sample <= 0:
        return token_ids

    counts = np.bincount(token_ids)
    freq = counts / counts.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(freq > 0, sample / freq, np.inf)
    keep_prob = np.minimum(np.sqrt(ratio) + ratio, 1.0)

    keep = rng.random(len(token_ids)) < keep_prob[token_ids]
    return token_ids[keep]


def generate_cbow_pairs(token_ids: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed symmetric window, matching the paper's "N history + N future" CBOW.

    Returns (contexts, targets):
      contexts: int32 array of shape (num_positions, 2*window)
      targets:  int32 array of shape (num_positions,)
    """
    n = len(token_ids)
    positions = np.arange(window, n - window)
    targets = token_ids[positions]

    offsets = np.array([o for o in range(-window, window + 1) if o != 0])
    contexts = token_ids[positions[:, None] + offsets[None, :]]
    return contexts.astype(np.int32), targets.astype(np.int32)


def generate_skipgram_pairs(
    token_ids: np.ndarray, window: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic window as described in Section 3.2: for each center word, sample
    R uniformly from 1..window, then pair it with the R words before and R
    words after it. Equivalent to including offset k (1 <= |k| <= window) with
    probability (window - |k| + 1) / window, i.e. nearby words are paired more
    often than distant ones without needing an explicit distance weighting.

    Returns (centers, contexts), both int32 arrays of shape (num_pairs,).
    """
    n = len(token_ids)
    positions = np.arange(window, n - window)
    centers_at = token_ids[positions]

    r = rng.integers(1, window + 1, size=len(positions))

    center_chunks = []
    context_chunks = []
    for k in range(-window, window + 1):
        if k == 0:
            continue
        mask = np.abs(k) <= r
        context_chunks.append(token_ids[positions[mask] + k])
        center_chunks.append(centers_at[mask])

    centers = np.concatenate(center_chunks).astype(np.int32)
    contexts = np.concatenate(context_chunks).astype(np.int32)
    return centers, contexts


def iterate_batches(
    *arrays: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
):
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays)

    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield tuple(a[idx] for a in arrays)
