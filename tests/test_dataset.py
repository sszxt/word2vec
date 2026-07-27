import numpy as np

from word2vec.dataset import generate_cbow_pairs, generate_skipgram_pairs, iterate_batches


def test_cbow_pairs_match_manual_window():
    # tokens: 0 1 2 3 4 5 6, window=2 -> valid targets are positions 2..4 (values 2,3,4)
    tokens = np.arange(7)
    contexts, targets = generate_cbow_pairs(tokens, window=2)

    assert list(targets) == [2, 3, 4]
    # context of target=2 (position 2) is [0,1,3,4]
    assert list(contexts[0]) == [0, 1, 3, 4]
    # context of target=3 (position 3) is [1,2,4,5]
    assert list(contexts[1]) == [1, 2, 4, 5]
    assert contexts.shape == (3, 4)


def test_skipgram_pairs_respect_window_and_symmetry():
    tokens = np.arange(50)
    rng = np.random.default_rng(0)
    centers, contexts = generate_skipgram_pairs(tokens, window=5, rng=rng)

    assert len(centers) == len(contexts)
    diffs = contexts.astype(int) - centers.astype(int)
    assert np.all(np.abs(diffs) >= 1)
    assert np.all(np.abs(diffs) <= 5)

    # every position contributes at least the immediate neighbors (R >= 1 always)
    positions = np.arange(5, len(tokens) - 5)
    assert len(centers) >= 2 * len(positions)


def test_skipgram_more_pairs_with_larger_window():
    tokens = np.arange(2000)
    rng = np.random.default_rng(0)
    _, small = generate_skipgram_pairs(tokens, window=2, rng=rng)
    _, large = generate_skipgram_pairs(tokens, window=8, rng=rng)
    assert len(large) > len(small)


def test_iterate_batches_covers_all_examples_without_repeats():
    a = np.arange(97)
    b = np.arange(97) * 10
    rng = np.random.default_rng(1)

    seen_a = []
    for batch_a, batch_b in iterate_batches(a, b, batch_size=16, shuffle=True, rng=rng):
        assert np.array_equal(batch_a * 10, batch_b)
        seen_a.extend(batch_a.tolist())

    assert sorted(seen_a) == list(range(97))
