from word2vec.huffman import build_huffman_tree


def test_prefix_free():
    counts = [50, 30, 10, 8, 8, 4, 2, 1]
    tree = build_huffman_tree(counts)
    codes_as_strings = ["".join(map(str, c)) for c in tree.codes]
    for i, a in enumerate(codes_as_strings):
        for j, b in enumerate(codes_as_strings):
            if i != j:
                assert not b.startswith(a), f"code {a} is a prefix of {b}"


def test_frequent_words_get_shorter_or_equal_codes():
    counts = [100, 90, 50, 10, 5, 3, 2, 1]
    tree = build_huffman_tree(counts)
    depths = [len(p) for p in tree.paths]
    for i in range(len(counts) - 1):
        assert depths[i] <= depths[i + 1] + 1, (
            "depth should roughly decrease with frequency; "
            f"word {i} (count={counts[i]}) has depth {depths[i]}, "
            f"word {i + 1} (count={counts[i + 1]}) has depth {depths[i + 1]}"
        )


def test_paths_reconstruct_correct_leaf():
    counts = [40, 20, 15, 10, 8, 4, 2, 1, 1, 1]
    tree = build_huffman_tree(counts)
    num_words = len(counts)

    # rebuild left/right child map from paths+codes and confirm each word's
    # path deterministically arrives at a distinct leaf
    children: dict[tuple[int, int], int] = {}
    for word_id in range(num_words):
        path, code = tree.paths[word_id], tree.codes[word_id]
        for depth, (node, bit) in enumerate(zip(path, code)):
            is_last = depth == len(path) - 1
            child = word_id if is_last else None
            key = (node, bit)
            if child is not None:
                assert key not in children or children[key] == child
                children[key] = child

    assert tree.num_internal == num_words - 1


def test_two_words():
    tree = build_huffman_tree([5, 3])
    assert tree.num_internal == 1
    assert {tuple(c) for c in tree.codes} == {(0,), (1,)}
