"""Huffman binary tree over the vocabulary, for hierarchical softmax.

Every word becomes a leaf. Every internal node holds a trainable vector and
acts as a binary logistic-regression classifier deciding "left or right"
towards the target leaf. Frequent words get short root-to-leaf paths (few
classifier evaluations), rare words get longer ones -- this is what turns an
O(V) softmax into an O(log V)-ish one on average.

Construction: repeatedly merge the two lowest-weight remaining nodes into a
new internal node whose weight is their sum, until one node (the root)
remains. Using a min-heap this is the standard Huffman algorithm; word
frequencies are the initial leaf weights.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class HuffmanTree:
    # paths[word_id] = internal-node indices visited from root to leaf,
    # already remapped to 0..num_internal-1 (rows of the node embedding matrix)
    paths: list[list[int]]
    # codes[word_id][k] = bit taken at paths[word_id][k]; 0 = left, 1 = right
    codes: list[list[int]]
    num_internal: int

    def max_depth(self) -> int:
        return max(len(p) for p in self.paths)

    def mean_depth(self) -> float:
        return sum(len(p) for p in self.paths) / len(self.paths)


def build_huffman_tree(counts: list[int]) -> HuffmanTree:
    num_words = len(counts)
    if num_words < 2:
        raise ValueError("need at least 2 words to build a Huffman tree")

    # node ids: 0..num_words-1 are leaves (word ids); num_words.. are internal
    # nodes, assigned in merge order.
    heap = [(counts[i], i) for i in range(num_words)]
    heapq.heapify(heap)

    total_nodes = 2 * num_words - 1
    parent = [-1] * total_nodes
    bit = [0] * total_nodes

    next_internal = num_words
    while len(heap) > 1:
        w1, i1 = heapq.heappop(heap)
        w2, i2 = heapq.heappop(heap)
        parent[i1] = next_internal
        bit[i1] = 0
        parent[i2] = next_internal
        bit[i2] = 1
        heapq.heappush(heap, (w1 + w2, next_internal))
        next_internal += 1

    num_internal = num_words - 1

    paths: list[list[int]] = []
    codes: list[list[int]] = []
    for word_id in range(num_words):
        path: list[int] = []
        code: list[int] = []
        node = word_id
        while parent[node] != -1:
            p = parent[node]
            path.append(p - num_words)  # -> row index into node embedding table
            code.append(bit[node])
            node = p
        path.reverse()
        code.reverse()
        paths.append(path)
        codes.append(code)

    return HuffmanTree(paths=paths, codes=codes, num_internal=num_internal)
