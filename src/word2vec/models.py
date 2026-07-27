"""CBOW and Skip-gram, both trained with hierarchical softmax (docs/03).

Both models reduce to the same two steps:
  1. produce a single "input vector" per training example (CBOW: the mean of
     the context word vectors; Skip-gram: the center word's own vector)
  2. score that input vector against a target word's Huffman tree path via
     HierarchicalSoftmax

The path/code lookup is precomputed once for the whole vocabulary as padded
tensors (`_PaddedTree`), so a training step is pure tensor indexing + matmul
with no Python-level loop over the batch or over tree depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from word2vec.huffman import HuffmanTree


class _PaddedTree:
    """Vocabulary-wide path/code/mask tensors, one row per word.

    Row i holds word i's root-to-leaf path, right-padded to the tree's max
    depth. `mask` marks which entries in a row are real (1.0) vs padding
    (0.0); `sign` is +1.0 where the correct turn was "right", -1.0 where
    "left", and 0.0 on padding so padded positions can never contribute to a
    dot product regardless of what garbage node id sits in `path_ids` there.
    """

    def __init__(self, tree: HuffmanTree, device: torch.device):
        vocab_size = len(tree.paths)
        max_depth = tree.max_depth()

        path_ids = torch.zeros(vocab_size, max_depth, dtype=torch.long)
        sign = torch.zeros(vocab_size, max_depth, dtype=torch.float32)
        mask = torch.zeros(vocab_size, max_depth, dtype=torch.float32)

        for word_id, (path, code) in enumerate(zip(tree.paths, tree.codes)):
            depth = len(path)
            path_ids[word_id, :depth] = torch.tensor(path, dtype=torch.long)
            sign[word_id, :depth] = torch.tensor(
                [1.0 if b == 1 else -1.0 for b in code], dtype=torch.float32
            )
            mask[word_id, :depth] = 1.0

        self.path_ids = path_ids.to(device)
        self.sign = sign.to(device)
        self.mask = mask.to(device)


class HierarchicalSoftmax(nn.Module):
    def __init__(self, tree: HuffmanTree, embed_dim: int, device: torch.device):
        super().__init__()
        self.node_embeddings = nn.Embedding(tree.num_internal, embed_dim)
        nn.init.zeros_(self.node_embeddings.weight)
        self.tree = _PaddedTree(tree, device)

    def forward(self, input_vectors: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """input_vectors: (batch, D). target_ids: (batch,). Returns mean loss."""
        path_ids = self.tree.path_ids[target_ids]  # (batch, max_depth)
        sign = self.tree.sign[target_ids]  # (batch, max_depth)
        mask = self.tree.mask[target_ids]  # (batch, max_depth)

        node_vecs = self.node_embeddings(path_ids)  # (batch, max_depth, D)
        scores = torch.einsum("bd,bnd->bn", input_vectors, node_vecs)  # (batch, max_depth)

        log_probs = F.logsigmoid(scores * sign) * mask
        per_example_loss = -log_probs.sum(dim=1)
        return per_example_loss.mean()


class _EmbeddingBase(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, tree: HuffmanTree, device: torch.device):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.in_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        self.loss_fn = HierarchicalSoftmax(tree, embed_dim, device)

    def word_vectors(self) -> torch.Tensor:
        return self.in_embeddings.weight.detach()


class CBOWModel(_EmbeddingBase):
    def forward(self, contexts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """contexts: (batch, 2*window) word ids. targets: (batch,) word ids."""
        context_vecs = self.in_embeddings(contexts).mean(dim=1)
        return self.loss_fn(context_vecs, targets)


class SkipGramModel(_EmbeddingBase):
    def forward(self, centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """centers: (batch,) word ids. contexts: (batch,) word ids (one per pair)."""
        center_vecs = self.in_embeddings(centers)
        return self.loss_fn(center_vecs, contexts)
