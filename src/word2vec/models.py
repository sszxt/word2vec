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


class _ScaleGrad(torch.autograd.Function):
    """Identity forward, gradient scaled by `scale` on the way back."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class _EmbeddingBase(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, tree: HuffmanTree, device: torch.device):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.in_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        self.loss_fn = HierarchicalSoftmax(tree, embed_dim, device)

    def word_vectors(self) -> torch.Tensor:
        return self.in_embeddings.weight.detach()


class CBOWModel(_EmbeddingBase):
    """CBOW, with a switch for how gradient is distributed to context words.

    `context_grad="mean"` is the mathematically correct gradient of the
    averaged context vector: each of the C context words receives 1/C of the
    upstream gradient.

    `context_grad="sum"` reproduces what the reference `word2vec.c` actually
    does, which is not the true gradient of its own forward pass: it averages
    the context vectors going forward (`neu1[c] /= cw`) but then adds the
    accumulated gradient `neu1e` to every context word *undivided*. Each
    context word therefore receives C times more update than "mean" gives it
    -- effectively a C-fold larger learning rate on the input embeddings,
    which matters because CBOW already sees far less gradient signal per token
    than Skip-gram does (docs/04-training.md).
    """

    def __init__(self, *args, context_grad: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if context_grad not in ("mean", "sum"):
            raise ValueError(f"context_grad must be 'mean' or 'sum', got {context_grad!r}")
        self.context_grad = context_grad

    def forward(self, contexts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """contexts: (batch, 2*window) word ids. targets: (batch,) word ids."""
        context_embs = self.in_embeddings(contexts)
        if self.context_grad == "sum":
            # Cancel the 1/C that mean() will apply on the backward pass, so
            # each context word ends up with the full upstream gradient.
            context_embs = _ScaleGrad.apply(context_embs, context_embs.shape[1])
        context_vecs = context_embs.mean(dim=1)
        return self.loss_fn(context_vecs, targets)


class SkipGramModel(_EmbeddingBase):
    def forward(self, centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """centers: (batch,) word ids. contexts: (batch,) word ids (one per pair)."""
        center_vecs = self.in_embeddings(centers)
        return self.loss_fn(center_vecs, contexts)
