import numpy as np
import torch

from word2vec.dataset import generate_cbow_pairs, generate_skipgram_pairs
from word2vec.huffman import build_huffman_tree
from word2vec.models import CBOWModel, SkipGramModel

torch.manual_seed(0)


def _toy_corpus():
    # 500 tokens drawn from a 20-word vocab with skewed frequency, repeated
    # patterns so there's real structure for the model to fit
    rng = np.random.default_rng(0)
    vocab_size = 20
    weights = 1.0 / np.arange(1, vocab_size + 1)
    weights /= weights.sum()
    tokens = rng.choice(vocab_size, size=2000, p=weights)
    counts = np.bincount(tokens, minlength=vocab_size).tolist()
    return tokens.astype(np.int32), counts, vocab_size


def test_cbow_loss_decreases():
    tokens, counts, vocab_size = _toy_corpus()
    tree = build_huffman_tree(counts)
    device = torch.device("cpu")
    model = CBOWModel(vocab_size, embed_dim=16, tree=tree, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    contexts, targets = generate_cbow_pairs(tokens, window=2)
    contexts_t = torch.tensor(contexts, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = model(contexts_t, targets_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0]


def test_skipgram_loss_decreases():
    tokens, counts, vocab_size = _toy_corpus()
    tree = build_huffman_tree(counts)
    device = torch.device("cpu")
    model = SkipGramModel(vocab_size, embed_dim=16, tree=tree, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    rng = np.random.default_rng(0)
    centers, contexts = generate_skipgram_pairs(tokens, window=3, rng=rng)
    centers_t = torch.tensor(centers, dtype=torch.long)
    contexts_t = torch.tensor(contexts, dtype=torch.long)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = model(centers_t, contexts_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0]


def test_node_embeddings_receive_gradient():
    tokens, counts, vocab_size = _toy_corpus()
    tree = build_huffman_tree(counts)
    device = torch.device("cpu")
    model = CBOWModel(vocab_size, embed_dim=8, tree=tree, device=device)

    contexts, targets = generate_cbow_pairs(tokens, window=2)
    loss = model(torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long))
    loss.backward()

    assert model.in_embeddings.weight.grad is not None
    assert model.loss_fn.node_embeddings.weight.grad is not None
    assert torch.any(model.loss_fn.node_embeddings.weight.grad != 0)
