"""Training loop: plain SGD with a linearly decaying learning rate, matching
Section 4.2 ("we chose starting learning rate 0.025 and decreased it linearly,
so that it approaches zero at the end of the last training epoch").

Batching: the paper's own large-scale runs use "mini-batch asynchronous
gradient descent" (Section 2.3), so training in shuffled minibatches here
rather than one example at a time is consistent with the paper's own
methodology, not a deviation from it -- it's simply the single-machine
equivalent of what DistBelief did across machines.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from word2vec._progress import LogProgress
from word2vec.dataset import generate_cbow_pairs, generate_skipgram_pairs, iterate_batches
from word2vec.huffman import build_huffman_tree
from word2vec.models import CBOWModel, SkipGramModel
from word2vec.vocab import Vocab, read_tokens


# Gradient norm is proportional to batch size here, because the loss is
# scaled by len(batch) before backward() (see the comment in the training
# loop). A fixed clip threshold is therefore only meaningful at the batch
# size it was calibrated on. Measured pre-clip norms for CBOW/mean, 300
# steps: batch 512 -> median 24, batch 2048 -> median 75, batch 8192 ->
# median 407. The original fixed 150 was calibrated at batch 2048, where it
# clips 0% of batches -- but at batch 8192 it clips 98% of them, silently
# throttling training for anyone who changes the batch size.
#
# 0.073 = 150/2048 reproduces the original calibration exactly at batch 2048
# and scales it correctly elsewhere.
_CLIP_PER_EXAMPLE = 150.0 / 2048.0

# context_grad="sum" makes every context word receive C times more gradient,
# which raises the total norm by roughly 4.4x at batch 2048 (measured) --
# less than C because the output-node embeddings are unaffected.
_SUM_GRAD_FACTOR = 4.5


def auto_grad_clip(batch_size: int, arch: str, context_grad: str) -> float:
    """Clip threshold scaled to the batch size it will actually be used at."""
    clip = _CLIP_PER_EXAMPLE * batch_size
    if arch == "cbow" and context_grad == "sum":
        clip *= _SUM_GRAD_FACTOR
    return clip


def restrict_vocab(vocab: Vocab, vocab_size: int) -> Vocab:
    id2word = vocab.id2word[:vocab_size]
    counts = vocab.counts[:vocab_size]
    word2id = {w: i for i, w in enumerate(id2word)}
    return Vocab(word2id=word2id, id2word=id2word, counts=counts)


def train(
    *,
    corpus_path: Path,
    vocab_path: Path,
    arch: str,
    dim: int,
    window: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    max_words: int | None,
    vocab_size: int | None,
    seed: int,
    device: torch.device,
    out_path: Path,
    grad_clip_norm: float | None = None,
    context_grad: str = "mean",
) -> dict:
    if window is None:
        window = 4 if arch == "cbow" else 10

    if grad_clip_norm is None:
        grad_clip_norm = auto_grad_clip(batch_size, arch, context_grad)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    vocab = Vocab.load(vocab_path)
    if vocab_size is not None:
        vocab = restrict_vocab(vocab, vocab_size)

    tokens = read_tokens(corpus_path)
    if max_words is not None:
        tokens = tokens[:max_words]
    token_ids = np.array(vocab.encode(tokens), dtype=np.int32)
    print(
        f"training tokens: {len(token_ids):,}  vocab: {len(vocab):,}  arch: {arch}  dim: {dim}"
        f"  batch: {batch_size}  grad_clip: {grad_clip_norm:.0f}"
    )

    tree = build_huffman_tree(vocab.counts)
    if arch == "cbow":
        model = CBOWModel(len(vocab), dim, tree, device, context_grad=context_grad)
    else:
        model = SkipGramModel(len(vocab), dim, tree, device)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def make_epoch_pairs():
        if arch == "cbow":
            return generate_cbow_pairs(token_ids, window)
        return generate_skipgram_pairs(token_ids, window, rng)

    a_arr, b_arr = make_epoch_pairs()
    steps_per_epoch = math.ceil(len(a_arr) / batch_size)
    total_steps = steps_per_epoch * epochs

    global_step = 0
    t_start = time.time()
    for epoch in range(epochs):
        if epoch > 0:
            a_arr, b_arr = make_epoch_pairs()

        batch_iter = iterate_batches(a_arr, b_arr, batch_size=batch_size, shuffle=True, rng=rng)
        progress = LogProgress(steps_per_epoch, desc=f"epoch {epoch + 1}/{epochs}")

        recent_losses: deque[float] = deque(maxlen=200)
        for i, (a, b) in enumerate(batch_iter):
            frac = global_step / total_steps
            current_lr = max(lr * (1 - frac), lr * 1e-4)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            a_t = torch.from_numpy(a.astype(np.int64)).to(device, non_blocking=True)
            b_t = torch.from_numpy(b.astype(np.int64)).to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(a_t, b_t)  # mean per-example loss
            # The paper's SGD updates once per single training example
            # (batch size 1). We batch for GPU throughput, but model()
            # returns the *mean* loss over the batch, so backpropagating
            # it directly would shrink the effective gradient by a factor
            # of batch_size relative to the paper's per-example updates --
            # one epoch of batches would barely move the weights. Scaling
            # by batch size before backward() restores gradients of the
            # same magnitude the paper's per-example SGD would produce,
            # while `--lr` keeps meaning "per-example learning rate".
            #
            # Caveat: plain SGD with no momentum/Adam-style normalization can
            # enter a runaway feedback loop -- an unlucky large step pushes a
            # vector far enough that the next batch's gradient through it is
            # even larger, compounding exponentially. Measured directly:
            # Skip-gram's gradient norm was a normal ~100-150 for its first
            # ~500 steps, then reached ~2 billion by step ~2000 and produced
            # NaN word vectors by the end of one epoch. CBOW's norms sit in
            # the same ~75-150 range but its epoch is 11x shorter (fewer
            # pairs), so in these runs it happened not to hit a destabilizing
            # sequence of batches. Clipping the global gradient norm to just
            # above the observed healthy range (150) caps the worst-case
            # step, breaking the feedback loop before it can compound,
            # without measurably affecting CBOW (verified: identical 4.2%
            # total accuracy clipped vs unclipped).
            (loss * len(a)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            recent_losses.append(loss.item())
            global_step += 1
            windowed = sum(recent_losses) / len(recent_losses)
            progress.update(i + 1, loss=f"{windowed:.4f}", lr=f"{current_lr:.5f}")

    elapsed = time.time() - t_start
    print(f"training done in {elapsed / 60:.1f} min ({global_step} steps)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "word_vectors": model.word_vectors().cpu(),
        "id2word": vocab.id2word,
        "arch": arch,
        "dim": dim,
        "window": window,
        "train_tokens": len(token_ids),
        "epochs": epochs,
        "train_seconds": elapsed,
        "context_grad": context_grad if arch == "cbow" else None,
        "batch_size": batch_size,
        "grad_clip_norm": grad_clip_norm,
    }
    torch.save(checkpoint, out_path)
    print(f"saved to {out_path}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/text8"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--arch", choices=["cbow", "skipgram"], required=True)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="max global gradient norm; default scales with batch size (see auto_grad_clip)",
    )
    parser.add_argument(
        "--context-grad",
        choices=["mean", "sum"],
        default="mean",
        help="CBOW only: how gradient reaches context words. 'mean' is the true "
        "gradient; 'sum' reproduces reference word2vec.c (see models.CBOWModel)",
    )
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        vocab_path=args.vocab,
        arch=args.arch,
        dim=args.dim,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_words=args.max_words,
        vocab_size=args.vocab_size,
        seed=args.seed,
        device=torch.device(args.device),
        out_path=args.out,
        grad_clip_norm=args.grad_clip,
        context_grad=args.context_grad,
    )


if __name__ == "__main__":
    main()
