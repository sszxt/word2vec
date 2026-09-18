"""Train gensim's word2vec with matching hyperparameters, as a correctness check.

A from-scratch reproduction needs a baseline on the *same* corpus: the paper's
own numbers come from 50-400x more data than text8, so they cannot tell a
correct implementation from a subtly broken one apart at this scale. gensim is
a well-tested port of word2vec.c; training it with matching hyperparameters
and scoring both models with our own evaluator isolates whether a gap comes
from our implementation or just from the small corpus.

Requires the `reference` extra:  pip install -e ".[reference]"

Example:
  python scripts/reference_gensim.py --arch cbow --dim 100 --loss ns
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from word2vec.evaluate import evaluate_analogies, load_questions
from word2vec.train import train
from word2vec.vocab import Vocab, read_tokens

try:
    from gensim.models import Word2Vec as GensimWord2Vec
except ImportError as exc:  # pragma: no cover
    raise SystemExit('gensim is not installed: pip install -e ".[reference]"') from exc

GENSIM_MAX_SENTENCE = 10_000  # gensim silently truncates longer sentences


def train_gensim(
    *,
    tokens: list[str],
    vocab: Vocab,
    arch: str,
    dim: int,
    window: int,
    epochs: int,
    loss: str,
    negative: int,
    sample: float,
    lr: float,
    seed: int,
    workers: int,
) -> tuple[torch.Tensor, float]:
    model = GensimWord2Vec(
        vector_size=dim,
        window=window,
        sg=int(arch == "skipgram"),
        hs=int(loss == "hs"),
        negative=negative if loss == "ns" else 0,
        sample=sample,
        alpha=lr,
        min_alpha=lr * 1e-4,
        cbow_mean=1,
        seed=seed,
        workers=workers,
    )
    # build the vocabulary from our counts so both implementations see the same words
    model.build_vocab_from_freq(dict(zip(vocab.id2word, vocab.counts, strict=True)))
    sentences = [
        tokens[i : i + GENSIM_MAX_SENTENCE] for i in range(0, len(tokens), GENSIM_MAX_SENTENCE)
    ]
    start = time.time()
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    elapsed = time.time() - start

    vectors = torch.from_numpy(np.stack([model.wv[w] for w in vocab.id2word]))
    return vectors, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=Path("data/text8"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions-words.txt"))
    parser.add_argument("--arch", choices=["cbow", "skipgram"], required=True)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--loss", choices=["hs", "ns"], default="hs")
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--sample", type=float, default=0.0)
    parser.add_argument(
        "--context-grad",
        choices=["mean", "sum"],
        default="sum",
        help="ours, CBOW only; 'sum' matches reference word2vec.c (see models.CBOWModel)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    window = args.window if args.window is not None else (4 if args.arch == "cbow" else 10)
    device = torch.device(args.device)

    vocab = Vocab.load(args.vocab)
    tokens = read_tokens(args.corpus)
    questions, stats = load_questions(args.questions, vocab)
    print(f"questions: total={stats['total']} evaluated={stats['evaluated']} skipped_oov={stats['skipped_oov']}")

    print(f"\n=== ours: arch={args.arch} loss={args.loss} dim={args.dim} window={window}")
    tmp_ckpt = Path("results/checkpoints/_reference_ours.pt")
    ckpt = train(
        corpus_path=args.corpus,
        vocab_path=args.vocab,
        arch=args.arch,
        dim=args.dim,
        window=window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_words=None,
        vocab_size=None,
        seed=args.seed,
        device=device,
        out_path=tmp_ckpt,
        context_grad=args.context_grad,
        sample=args.sample,
        loss=args.loss,
        negative=args.negative,
    )
    tmp_ckpt.unlink(missing_ok=True)
    ours = evaluate_analogies(ckpt["word_vectors"], questions, device=device)

    print(f"\n=== gensim: arch={args.arch} loss={args.loss} dim={args.dim} window={window}")
    gensim_vectors, gensim_seconds = train_gensim(
        tokens=tokens,
        vocab=vocab,
        arch=args.arch,
        dim=args.dim,
        window=window,
        epochs=args.epochs,
        loss=args.loss,
        negative=args.negative,
        sample=args.sample,
        lr=args.lr,
        seed=args.seed,
        workers=args.workers,
    )
    gensim_scores = evaluate_analogies(gensim_vectors, questions, device=torch.device("cpu"))
    print(f"    trained in {gensim_seconds:.1f}s")

    print()
    print(f"{'':10s} {'semantic':>10s} {'syntactic':>10s} {'total':>10s}")
    for label, scores, seconds in (
        ("ours", ours, ckpt["train_seconds"]),
        ("gensim", gensim_scores, gensim_seconds),
    ):
        cells = "".join(f"{scores.get(k, {}).get('accuracy', 0.0) * 100:9.1f}% " for k in ("semantic", "syntactic", "total"))
        print(f"{label:10s} {cells} ({seconds:.1f}s)")


if __name__ == "__main__":
    main()
