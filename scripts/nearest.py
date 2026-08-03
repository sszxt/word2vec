"""Nearest neighbours by cosine similarity -- qualitative inspection of trained vectors.

The analogy task (scripts/evaluate.py) reduces a model to a single accuracy
number, which says nothing about *how* a model is wrong. Neighbour lists do:
a model that is merely undertrained returns plausible-but-loose neighbours
(king -> prince, throne, duke), while a model with a broken update rule
returns unrelated high-frequency words, because its vectors never moved far
from their random initialization.

Pass several checkpoints to print their neighbour lists side by side.
"""

import argparse
from pathlib import Path

import torch


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, weights_only=False)
    vecs = ckpt["word_vectors"].float()
    return {
        "label": path.stem,
        "arch": ckpt["arch"],
        "dim": ckpt["dim"],
        "id2word": ckpt["id2word"],
        "word2id": {w: i for i, w in enumerate(ckpt["id2word"])},
        "normed": torch.nn.functional.normalize(vecs, dim=1),
    }


def neighbours(model: dict, word: str, topk: int) -> list[tuple[str, float]] | None:
    """Top-k most cosine-similar words, excluding the query word itself."""
    word_id = model["word2id"].get(word)
    if word_id is None:
        return None

    sims = model["normed"] @ model["normed"][word_id]
    sims[word_id] = torch.finfo(sims.dtype).min
    scores, ids = sims.topk(topk)
    return [(model["id2word"][i], s) for i, s in zip(ids.tolist(), scores.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="+", help="query words (lowercase, as in text8)")
    parser.add_argument(
        "--ckpt",
        type=Path,
        nargs="+",
        default=[
            Path("results/checkpoints/expA_cbow_d100_full.pt"),
            Path("results/checkpoints/expA_skipgram_d100_full.pt"),
        ],
        help="one or more checkpoints; multiple are shown side by side",
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    models = [load_checkpoint(p) for p in args.ckpt]
    col = 30

    print(" | ".join(f"{m['label']} ({m['arch']} d{m['dim']})".ljust(col) for m in models))
    print("-" * (col * len(models) + 3 * (len(models) - 1)))

    for word in args.words:
        print(f"\n=== {word} ===")
        results = [neighbours(m, word, args.topk) for m in models]

        for model, result in zip(models, results):
            if result is None:
                print(f"  [{model['label']}] '{word}' not in vocabulary ({len(model['id2word']):,} words)")

        if all(r is None for r in results):
            continue

        for rank in range(args.topk):
            cells = []
            for result in results:
                if result is None or rank >= len(result):
                    cells.append(" " * col)
                else:
                    neighbour, score = result[rank]
                    cells.append(f"  {rank + 1:2d}. {neighbour:<18s} {score:.3f}".ljust(col))
            print(" | ".join(cells))


if __name__ == "__main__":
    main()
