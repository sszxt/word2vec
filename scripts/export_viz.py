"""Build the self-contained embedding explorer at docs/explorer.html.

Reads trained checkpoints, packs the pieces a browser needs to do vector
arithmetic locally, and injects the result into docs/explorer.template.html.
The output has no external requests -- it opens straight off disk.

Vectors are unit-normalized before quantizing, which is exactly what the
analogy evaluation does anyway (cosine similarity ignores magnitude), so
normalization loses nothing the explorer needs. They are then stored as int8,
which does cost a little precision: components of a 100-dimensional unit
vector are around 0.1, against a quantization step of 1/127, so cosine scores
carry roughly +/-0.02 of error. Near-tied neighbours can swap places. The
explorer says so on screen rather than implying it matches scripts/evaluate.py.

The explorer also searches a truncated vocabulary (--words, default 5000 most
frequent) while scripts/evaluate.py searches all 71,290. Fewer candidates
means fewer distractors, so accuracy measured in the browser runs higher than
the real benchmark. The per-category accuracies shown in the results panel are
therefore computed here over the FULL vocabulary and passed through, rather
than being recomputed in the page.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from word2vec.evaluate import SEMANTIC_CATEGORIES, evaluate_analogies, load_questions
from word2vec.huffman import build_huffman_tree
from word2vec.vocab import Vocab

TEMPLATE = Path("docs/explorer.template.html")
OUT = Path("docs/explorer.html")
PLACEHOLDER = "/*__WORD2VEC_DATA__*/"


def pack_model(path: Path, n_words: int, questions, device) -> dict:
    ckpt = torch.load(path, weights_only=False)
    id2word = ckpt["id2word"]
    vecs = ckpt["word_vectors"].float()

    # Full-vocabulary accuracy, so the reported numbers are the real ones.
    result = evaluate_analogies(vecs, questions, device=device)

    top = vecs[:n_words]
    unit = torch.nn.functional.normalize(top, dim=1)
    quant = torch.clamp(torch.round(unit * 127), -127, 127).to(torch.int8)

    # 2D PCA of the unit vectors, for the map panel.
    centered = (unit - unit.mean(dim=0)).numpy()
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    coords = coords / np.abs(coords).max()

    label = {"cbow": "CBOW", "skipgram": "Skip-gram"}.get(ckpt["arch"], ckpt["arch"])
    if ckpt.get("sample"):
        label += " +sub"
    if ckpt.get("context_grad") == "sum":
        label += ""

    return {
        "label": label,
        "arch": ckpt["arch"],
        "dim": ckpt["dim"],
        "window": ckpt["window"],
        "trainTokens": ckpt["train_tokens"],
        "epochs": ckpt["epochs"],
        "contextGrad": ckpt.get("context_grad") or "",
        "sample": ckpt.get("sample") or 0,
        "vocabSize": len(id2word),
        "vectors": base64.b64encode(quant.numpy().tobytes()).decode("ascii"),
        "coords": [[round(float(x), 4), round(float(y), 4)] for x, y in coords],
        "accuracy": {
            name: round(r["accuracy"] * 100, 2) for name, r in sorted(result.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        nargs="+",
        default=[
            Path("results/checkpoints/expA_cbow_d100_full_sum.pt"),
            Path("results/checkpoints/expA_skipgram_d100_full.pt"),
        ],
    )
    parser.add_argument("--words", type=int, default=5000, help="vocabulary to expose")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions-words.txt"))
    parser.add_argument("--summary", type=Path, default=Path("results/results_summary.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(args.vocab)
    questions, _ = load_questions(args.questions, vocab)

    n = min(args.words, len(vocab))
    words = vocab.id2word[:n]

    models = [pack_model(p, n, questions, device) for p in args.ckpt]

    # Analogy questions whose four words all survive the truncation.
    kept, categories = [], []
    cat_index: dict[str, int] = {}
    for q in questions:
        if max(q.a, q.b, q.c, q.d) >= n:
            continue
        if q.category not in cat_index:
            cat_index[q.category] = len(categories)
            categories.append({"name": q.category, "semantic": q.category in SEMANTIC_CATEGORIES})
        kept.append([q.a, q.b, q.c, q.d, cat_index[q.category]])

    # Huffman depth/code per word, for the hierarchical-softmax panel.
    tree = build_huffman_tree(vocab.counts)
    codes = ["".join(str(b) for b in tree.codes[i]) for i in range(n)]

    payload = {
        "words": words,
        "counts": vocab.counts[:n],
        "totalTokens": vocab.total_tokens,
        "models": models,
        "questions": kept,
        "categories": categories,
        "huffman": {"codes": codes, "maxDepth": tree.max_depth(), "internal": tree.num_internal},
        "summary": args.summary.read_text(encoding="utf-8") if args.summary.exists() else "",
        "fullVocab": len(vocab),
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"{TEMPLATE} is missing the {PLACEHOLDER} placeholder")
    html = template.replace(PLACEHOLDER, json.dumps(payload, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")

    size_mb = len(html.encode("utf-8")) / 1024 / 1024
    print(f"words exposed:  {n:,} of {len(vocab):,}")
    print(f"models:         {', '.join(m['label'] for m in models)}")
    print(f"questions:      {len(kept):,} of {len(questions):,} fully in-vocabulary")
    print(f"categories:     {len(categories)}")
    print(f"wrote {OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
