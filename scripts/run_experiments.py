"""Run the full experiment suite and collect results into results/results.json
and results/results.csv.

Experiment A (Table 3 style): CBOW vs Skip-gram at fixed dim=100, full
vocabulary, full text8 data, 1 epoch -- which architecture wins on semantic
vs syntactic questions.

Experiment B (Table 2 style): CBOW, vocabulary restricted to the most
frequent 30k words, dimensionality x amount-of-training-data grid, mirroring
the paper's own reduced-scale methodology for exploring hyperparameters
before committing to full runs.

Experiment C: Skip-gram dimensionality sweep at full data, to extend
Experiment A's architecture comparison across dimensionalities.
"""

import csv
import json
from pathlib import Path

import torch

from word2vec.evaluate import evaluate_analogies, load_questions
from word2vec.train import train
from word2vec.vocab import Vocab

CORPUS = Path("data/text8")
VOCAB = Path("data/vocab.json")
QUESTIONS = Path("data/questions-words.txt")
CKPT_DIR = Path("results/checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one(name: str, **train_kwargs) -> dict:
    out_path = CKPT_DIR / f"{name}.pt"
    print(f"\n=== {name} ===")
    ckpt = train(
        corpus_path=CORPUS,
        vocab_path=VOCAB,
        seed=0,
        device=DEVICE,
        out_path=out_path,
        **train_kwargs,
    )

    id2word = ckpt["id2word"]
    vocab = Vocab(word2id={w: i for i, w in enumerate(id2word)}, id2word=id2word, counts=[0] * len(id2word))
    questions, stats = load_questions(QUESTIONS, vocab)
    result = evaluate_analogies(ckpt["word_vectors"], questions, device=DEVICE)

    row = {
        "name": name,
        "arch": ckpt["arch"],
        "dim": ckpt["dim"],
        "window": ckpt["window"],
        "vocab_size": len(id2word),
        "train_tokens": ckpt["train_tokens"],
        "epochs": ckpt["epochs"],
        "train_seconds": round(ckpt["train_seconds"], 1),
        "questions_evaluated": stats["evaluated"],
        "questions_skipped_oov": stats["skipped_oov"],
        "semantic_acc": round(result.get("semantic", {}).get("accuracy", 0.0) * 100, 2),
        "syntactic_acc": round(result.get("syntactic", {}).get("accuracy", 0.0) * 100, 2),
        "total_acc": round(result.get("total", {}).get("accuracy", 0.0) * 100, 2),
    }
    print(
        f"    dim={row['dim']:<4} tokens={row['train_tokens']:>9,} vocab={row['vocab_size']:>6,} "
        f"-> semantic={row['semantic_acc']:5.1f}% syntactic={row['syntactic_acc']:5.1f}% "
        f"total={row['total_acc']:5.1f}%  ({row['train_seconds']}s)"
    )
    return row


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- Experiment A: architecture comparison, fixed dim, full data/vocab ---
    for arch in ["cbow", "skipgram"]:
        rows.append(run_one(f"expA_{arch}_d100_full", arch=arch, dim=100, window=None, epochs=1, batch_size=2048, lr=0.025, max_words=None, vocab_size=None))

    # --- Experiment B: CBOW, 30k vocab, dim x data-amount grid ---
    data_amounts = [2_000_000, 4_000_000, 8_000_000, None]  # None = full ~16.7M tokens
    dims = [50, 100, 300]
    for dim in dims:
        for max_words in data_amounts:
            tag = f"{max_words // 1_000_000}M" if max_words else "full"
            rows.append(run_one(f"expB_cbow_d{dim}_{tag}", arch="cbow", dim=dim, window=None, epochs=1, batch_size=2048, lr=0.025, max_words=max_words, vocab_size=30_000))

    # --- Experiment C: Skip-gram dimensionality sweep, full data/vocab ---
    for dim in [50, 100, 300]:
        rows.append(run_one(f"expC_skipgram_d{dim}_full", arch="skipgram", dim=dim, window=None, epochs=1, batch_size=2048, lr=0.025, max_words=None, vocab_size=None))

    out_json = Path("results/results.json")
    out_csv = Path("results/results.csv")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved {len(rows)} results to {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
