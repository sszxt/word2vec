"""Run the full experiment suite into results/results.{json,csv} plus an
aggregated results/results_summary.csv.

Every configuration is run over several seeds. Earlier versions of this suite
reported single-seed numbers, which made statements like "these
dimensionalities are within noise of each other" unfalsifiable -- the noise
had never been measured. Seed spread is now reported alongside every mean.

Experiment A (Table 3 style): CBOW vs Skip-gram at fixed dim=100, full
vocabulary, full text8 data, 1 epoch -- which architecture wins on semantic
vs syntactic questions. CBOW appears twice, once per context-gradient rule
(see models.CBOWModel), since that choice turned out to matter more than
anything else in the CBOW column.

Experiment B (Table 2 style): CBOW, vocabulary restricted to the most
frequent 30k words, dimensionality x amount-of-training-data grid, mirroring
the paper's own reduced-scale methodology.

Experiment C: Skip-gram dimensionality sweep at full data.

Experiment D (Table 5 style): epochs x data-amount. The paper claims three
epochs over some amount of data is worth roughly one epoch over three times
as much. The 4M x 2 epochs vs 8M x 1 epoch cell is an exact equal-compute
comparison and is the cleanest test of that claim available here.

Experiment E is explicitly NOT part of the reproduction: frequent-word
subsampling comes from the NIPS 2013 follow-up paper. It is reported
separately so its contribution can be seen without contaminating the
reproduction results above.
"""

import argparse
import csv
import json
import statistics
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

# Reproduction runs use the reference word2vec.c context-gradient rule: the
# paper's text does not specify the backward pass at all, so the authors' own
# implementation is the tiebreaker for what produced the published numbers.
REPRO_CONTEXT_GRAD = "sum"


def run_one(name: str, seed: int, keep_checkpoint: bool, **train_kwargs) -> dict:
    out_path = CKPT_DIR / (f"{name}.pt" if keep_checkpoint else f"_tmp_{name}_s{seed}.pt")
    ckpt = train(
        corpus_path=CORPUS,
        vocab_path=VOCAB,
        seed=seed,
        device=DEVICE,
        out_path=out_path,
        **train_kwargs,
    )

    id2word = ckpt["id2word"]
    vocab = Vocab(
        word2id={w: i for i, w in enumerate(id2word)}, id2word=id2word, counts=[0] * len(id2word)
    )
    questions, stats = load_questions(QUESTIONS, vocab)
    result = evaluate_analogies(ckpt["word_vectors"], questions, device=DEVICE)

    if not keep_checkpoint:
        out_path.unlink(missing_ok=True)

    row = {
        "name": name,
        "seed": seed,
        "arch": ckpt["arch"],
        "dim": ckpt["dim"],
        "window": ckpt["window"],
        "context_grad": ckpt["context_grad"] or "",
        "sample": ckpt["sample"],
        "vocab_size": len(id2word),
        "train_tokens": ckpt["train_tokens"],
        "epochs": ckpt["epochs"],
        "batch_size": ckpt["batch_size"],
        "grad_clip_norm": round(ckpt["grad_clip_norm"], 1),
        "train_seconds": round(ckpt["train_seconds"], 1),
        "questions_evaluated": stats["evaluated"],
        "questions_skipped_oov": stats["skipped_oov"],
        "semantic_acc": round(result.get("semantic", {}).get("accuracy", 0.0) * 100, 2),
        "syntactic_acc": round(result.get("syntactic", {}).get("accuracy", 0.0) * 100, 2),
        "total_acc": round(result.get("total", {}).get("accuracy", 0.0) * 100, 2),
    }
    print(
        f"    seed={seed} tokens={row['train_tokens']:>9,} -> "
        f"sem={row['semantic_acc']:5.1f}% syn={row['syntactic_acc']:5.1f}% "
        f"tot={row['total_acc']:5.1f}%  ({row['train_seconds']}s)"
    )
    return row


def run_config(name: str, seeds: list[int], **train_kwargs) -> list[dict]:
    print(f"\n=== {name} ===")
    rows = [run_one(name, s, keep_checkpoint=(s == seeds[0]), **train_kwargs) for s in seeds]
    totals = [r["total_acc"] for r in rows]
    if len(totals) > 1:
        print(f"    -> total {statistics.mean(totals):.2f}% +/- {statistics.stdev(totals):.2f}")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Collapse per-seed rows into one row per configuration, with spread."""
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)

    summary = []
    for name, group in by_name.items():
        entry = {
            k: group[0][k]
            for k in ("name", "arch", "dim", "context_grad", "sample", "vocab_size",
                      "train_tokens", "epochs")
        }
        entry["n_seeds"] = len(group)
        for metric in ("semantic_acc", "syntactic_acc", "total_acc"):
            values = [r[metric] for r in group]
            entry[f"{metric}_mean"] = round(statistics.mean(values), 2)
            entry[f"{metric}_std"] = round(statistics.stdev(values), 2) if len(values) > 1 else 0.0
        entry["train_seconds_mean"] = round(
            statistics.mean([r["train_seconds"] for r in group]), 1
        )
        summary.append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="seeds per configuration")
    parser.add_argument(
        "--skipgram-seeds",
        type=int,
        default=None,
        help="override seed count for the expensive full-data Skip-gram sweeps "
        "(default: same as --seeds)",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    sg_seeds = list(range(args.skipgram_seeds if args.skipgram_seeds is not None else args.seeds))
    if sg_seeds != seeds:
        print(f"NOTE: Skip-gram sweeps use {len(sg_seeds)} seed(s), other runs use {len(seeds)}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    base = dict(window=None, batch_size=2048, lr=0.025, grad_clip_norm=None, sample=0.0)
    rows = []

    # --- Experiment A: architecture comparison ---
    for context_grad in ("mean", "sum"):
        rows += run_config(
            f"expA_cbow_d100_full_{context_grad}", seeds, arch="cbow", dim=100, epochs=1,
            max_words=None, vocab_size=None, context_grad=context_grad, **base
        )
    rows += run_config(
        "expA_skipgram_d100_full", sg_seeds, arch="skipgram", dim=100, epochs=1,
        max_words=None, vocab_size=None, context_grad="mean", **base
    )

    # --- Experiment B: CBOW, 30k vocab, dim x data-amount grid ---
    for dim in (50, 100, 300):
        for max_words in (2_000_000, 4_000_000, 8_000_000, None):
            tag = f"{max_words // 1_000_000}M" if max_words else "full"
            rows += run_config(
                f"expB_cbow_d{dim}_{tag}", seeds, arch="cbow", dim=dim, epochs=1,
                max_words=max_words, vocab_size=30_000, context_grad=REPRO_CONTEXT_GRAD, **base
            )

    # --- Experiment C: Skip-gram dimensionality sweep ---
    for dim in (50, 100, 300):
        rows += run_config(
            f"expC_skipgram_d{dim}_full", sg_seeds, arch="skipgram", dim=dim, epochs=1,
            max_words=None, vocab_size=None, context_grad="mean", **base
        )

    # --- Experiment D: epochs x data amount ---
    for max_words in (2_000_000, 4_000_000, 8_000_000):
        for epochs in (1, 2, 3):
            tag = f"{max_words // 1_000_000}M"
            rows += run_config(
                f"expD_cbow_d100_{tag}_e{epochs}", seeds, arch="cbow", dim=100, epochs=epochs,
                max_words=max_words, vocab_size=30_000, context_grad=REPRO_CONTEXT_GRAD, **base
            )

    # --- Experiment E: beyond the paper -- frequent-word subsampling ---
    non_sample_base = {k: v for k, v in base.items() if k != "sample"}
    rows += run_config(
        "expE_cbow_d100_full_sample1e-3", seeds, arch="cbow", dim=100, epochs=1,
        max_words=None, vocab_size=None, context_grad=REPRO_CONTEXT_GRAD,
        sample=1e-3, **non_sample_base
    )
    rows += run_config(
        "expE_skipgram_d100_full_sample1e-3", sg_seeds, arch="skipgram", dim=100, epochs=1,
        max_words=None, vocab_size=None, context_grad="mean",
        sample=1e-3, **non_sample_base
    )

    summary = summarize(rows)

    Path("results/results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for path, data in (
        ("results/results.csv", rows),
        ("results/results_summary.csv", summary),
    ):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    print(f"\nsaved {len(rows)} runs ({len(summary)} configurations) to results/")


if __name__ == "__main__":
    main()
