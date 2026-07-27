import argparse
from pathlib import Path

import torch

from word2vec.evaluate import evaluate_analogies, load_questions
from word2vec.vocab import Vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("data/questions-words.txt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    id2word = ckpt["id2word"]
    vocab = Vocab(word2id={w: i for i, w in enumerate(id2word)}, id2word=id2word, counts=[0] * len(id2word))

    questions, stats = load_questions(args.questions, vocab)
    print(
        f"checkpoint: arch={ckpt['arch']} dim={ckpt['dim']} window={ckpt['window']} "
        f"train_tokens={ckpt['train_tokens']:,} vocab={len(id2word):,}"
    )
    print(f"questions: total={stats['total']} evaluated={stats['evaluated']} skipped_oov={stats['skipped_oov']}")

    device = torch.device(args.device)
    result = evaluate_analogies(ckpt["word_vectors"], questions, device=device)

    print()
    print(f"{'category':32s} {'correct':>8s} {'total':>8s} {'accuracy':>9s}")
    for name in sorted(result):
        if name in ("semantic", "syntactic", "total"):
            continue
        r = result[name]
        print(f"{name:32s} {r['correct']:8d} {r['total']:8d} {r['accuracy'] * 100:8.1f}%")
    print("-" * 60)
    for name in ("semantic", "syntactic", "total"):
        if name not in result:
            continue
        r = result[name]
        print(f"{name:32s} {r['correct']:8d} {r['total']:8d} {r['accuracy'] * 100:8.1f}%")
