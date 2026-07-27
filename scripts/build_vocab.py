import argparse
from pathlib import Path

from word2vec.vocab import build_vocab, read_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/text8"))
    parser.add_argument("--out", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    tokens = read_tokens(args.corpus)
    vocab = build_vocab(tokens, min_count=args.min_count)
    vocab.save(args.out)

    dropped = len(set(tokens)) - len(vocab)
    print(f"corpus tokens:      {len(tokens):,}")
    print(f"unique raw words:   {len(set(tokens)):,}")
    print(f"vocab size (>= {args.min_count}): {len(vocab):,}  ({dropped:,} rare words dropped)")
    print(f"tokens after filter: {vocab.total_tokens:,}")
    print(f"top 10 words: {vocab.id2word[:10]}")
    print(f"saved to {args.out}")
