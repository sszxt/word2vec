"""Vocabulary construction: read raw tokens, count frequencies, assign integer ids.

Mirrors the original word2vec.c behavior: words occurring fewer than `min_count`
times are dropped from the vocabulary entirely (not mapped to an <UNK> token),
and ids are assigned in descending frequency order so id 0 is the most frequent
word. Descending-frequency ordering matters later for Huffman tree construction.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def read_tokens(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().split()


@dataclass
class Vocab:
    word2id: dict[str, int]
    id2word: list[str]
    counts: list[int]

    def __len__(self) -> int:
        return len(self.id2word)

    @property
    def total_tokens(self) -> int:
        return sum(self.counts)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2id[t] for t in tokens if t in self.word2id]

    def save(self, path: Path) -> None:
        payload = {"id2word": self.id2word, "counts": self.counts}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id2word = payload["id2word"]
        counts = payload["counts"]
        word2id = {w: i for i, w in enumerate(id2word)}
        return cls(word2id=word2id, id2word=id2word, counts=counts)


def build_vocab(tokens: list[str], min_count: int = 5) -> Vocab:
    freq = Counter(tokens)
    kept = [(w, c) for w, c in freq.items() if c >= min_count]
    kept.sort(key=lambda wc: wc[1], reverse=True)

    id2word = [w for w, _ in kept]
    counts = [c for _, c in kept]
    word2id = {w: i for i, w in enumerate(id2word)}
    return Vocab(word2id=word2id, id2word=id2word, counts=counts)
