"""Analogy evaluation: Section 4.1's "3CosAdd" procedure.

For a question (a, b, c, d) read as "a is to b as c is to d" (e.g. Athens is
to Greece as Oslo is to Norway), compute X = vec(b) - vec(a) + vec(c) and
predict the vocabulary word (other than a, b, c) whose vector is closest to
X by cosine similarity. Correct only on an exact match to d -- synonyms count
as wrong, per the paper's stated scoring rule (Section 4.1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from word2vec.vocab import Vocab

SEMANTIC_CATEGORIES = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}


@dataclass
class Question:
    category: str
    is_semantic: bool
    a: int
    b: int
    c: int
    d: int


def load_questions(path: Path, vocab: Vocab) -> tuple[list[Question], dict]:
    """text8 (and therefore our vocab) is all lowercase; the analogy file
    uses standard capitalization, so we lowercase before lookup. Questions
    with any word missing from the vocabulary are skipped and counted.
    """
    total = 0
    skipped = 0
    questions: list[Question] = []
    category = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                continue

            words = line.lower().split()
            total += 1
            if len(words) != 4 or not all(w in vocab.word2id for w in words):
                skipped += 1
                continue

            a, b, c, d = (vocab.word2id[w] for w in words)
            questions.append(Question(category, category in SEMANTIC_CATEGORIES, a, b, c, d))

    stats = {"total": total, "skipped_oov": skipped, "evaluated": len(questions)}
    return questions, stats


def evaluate_analogies(
    word_vectors: torch.Tensor,
    questions: list[Question],
    device: torch.device,
    batch_size: int = 1000,
) -> dict:
    """Returns per-category accuracy plus semantic/syntactic/total rollups,
    each as {"correct": int, "total": int, "accuracy": float}.
    """
    if not questions:
        return {"total": {"correct": 0, "total": 0, "accuracy": 0.0}}

    vecs = word_vectors.to(device)
    normed = torch.nn.functional.normalize(vecs, dim=1)

    per_question_correct = torch.empty(len(questions), dtype=torch.bool)

    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        a = torch.tensor([q.a for q in batch], device=device)
        b = torch.tensor([q.b for q in batch], device=device)
        c = torch.tensor([q.c for q in batch], device=device)
        d = torch.tensor([q.d for q in batch], device=device)

        # cosine(target, v) for unit-norm v is proportional to target . v
        # regardless of target's own norm, so skipping target normalization
        # doesn't change the argmax -- only the (unused) similarity scale.
        target = normed[b] - normed[a] + normed[c]
        sims = target @ normed.T  # (batch, V)

        neg_inf = torch.finfo(sims.dtype).min
        sims.scatter_(1, a.unsqueeze(1), neg_inf)
        sims.scatter_(1, b.unsqueeze(1), neg_inf)
        sims.scatter_(1, c.unsqueeze(1), neg_inf)

        pred = sims.argmax(dim=1)
        per_question_correct[start : start + len(batch)] = (pred == d).cpu()

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for q, correct in zip(questions, per_question_correct.tolist()):
        tally[q.category][0] += int(correct)
        tally[q.category][1] += 1
        bucket = "semantic" if q.is_semantic else "syntactic"
        tally[bucket][0] += int(correct)
        tally[bucket][1] += 1
        tally["total"][0] += int(correct)
        tally["total"][1] += 1

    return {
        name: {"correct": c, "total": t, "accuracy": c / t if t else 0.0}
        for name, (c, t) in tally.items()
    }
