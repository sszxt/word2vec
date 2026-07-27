"""Download the word-analogy test set described in Section 4.1 of the paper.

The paper's own hosted copy (fit.vutbr.cz) has moved around over the years.
We use the copy in the original author's word2vec repository instead, which
is the same file distributed with the released C tool: 5 semantic + 9
syntactic categories, 19544 questions total (matches the paper's reported
8869 semantic + 10675 syntactic exactly).
"""

import argparse
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/questions-words.txt"))
    args = parser.parse_args()

    if args.out.exists():
        print(f"{args.out} already exists, skipping download.")
    else:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(resp.text, encoding="utf-8")

    lines = args.out.read_text(encoding="utf-8").splitlines()
    categories = [l for l in lines if l.startswith(":")]
    questions = [l for l in lines if l and not l.startswith(":")]
    print(f"saved to {args.out}")
    print(f"{len(categories)} categories, {len(questions)} questions")
