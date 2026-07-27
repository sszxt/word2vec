"""Download and extract the text8 corpus.

text8 is the first 10^8 bytes of a cleaned English Wikipedia dump (enwik9),
lowercased with punctuation stripped, whitespace-tokenized. It is the corpus
the original word2vec.c demo scripts train on, ~17M tokens / ~253K unique
words before frequency filtering. Source: http://mattmahoney.net/dc/textdata.html
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "http://mattmahoney.net/dc/text8.zip"


def download(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "text8.zip"
    txt_path = dest_dir / "text8"

    if txt_path.exists():
        print(f"{txt_path} already exists, skipping download.")
        return txt_path

    if not zip_path.exists():
        resp = requests.get(URL, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="downloading text8.zip"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    zip_path.unlink()
    return txt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = download(args.dest)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"text8 ready at {path} ({size_mb:.1f} MB)")
