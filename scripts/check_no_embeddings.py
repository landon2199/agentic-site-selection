"""Licensing guard: fail if anything tracked or staged in git carries Esri embedding values.

    python scripts/check_no_embeddings.py          # checks staged + tracked files (also run by the pre-commit hook)

The USA Geodemographic Embeddings are Esri subscriber content (Esri Master License Agreement). Only scripts,
configs, derived scores and metrics may be committed. This fails on: any per-dimension embedding column name
(emb_ followed by digits), an 'Embedding' column header in a table, a long run of float values in one row
(a vector), or any binary data file type.
"""

from __future__ import annotations

import re
import subprocess
import sys

BAD_EXT = (".parquet", ".gdb", ".gdbtable", ".gdbtablx", ".npy", ".npz", ".feather", ".arrow", ".h5", ".pkl")
COL = re.compile(r"\bemb_\d+\b")
HEADER = re.compile(r'(^|[,|"\t])Embedding([,|"\t]|$)', re.M)
VECTOR = re.compile(r"(-?\d+\.\d+\s*,\s*){32,}")  # 32+ floats in a row looks like an embedding vector


def files() -> list[str]:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.split("\n")
    return sorted({f for f in run("ls-files") + run("diff", "--cached", "--name-only", "--diff-filter=ACMR") if f})


def main() -> int:
    bad = []
    for f in files():
        if f.lower().endswith(BAD_EXT) or ".gdb/" in f.lower():
            bad.append(f"{f}: binary data file type")
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except (FileNotFoundError, IsADirectoryError):
            continue
        if f.endswith("check_no_embeddings.py"):
            continue
        for rx, why in ((COL, "embedding column name"), (HEADER, "Embedding column header"), (VECTOR, "float vector")):
            m = rx.search(text)
            if m:
                bad.append(f"{f}: {why} near {m.group(0)[:40]!r}")
    if bad:
        print("BLOCKED (Esri embedding values must not be committed):\n  " + "\n  ".join(bad))
        return 1
    print(f"ok: {len(files())} files checked, no embedding columns or vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
