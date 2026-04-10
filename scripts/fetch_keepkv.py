#!/usr/bin/env python3
"""Fetch or update the external KeepKV checkout used for operator reference."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "external" / "KeepKV"
KEEPKV_REPO = "https://github.com/kkvcache/KeepKV.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--ref", default=None, help="Optional branch, tag, or commit to checkout after sync.")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(command, cwd=str(cwd) if cwd is not None else None)


def main() -> None:
    args = parse_args()
    dest = args.dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        run(["git", "clone", KEEPKV_REPO, str(dest)])
    else:
        run(["git", "-C", str(dest), "fetch", "origin", "--tags"])
        run(["git", "-C", str(dest), "pull", "--ff-only"])
    if args.ref:
        run(["git", "-C", str(dest), "checkout", args.ref])
    print(f"keepkv_checkout={dest}")


if __name__ == "__main__":
    main()
