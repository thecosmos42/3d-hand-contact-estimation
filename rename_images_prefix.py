#!/usr/bin/env python3
"""Rename files in a folder by prefixing their filenames.

Default behavior is a dry-run (prints planned renames). Use --apply to perform changes.

Examples:
  python scripts/rename_images_prefix.py --dir images
  python scripts/rename_images_prefix.py --dir images --prefix frame_0 --sep _ --apply

Notes:
- Skips directories.
- Skips files that already start with the prefix (optionally with separator).
- Aborts on name collisions unless --dedupe is used.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenameOp:
    src: Path
    dst: Path


def _already_prefixed(name: str, prefix: str, sep: str) -> bool:
    if name.startswith(prefix):
        return True
    if sep and name.startswith(prefix + sep):
        return True
    return False


def _dedupe_target(target: Path) -> Path:
    """Generate a non-existing target path by appending _N before suffix."""
    stem = target.stem
    suffix = target.suffix
    parent = target.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def build_plan(directory: Path, prefix: str, sep: str, dedupe: bool) -> list[RenameOp]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    ops: list[RenameOp] = []

    # Deterministic ordering helps review and avoids surprises.
    for src in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not src.is_file():
            continue

        name = src.name
        if _already_prefixed(name, prefix, sep):
            continue

        dst_name = f"{prefix}{sep}{name}" if sep else f"{prefix}{name}"
        dst = src.with_name(dst_name)

        if dst.exists():
            if dedupe:
                dst = _dedupe_target(dst)
            else:
                raise FileExistsError(
                    f"Target already exists for {src.name} -> {dst.name}. "
                    "Re-run with --dedupe or choose a different --prefix/--sep."
                )

        ops.append(RenameOp(src=src, dst=dst))

    # Safety: ensure we never rename two sources to the same destination.
    dsts = [op.dst.name.lower() for op in ops]
    if len(dsts) != len(set(dsts)):
        raise RuntimeError("Planned renames contain duplicate destinations (case-insensitive).")

    return ops


def build_plan_recursive(directory: Path, prefix: str, sep: str, dedupe: bool) -> list[RenameOp]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    ops: list[RenameOp] = []

    # Build plan first (no in-place renames while traversing).
    sources = [p for p in directory.rglob("*") if p.is_file()]
    sources.sort(key=lambda p: str(p.relative_to(directory)).lower())

    for src in sources:
        name = src.name
        if _already_prefixed(name, prefix, sep):
            continue

        dst_name = f"{prefix}{sep}{name}" if sep else f"{prefix}{name}"
        dst = src.with_name(dst_name)

        if dst.exists():
            if dedupe:
                dst = _dedupe_target(dst)
            else:
                raise FileExistsError(
                    f"Target already exists for {src} -> {dst}. "
                    "Re-run with --dedupe or choose a different --prefix/--sep."
                )

        ops.append(RenameOp(src=src, dst=dst))

    # Safety: ensure we never rename two sources to the same destination path.
    dsts = [str(op.dst).lower() for op in ops]
    if len(dsts) != len(set(dsts)):
        raise RuntimeError("Planned renames contain duplicate destination paths (case-insensitive).")

    return ops


def apply_plan(ops: list[RenameOp]) -> None:
    for op in ops:
        os.replace(op.src, op.dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefix all filenames in a directory")
    parser.add_argument(
        "--dir",
        default="images",
        type=Path,
        help="Directory containing images (default: images)",
    )
    parser.add_argument(
        "--prefix",
        default="frame_0",
        help="Prefix to add to the start of each filename (default: frame_0)",
    )
    parser.add_argument(
        "--sep",
        default="",
        help="Optional separator between prefix and original name (e.g. '_' => frame_0_<name>)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames (otherwise dry-run)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="If a target name exists, append _N to make it unique",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only rename files directly inside --dir (default renames recursively)",
    )

    args = parser.parse_args()

    directory = args.dir
    if args.no_recursive:
        ops = build_plan(directory=directory, prefix=args.prefix, sep=args.sep, dedupe=args.dedupe)
    else:
        ops = build_plan_recursive(directory=directory, prefix=args.prefix, sep=args.sep, dedupe=args.dedupe)

    if not ops:
        print("No files to rename (already prefixed or directory is empty).")
        return 0

    for op in ops:
        # Show relative paths for clarity with recursive runs
        try:
            src_rel = op.src.relative_to(directory)
            dst_rel = op.dst.relative_to(directory)
            print(f"{src_rel} -> {dst_rel}")
        except ValueError:
            print(f"{op.src} -> {op.dst}")

    if not args.apply:
        print(f"\nDry-run: {len(ops)} rename(s) planned. Re-run with --apply to execute.")
        return 0

    apply_plan(ops)
    print(f"\nDone: {len(ops)} file(s) renamed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
