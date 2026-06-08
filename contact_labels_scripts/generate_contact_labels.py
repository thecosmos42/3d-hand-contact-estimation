#!/usr/bin/env python3
"""
generate_contact_labels.py
==========================
Generate per-frame binary contact labels for the ARCTIC bimanual hand-object
interaction dataset from preprocessed vertex files.

Contact definition (from the ARCTIC paper):
    A hand vertex is "in contact" if its Euclidean distance to the nearest
    object vertex is less than `threshold` metres (default 3 mm = 0.003 m).

Assumptions about the processed .npy files (created via process_seqs.py --export_verts):
    - Each file contains a dict with a top-level key 'world_coord'.
    - 'world_coord' is a dict with sub-keys:
        'verts.left'   : (T, 778, 3)  left hand vertices (meters)
        'verts.right'  : (T, 778, 3)  right hand vertices (meters)
        'verts.object' : (T, N_obj, 3) articulated object vertices (meters)
    - Other keys (joints, diameter, ...) are ignored.

Usage example:
    python generate_contact_labels.py \
        --processed_verts_dir outputs/processed_verts/seqs/ \
        --output_dir outputs/contact_labels/ \
        --threshold 0.003 \
        --include_verts \
        --verbose
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def compute_contact_for_frame(
    left_verts: np.ndarray,   # (778, 3)
    right_verts: np.ndarray,  # (778, 3)
    obj_verts: np.ndarray,    # (N_obj, 3)
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a KD-Tree on object vertices and query nearest-neighbour distances
    for every left- and right-hand vertex.

    Returns
    -------
    contact_left  : bool array, shape (778,)
    contact_right : bool array, shape (778,)
    """
    if obj_verts.shape[0] == 0:
        warnings.warn("Object vertex array is empty; returning all-False contact.")
        return (
            np.zeros(left_verts.shape[0], dtype=bool),
            np.zeros(right_verts.shape[0], dtype=bool),
        )

    tree = cKDTree(obj_verts)
    dist_left, _ = tree.query(left_verts)
    dist_right, _ = tree.query(right_verts)

    # Replace NaN distances (can happen with degenerate points) with inf
    if np.any(np.isnan(dist_left)) or np.any(np.isnan(dist_right)):
        warnings.warn("NaN distances detected; replacing with infinity.")
        dist_left = np.nan_to_num(dist_left, nan=np.inf)
        dist_right = np.nan_to_num(dist_right, nan=np.inf)

    contact_left = dist_left <= threshold
    contact_right = dist_right <= threshold

    return contact_left, contact_right


def process_sequence(
    npy_path: Path,
    output_seq_dir: Path,
    threshold: float,
    overwrite: bool,
    include_verts: bool,
    verbose: bool,
) -> dict:
    """
    Load one processed sequence .npy file and write per-frame .npz contact
    label files.

    Returns a stats dict with keys:
        processed   – frames newly computed
        skipped     – frames skipped (output already existed)
        left_ratios – list of per-frame left-hand contact fractions
        right_ratios– list of per-frame right-hand contact fractions
    """
    stats = {"processed": 0, "skipped": 0, "left_ratios": [], "right_ratios": []}

    # ---- Load sequence data ------------------------------------------------
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as exc:
        warnings.warn(f"Failed to load {npy_path}: {exc}. Skipping.")
        return stats

    # ---- Extract world-space vertices from the expected structure ----------
    try:
        wc = data["world_coord"]
        left_all = wc["verts.left"]      # (T, 778, 3)
        right_all = wc["verts.right"]    # (T, 778, 3)
        obj_all = wc["verts.object"]     # (T, N_obj, 3)
    except KeyError as exc:
        available_top = list(data.keys())
        available_wc = list(data["world_coord"].keys()) if "world_coord" in data else []
        warnings.warn(
            f"Unexpected data structure in {npy_path}: {exc}. "
            f"Top-level keys: {available_top}, world_coord keys: {available_wc}. "
            "Skipping sequence."
        )
        return stats

    num_frames = left_all.shape[0]

    # Basic sanity check on vertex counts
    if left_all.shape[1] != 778 or right_all.shape[1] != 778:
        warnings.warn(
            f"Unexpected hand vertex count in {npy_path} "
            f"(left={left_all.shape[1]}, right={right_all.shape[1]}). "
            "Expected 778. Continuing anyway."
        )

    output_seq_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-frame loop ----------------------------------------------------
    for frame_idx in range(num_frames):
        out_path = output_seq_dir / f"frame_{frame_idx:05d}.npz"

        # Skip if output already exists and we are not overwriting
        if out_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        left_verts_f = left_all[frame_idx]
        right_verts_f = right_all[frame_idx]
        obj_verts_f = obj_all[frame_idx]

        contact_left, contact_right = compute_contact_for_frame(
            left_verts_f, right_verts_f, obj_verts_f, threshold
        )

        save_dict = {
            "left_contact": contact_left,
            "right_contact": contact_right,
        }
        if include_verts:
            save_dict["left_verts"] = left_verts_f
            save_dict["right_verts"] = right_verts_f
            save_dict["obj_verts"] = obj_verts_f

        np.savez_compressed(out_path, **save_dict)

        stats["processed"] += 1
        stats["left_ratios"].append(contact_left.mean())
        stats["right_ratios"].append(contact_right.mean())

        if verbose and (frame_idx % 100 == 0 or frame_idx == num_frames - 1):
            print(
                f"  frame {frame_idx+1:>5}/{num_frames}  "
                f"L-contact: {contact_left.sum():>3}/{contact_left.size}  "
                f"R-contact: {contact_right.sum():>3}/{contact_right.size}"
            )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary hand-object contact labels for ARCTIC."
    )
    parser.add_argument(
        "--processed_verts_dir",
        required=True,
        type=Path,
        help="Root directory containing per-subject subdirs with .npy sequence files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Root directory where per-frame .npz contact labels will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.003,
        help="Contact distance threshold in metres (default: 0.003 = 3 mm).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing output .npz files.",
    )
    parser.add_argument(
        "--include_verts",
        action="store_true",
        help=(
            "Also store left_verts, right_verts, obj_verts in each .npz "
            "(useful for the visualisation script)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_verts_dir: Path = args.processed_verts_dir
    output_dir: Path = args.output_dir

    if not processed_verts_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {processed_verts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all .npy files recursively
    npy_files = sorted(processed_verts_dir.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found under {processed_verts_dir}. Exiting.")
        return

    print(f"Found {len(npy_files)} sequence file(s) under {processed_verts_dir}.")
    print(f"Contact threshold: {args.threshold*1000:.1f} mm  |  overwrite={args.overwrite}")
    print("-" * 60)

    total_processed = 0
    total_skipped = 0
    all_left_ratios: list[float] = []
    all_right_ratios: list[float] = []

    for npy_path in npy_files:
        # Mirror directory structure: output_dir / <subject> / <seq_stem>
        rel_path = npy_path.relative_to(processed_verts_dir)   # e.g. s01/cap_use_01.npy
        output_seq_dir = output_dir / rel_path.parent / rel_path.stem

        if args.verbose:
            print(f"\n[SEQ] {rel_path}  →  {output_seq_dir}")

        seq_stats = process_sequence(
            npy_path=npy_path,
            output_seq_dir=output_seq_dir,
            threshold=args.threshold,
            overwrite=args.overwrite,
            include_verts=args.include_verts,
            verbose=args.verbose,
        )

        total_processed += seq_stats["processed"]
        total_skipped += seq_stats["skipped"]
        all_left_ratios.extend(seq_stats["left_ratios"])
        all_right_ratios.extend(seq_stats["right_ratios"])

        if not args.verbose:
            print(f"  {rel_path}: processed={seq_stats['processed']}, skipped={seq_stats['skipped']}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Total frames processed : {total_processed}")
    print(f"  Total frames skipped   : {total_skipped}")
    if all_left_ratios:
        avg_left = float(np.mean(all_left_ratios)) * 100
        avg_right = float(np.mean(all_right_ratios)) * 100
        print(f"  Avg left-hand  contact ratio : {avg_left:.2f}%")
        print(f"  Avg right-hand contact ratio : {avg_right:.2f}%")
    else:
        print("  (No frames were processed — nothing to report.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
