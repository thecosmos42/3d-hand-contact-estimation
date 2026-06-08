#!/usr/bin/env python3
"""
analyze_hcontact.py – Inspect InteractVLM hcontact_vertices.npz files.
Uses a mapping file (MANO_SMPLX_vertex_ids.pkl) to get correct hand vertex indices.
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np


def load_hand_vertex_mapping(pkl_path):
    """Return (left_indices, right_indices) as numpy arrays of vertex IDs."""
    with open(pkl_path, 'rb') as f:
        mapping = pickle.load(f, encoding='latin1')
    left = np.asarray(mapping['left_hand']).astype(int)
    right = np.asarray(mapping['right_hand']).astype(int)
    return left, right


def analyze_file(npz_path, left_indices, right_indices, threshold=0.3):
    """Print detailed analysis of a single .npz file."""
    print(f"\n{'='*70}")
    print(f"File: {npz_path}")
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"  ERROR: cannot load – {e}")
        return

    keys = list(data.keys())
    print(f"  Keys: {keys}")

    for key in keys:
        arr = data[key]
        print(f"\n  Key '{key}':")
        print(f"    shape = {arr.shape}")
        print(f"    dtype = {arr.dtype}")
        flat = arr.ravel()
        print(f"    min   = {flat.min():.8f}")
        print(f"    max   = {flat.max():.8f}")
        print(f"    mean  = {flat.mean():.8f}")
        print(f"    std   = {flat.std():.8f}")

        # For contact prediction tensors, show hand‑specific stats
        if key in ("pred_contact_3d_smplx", "pred_contact_3d_smplh", "pred_contact_3d"):
            scores = flat
            # Check if scores array is long enough to contain hand indices
            max_idx = max(left_indices.max(), right_indices.max())
            if scores.shape[0] > max_idx:
                right_scores = scores[right_indices]
                left_scores = scores[left_indices]
                print(f"    right hand (indices {right_indices.min()}–{right_indices.max()}):")
                print(f"      min={right_scores.min():.6f} max={right_scores.max():.6f} mean={right_scores.mean():.6f}")
                print(f"      # above {threshold}: {(right_scores > threshold).sum()} / {len(right_scores)}")
                print(f"    left hand (indices {left_indices.min()}–{left_indices.max()}):")
                print(f"      min={left_scores.min():.6f} max={left_scores.max():.6f} mean={left_scores.mean():.6f}")
                print(f"      # above {threshold}: {(left_scores > threshold).sum()} / {len(left_scores)}")
            else:
                print(f"    (hand slices not applicable – scores array too short: {scores.shape[0]} < {max_idx+1})")

    data.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze InteractVLM hcontact_vertices.npz files")
    parser.add_argument("input", type=str, help="Path to a .npz file or a directory containing .npz files")
    parser.add_argument("--mapping_pkl", type=str,
                        default="/scratch-shared/scur0805/InterFieldHands/MANO_SMPLX_vertex_ids.pkl",
                        help="Path to MANO_SMPLX_vertex_ids.pkl")
    parser.add_argument("--pattern", default="box__*_hcontact_vertices.npz",
                        help="Glob pattern (default: box__*_hcontact_vertices.npz)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Threshold for counting contact predictions (default: 0.3)")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all .npz files in directory (otherwise only first 5)")
    args = parser.parse_args()

    # Load the vertex mapping
    mapping_path = Path(args.mapping_pkl)
    if not mapping_path.exists():
        sys.exit(f"Mapping file not found: {mapping_path}")
    left_idx, right_idx = load_hand_vertex_mapping(mapping_path)
    print(f"Loaded left hand indices: {len(left_idx)} verts (min={left_idx.min()}, max={left_idx.max()})")
    print(f"Loaded right hand indices: {len(right_idx)} verts (min={right_idx.min()}, max={right_idx.max()})")

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        sys.exit(1)

    if in_path.is_file():
        analyze_file(in_path, left_idx, right_idx, args.threshold)
    else:
        files = sorted(in_path.glob(args.pattern))
        if not files:
            print(f"No files matching '{args.pattern}' in {in_path}")
            sys.exit(1)

        print(f"Found {len(files)} file(s). Analyzing first {len(files) if args.all else 5}...")
        for f in files[:None if args.all else 5]:
            analyze_file(f, left_idx, right_idx, args.threshold)
        if not args.all and len(files) > 5:
            print(f"\n... and {len(files)-5} more. Use --all to analyze all.")


if __name__ == "__main__":
    main()