#!/usr/bin/env python3
"""
eval_smplx_vs_gt.py
===================
Compare InteractVLM pred_contact_3d_smplx predictions against pre-computed
boolean ground-truth contact labels.

Pipeline
--------
1. Auto-discover prediction frames from contact_output dir
   (box__NNNNN_hcontact_vertices.npz → frame numbers)
2. For each frame:
   a. Load pred_contact_3d_smplx (10475,) from prediction .npz
   b. Apply MANO hand vertex mask from MANO_SMPLX_vertex_ids.pkl
      → pred_right (M_r,)  pred_left (M_l,)  where M_r, M_l ≤ 778
   c. Binarise with pred_threshold=0.3
   d. Load GT frame_NNNNN.npz → left_contact (bool array), right_contact (bool array)
   e. Accumulate TP/FP/FN for each hand
3. Print per-50-frame stats and final aggregate P/R/F1

Paths (edit here or pass via CLI)
----------------------------------
  PREDS_DIR  : directory containing box__*_hcontact_vertices.npz
  GT_DIR     : directory containing frame_*.npz
  PKL_PATH   : MANO_SMPLX_vertex_ids.pkl

Usage
-----
  python eval_smplx_vs_gt.py
  python eval_smplx_vs_gt.py --preds_dir /other/path --gt_dir /other/gt
  python eval_smplx_vs_gt.py --pred_threshold 0.4 --out_dir /results
"""

from __future__ import annotations

import os
import re
import glob
import pickle
import argparse
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS — edit these or override via CLI
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PREDS_DIR = "/scratch-shared/scur0805/arctic_data_ego/contact_output"
DEFAULT_GT_DIR    = "/scratch-shared/scur0805/gt_seqs"
DEFAULT_PKL_PATH  = "/scratch-shared/scur0805/InterFieldHands/MANO_SMPLX_vertex_ids.pkl"
DEFAULT_PRED_THR  = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_hand_vertex_mapping(pkl_path: str):
    """
    Return (left_indices, right_indices) as numpy int arrays of SMPL-X vertex IDs.
    These index into the 10475-vert pred_contact_3d_smplx array.
    """
    with open(pkl_path, "rb") as f:
        mapping = pickle.load(f, encoding="latin1")
    left  = np.asarray(mapping["left_hand"]).astype(int)
    right = np.asarray(mapping["right_hand"]).astype(int)
    return left, right


def discover_frame_ids(preds_dir: str) -> list[int]:
    """
    Glob all box__NNNNN_hcontact_vertices.npz files and return
    sorted list of integer frame numbers.
    """
    pattern = os.path.join(preds_dir, "*_hcontact_vertices.npz")
    ids = []
    for f in glob.glob(pattern):
        m = re.search(r"__(\d+)_hcontact_vertices\.npz$", os.path.basename(f))
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_pred_contact(npz_path: str,
                      left_idx: np.ndarray,
                      right_idx: np.ndarray,
                      threshold: float):
    """
    Load pred_contact_3d_smplx, apply hand masks, binarise.

    Returns
    -------
    pred_right : (len(right_idx),) bool
    pred_left  : (len(left_idx),)  bool
    raw_right  : (len(right_idx),) float  — raw scores for diagnostics
    raw_left   : (len(left_idx),)  float
    """
    d = np.load(npz_path)
    if "pred_contact_3d_smplx" not in d:
        raise KeyError(
            f"'pred_contact_3d_smplx' not found in {npz_path}. "
            f"Keys: {list(d.keys())}"
        )
    scores = d["pred_contact_3d_smplx"].squeeze().astype(np.float32)  # (10475,)

    raw_right  = scores[right_idx]
    raw_left   = scores[left_idx]
    pred_right = raw_right >= threshold
    pred_left  = raw_left  >= threshold
    return pred_right, pred_left, raw_right, raw_left


def load_gt_contact(gt_path: str):
    """
    Load ground-truth boolean contact arrays.

    Returns
    -------
    gt_right : (N_r,) bool
    gt_left  : (N_l,) bool
    """
    d = np.load(gt_path)
    missing = [k for k in ("right_contact", "left_contact") if k not in d]
    if missing:
        raise KeyError(f"GT file {gt_path} missing keys: {missing}. Found: {list(d.keys())}")
    gt_right = d["right_contact"].astype(bool)
    gt_left  = d["left_contact"].astype(bool)
    return gt_right, gt_left


def acc_metrics(pred: np.ndarray, gt: np.ndarray, acc: dict) -> dict:
    """
    Add TP/FP/FN from one (pred, gt) pair into accumulator dict.
    Both arrays must have the same length.

    If lengths differ (pred from SMPL-X mask, gt from MANO — different vertex
    counts are possible), truncates to the shorter length with a warning flag
    stored in acc["length_mismatch"].
    """
    n = min(len(pred), len(gt))
    if len(pred) != len(gt):
        acc["length_mismatch"] = acc.get("length_mismatch", 0) + 1
        pred, gt = pred[:n], gt[:n]

    acc["tp"] += int(np.sum( pred &  gt))
    acc["fp"] += int(np.sum( pred & ~gt))
    acc["fn"] += int(np.sum(~pred &  gt))
    acc["gt_pos"] += int(gt.sum())
    acc["pred_pos"] += int(pred.sum())
    return acc


def finalise(acc: dict) -> dict:
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {**acc, "precision": p, "recall": r, "f1": f1}


def fmt_metrics(label: str, m: dict) -> str:
    return (
        f"  {label:12s}  "
        f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
        f"(TP={m['tp']:5d}  FP={m['fp']:5d}  FN={m['fn']:5d}  "
        f"GT+={m['gt_pos']:5d}  Pred+={m['pred_pos']:5d})"
    )


def empty_acc() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0, "gt_pos": 0, "pred_pos": 0}


def print_per_frame_header():
    print(
        f"\n{'Frame':>7}  "
        f"{'GT_R':>6} {'Pred_R':>7}  "
        f"{'GT_L':>6} {'Pred_L':>7}  "
        f"{'R_F1':>6}  {'L_F1':>6}"
    )
    print("-" * 66)


def fmt_frame_row(frame_id: int,
                  gt_r: np.ndarray, pred_r: np.ndarray,
                  gt_l: np.ndarray, pred_l: np.ndarray) -> str:
    def f1(p, g):
        tp = int(np.sum(p & g))
        fp = int(np.sum(p & ~g))
        fn = int(np.sum(~p & g))
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    n = min(len(pred_r), len(gt_r))
    pr, gr = pred_r[:n], gt_r[:n]
    n = min(len(pred_l), len(gt_l))
    pl, gl = pred_l[:n], gt_l[:n]

    return (
        f"  {frame_id:5d}  "
        f"{int(gr.sum()):6d} {int(pr.sum()):7d}  "
        f"{int(gl.sum()):6d} {int(pl.sum()):7d}  "
        f"{f1(pr, gr):6.3f}  {f1(pl, gl):6.3f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InteractVLM 3D contact vs GT boolean contact labels."
    )
    parser.add_argument("--preds_dir",       type=str,   default=DEFAULT_PREDS_DIR)
    parser.add_argument("--gt_dir",          type=str,   default=DEFAULT_GT_DIR)
    parser.add_argument("--pkl_path",        type=str,   default=DEFAULT_PKL_PATH)
    parser.add_argument("--pred_threshold",  type=float, default=DEFAULT_PRED_THR,
                        help="Score threshold to binarise pred_contact_3d_smplx. "
                             "Default: 0.3")
    parser.add_argument("--out_dir",         type=str,   default=None,
                        help="Where to save results.txt. Defaults to --preds_dir.")
    parser.add_argument("--chunk_size",      type=int,   default=50,
                        help="Print running stats every N frames. Default: 50.")
    args = parser.parse_args()

    out_dir = args.out_dir or args.preds_dir
    os.makedirs(out_dir, exist_ok=True)

    sep = "=" * 70
    print(sep)
    print("InteractVLM 3D Contact Evaluation  (SMPL-X mask → GT boolean)")
    print(sep)
    print(f"  Preds dir        : {args.preds_dir}")
    print(f"  GT dir           : {args.gt_dir}")
    print(f"  PKL path         : {args.pkl_path}")
    print(f"  Pred threshold   : {args.pred_threshold}")
    print(f"  Chunk size       : {args.chunk_size} frames")
    print()

    # ── Load hand vertex mapping ──────────────────────────────────────────────
    print(f"Loading hand vertex mapping from:\n  {args.pkl_path}")
    left_idx, right_idx = load_hand_vertex_mapping(args.pkl_path)
    print(f"  Right-hand indices : {len(right_idx)} verts "
          f"(range [{right_idx.min()}, {right_idx.max()}])")
    print(f"  Left-hand  indices : {len(left_idx)} verts "
          f"(range [{left_idx.min()}, {left_idx.max()}])")

    # ── Discover frames ───────────────────────────────────────────────────────
    frame_ids = discover_frame_ids(args.preds_dir)
    if not frame_ids:
        print(f"\nERROR: No *_hcontact_vertices.npz files found in:\n  {args.preds_dir}")
        return

    print(f"\nDiscovered {len(frame_ids)} prediction frames: "
          f"{frame_ids[0]} … {frame_ids[-1]}")

    # ── Accumulators ─────────────────────────────────────────────────────────
    agg_right = empty_acc()
    agg_left  = empty_acc()

    # Chunk accumulators
    chunk_right = empty_acc()
    chunk_left  = empty_acc()
    chunk_start = frame_ids[0]

    skipped = []
    print_per_frame_header()

    for i, frame_id in enumerate(frame_ids):

        # Paths
        pred_fname = None
        # find the actual file (prefix may vary)
        pattern = os.path.join(args.preds_dir, f"*__{frame_id:05d}_hcontact_vertices.npz")
        matches = glob.glob(pattern)
        if not matches:
            skipped.append(frame_id)
            continue
        pred_path = matches[0]

        gt_path = os.path.join(args.gt_dir, f"frame_{frame_id:05d}.npz")
        if not os.path.exists(gt_path):
            skipped.append(frame_id)
            continue

        # Load
        try:
            pred_right, pred_left, _, _ = load_pred_contact(
                pred_path, left_idx, right_idx, args.pred_threshold
            )
            gt_right, gt_left = load_gt_contact(gt_path)
        except (KeyError, ValueError) as e:
            print(f"  WARNING frame {frame_id}: {e}")
            skipped.append(frame_id)
            continue

        # Per-frame row
        print(fmt_frame_row(frame_id, gt_right, pred_right, gt_left, pred_left))

        # Accumulate
        acc_metrics(pred_right, gt_right, agg_right)
        acc_metrics(pred_left,  gt_left,  agg_left)
        acc_metrics(pred_right, gt_right, chunk_right)
        acc_metrics(pred_left,  gt_left,  chunk_left)

        # Print chunk summary every `chunk_size` processed frames
        is_last   = (i == len(frame_ids) - 1)
        chunk_end = (i + 1) % args.chunk_size == 0

        if chunk_end or is_last:
            cr = finalise(chunk_right)
            cl = finalise(chunk_left)
            print(f"\n  ── Frames {chunk_start}–{frame_id} "
                  f"({args.chunk_size if chunk_end else (i % args.chunk_size) + 1} frames) ──")
            print(fmt_metrics("Right hand", cr))
            print(fmt_metrics("Left  hand", cl))

            if chunk_right.get("length_mismatch", 0) > 0:
                print(f"  WARNING: {chunk_right['length_mismatch']} frames had "
                      f"right-hand length mismatch between pred and GT.")
            if chunk_left.get("length_mismatch", 0) > 0:
                print(f"  WARNING: {chunk_left['length_mismatch']} frames had "
                      f"left-hand length mismatch between pred and GT.")

            # Reset chunk accumulators
            chunk_right = empty_acc()
            chunk_left  = empty_acc()
            chunk_start = frame_ids[i + 1] if not is_last else frame_id
            print_per_frame_header()

    # ── Final aggregate results ───────────────────────────────────────────────
    final_right = finalise(agg_right)
    final_left  = finalise(agg_left)

    both = {
        "tp":       agg_right["tp"]       + agg_left["tp"],
        "fp":       agg_right["fp"]       + agg_left["fp"],
        "fn":       agg_right["fn"]       + agg_left["fn"],
        "gt_pos":   agg_right["gt_pos"]   + agg_left["gt_pos"],
        "pred_pos": agg_right["pred_pos"] + agg_left["pred_pos"],
    }
    final_both = finalise(both)

    print(f"\n{sep}")
    print("Final Results (aggregate over all frames)")
    print(sep)
    print(f"  Pred threshold   : {args.pred_threshold}")
    print(f"  Frames processed : {len(frame_ids) - len(skipped)}")
    print(f"  Frames skipped   : {len(skipped)}"
          + (f"  {skipped[:5]}{'…' if len(skipped) > 5 else ''}" if skipped else ""))
    print()

    lines = []
    for label, m in [("Right hand", final_right),
                     ("Left  hand", final_left),
                     ("Both  hands", final_both)]:
        line = fmt_metrics(label, m)
        print(line)
        lines.append(line)

    if agg_right.get("length_mismatch", 0) or agg_left.get("length_mismatch", 0):
        warn = (
            f"\n  NOTE: {agg_right.get('length_mismatch', 0)} right / "
            f"{agg_left.get('length_mismatch', 0)} left frames had mismatched "
            f"pred vs GT vertex counts — truncated to shorter length."
        )
        print(warn)
        lines.append(warn)

    # ── Write results file ────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "results_smplx_gt.txt")
    try:
        with open(out_path, "w") as f:
            f.write("InteractVLM 3D Contact Evaluation (SMPL-X mask vs GT boolean)\n")
            f.write(f"Preds dir      : {args.preds_dir}\n")
            f.write(f"GT dir         : {args.gt_dir}\n")
            f.write(f"PKL path       : {args.pkl_path}\n")
            f.write(f"Pred threshold : {args.pred_threshold}\n")
            f.write(f"Frames         : {len(frame_ids) - len(skipped)} processed, "
                    f"{len(skipped)} skipped\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"\n  Results written to: {out_path}")
    except OSError as e:
        print(f"\n  WARNING: Could not write results file: {e}")

    print(sep)


if __name__ == "__main__":
    main()