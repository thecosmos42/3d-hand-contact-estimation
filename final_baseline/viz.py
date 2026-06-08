#!/usr/bin/env python3
"""
visualize_mano_contact.py
=========================
Render per-frame MANO hand contact visualisations using Matplotlib.
Produces two PNG files per frame:

  frame_NNNNN_gt.png   — GT contact  (left_contact / right_contact booleans)
  frame_NNNNN_pred.png — Pred contact (pred_contact_3d_smplx → MANO mask → thr)

Mesh topology (faces) comes from MANO_LEFT.pkl / MANO_RIGHT.pkl.
Vertex positions (XYZ) come from ARCTIC processed_verts .npy.
Contact labels:
  GT   → frame_NNNNN.npz  keys: left_contact, right_contact  (bool arrays)
  Pred → box__NNNNN_hcontact_vertices.npz  key: pred_contact_3d_smplx (10475,)
         masked via MANO_SMPLX_vertex_ids.pkl

Each PNG shows four 3D-projected panels (two projections × two hands):
  Top row    : XZ projection  (front view)
  Bottom row : XY projection  (top-down view)
  Left col   : Right hand
  Right col  : Left hand

Contact vertices are red, non-contact are steel-blue.
Mesh edges are drawn as thin lines for topology context.

Usage
-----
  # Single frame:
  python visualize_mano_contact.py --frame_id 200

  # Range of frames:
  python visualize_mano_contact.py --frame_id 200 --frame_id_end 250

  # Auto-discover all prediction frames:
  python visualize_mano_contact.py --all_frames

  # Override paths:
  python visualize_mano_contact.py --all_frames \\
      --preds_dir /other/path \\
      --gt_dir /other/gt \\
      --out_dir /results/vis

Cluster / headless usage
------------------------
Matplotlib backend is forced to Agg (no display needed).
"""

from __future__ import annotations

import os
import re
import glob
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PREDS_DIR      = "/gpfs/scratch1/shared/scur0805/arctic_data_front_100/contact_output"
DEFAULT_GT_DIR         = "/scratch-shared/scur0805/gt_seqs"
DEFAULT_PROCESSED_VERTS = "/home/scur0805/arctic/outputs/processed_verts/seqs/s01/box_use_01.npy"
DEFAULT_PKL_PATH       = "/scratch-shared/scur0805/InterFieldHands/MANO_SMPLX_vertex_ids.pkl"
DEFAULT_MANO_RIGHT_PKL = "/scratch-shared/scur0805/InterFieldHands/mano_v1_2/models/mano/MANO_RIGHT.pkl"
DEFAULT_MANO_LEFT_PKL  = "/scratch-shared/scur0805/InterFieldHands/mano_v1_2/models/mano/MANO_LEFT.pkl"
DEFAULT_OUT_DIR        = "./contact_vis_output"

DEFAULT_PRED_THR   = 0.3
DEFAULT_FRAME_OFF  = 1      # frame_id 200 → array row 199


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

class _ChumbyStub:
    """
    Dummy stand-in for any chumpy object encountered during unpickling.
    The original MANO .pkl was saved with chumpy arrays, but chumpy is
    broken on Python 3.11 (removed inspect.getargspec).  We only need
    the face array ('f'), which is a plain numpy array, so we stub out
    everything else and let numpy arrays deserialise normally.
    """
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _SafeUnpickler(pickle.Unpickler):
    """
    Redirect all chumpy classes to _ChumbyStub during unpickling.
    Subclasses pickle.Unpickler with encoding='latin1' to handle
    Python-2-pickled MANO files (bytes > 0x7f in string fields).
    """
    def __init__(self, f):
        # pickle.Unpickler.__init__ accepts encoding as of Python 3.x
        super().__init__(f, encoding="latin1")

    def find_class(self, module, name):
        if "chumpy" in module:
            return _ChumbyStub
        # Also handle scipy.sparse.csc (triggers DeprecationWarning on 3.11)
        if module in ("scipy.sparse.csc", "scipy.sparse.csr"):
            try:
                import scipy.sparse as sp
                return getattr(sp, name)
            except AttributeError:
                return _ChumbyStub
        return super().find_class(module, name)


def load_mano_faces(pkl_path: str) -> np.ndarray:
    """
    Load MANO face array (F, 3) int from a MANO .pkl model file.

    Uses a custom unpickler to bypass the chumpy / Python-3.11
    incompatibility (inspect.getargspec was removed in 3.11).
    encoding='latin1' handles Python-2-pickled byte strings.
    """
    with open(pkl_path, "rb") as f:
        model = _SafeUnpickler(f).load()

    for key in ("f", "faces"):
        if key in model:
            val = model[key]
            # chumpy arrays land as _ChumbyStub; fall through to next key
            if isinstance(val, _ChumbyStub):
                continue
            return np.asarray(val).astype(int)

    raise KeyError(
        f"Cannot find face array ('f' or 'faces') in {pkl_path}. "
        f"Available keys: {[k for k, v in model.items() if not isinstance(v, _ChumbyStub)]}"
    )


def load_hand_vertex_mapping(pkl_path: str):
    """Return (left_indices, right_indices) into the 10475-vert SMPL-X array."""
    with open(pkl_path, "rb") as f:
        mapping = pickle.load(f, encoding="latin1")
    left  = np.asarray(mapping["left_hand"]).astype(int)
    right = np.asarray(mapping["right_hand"]).astype(int)
    return left, right


def load_processed_verts(path: str) -> dict:
    """Load ARCTIC processed_verts .npy, handles nested-dict and flat-npz."""
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if raw.ndim == 0 else dict(raw)
    if "world_coord" not in data:
        wc = {k.replace("world_coord/", ""): v
              for k, v in data.items() if k.startswith("world_coord/")}
        if wc:
            data = {"world_coord": wc}
        else:
            raise KeyError(
                f"Cannot find 'world_coord' in {path}. "
                f"Keys: {list(data.keys())[:8]}"
            )
    return data


def get_frame_verts(data: dict, arr_idx: int):
    """
    Extract (verts_right, verts_left) for one frame.
    Returns (778, 3) arrays in world coords (metres).
    """
    wc = data["world_coord"]
    return wc["verts.right"][arr_idx], wc["verts.left"][arr_idx]


def load_gt_contact(gt_path: str):
    """Load GT boolean contact arrays. Returns (gt_right, gt_left) bool arrays."""
    d = np.load(gt_path)
    return d["right_contact"].astype(bool), d["left_contact"].astype(bool)


def load_pred_contact(npz_path: str,
                      left_idx: np.ndarray,
                      right_idx: np.ndarray,
                      threshold: float):
    """
    Load pred_contact_3d_smplx, apply MANO hand masks, binarise.
    Returns (pred_right, pred_left) bool arrays.
    """
    d = np.load(npz_path)
    scores = d["pred_contact_3d_smplx"].squeeze().astype(np.float32)
    pred_right = scores[right_idx] >= threshold
    pred_left  = scores[left_idx]  >= threshold
    return pred_right, pred_left


def discover_frame_ids(preds_dir: str) -> list[int]:
    """Return sorted list of frame IDs from *_hcontact_vertices.npz filenames."""
    ids = []
    for f in glob.glob(os.path.join(preds_dir, "*_hcontact_vertices.npz")):
        m = re.search(r"__(\d+)_hcontact_vertices\.npz$", os.path.basename(f))
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


# ──────────────────────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Colour scheme
COL_CONTACT     = "#e84545"    # red   — contact vertex
COL_NO_CONTACT  = "#4a90d9"    # steel blue — non-contact vertex
COL_EDGE        = "#333355"    # dark navy — mesh edges
COL_FACE_BASE   = "#1e1e3a"    # face fill base colour
ALPHA_FACE      = 0.18         # face transparency (low so vertices show through)
ALPHA_SCATTER   = 0.92
BG_COLOUR       = "#0d0d1a"
TEXT_COLOUR     = "white"


def _style_3d_ax(ax, title: str, xlabel: str, ylabel: str, zlabel: str):
    ax.set_facecolor(BG_COLOUR)
    ax.set_title(title, color=TEXT_COLOUR, fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color=TEXT_COLOUR, fontsize=7, labelpad=2)
    ax.set_ylabel(ylabel, color=TEXT_COLOUR, fontsize=7, labelpad=2)
    ax.set_zlabel(zlabel, color=TEXT_COLOUR, fontsize=7, labelpad=2)
    ax.tick_params(colors=TEXT_COLOUR, labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(True, color="#222244", linewidth=0.4, linestyle="--")


def draw_mano_hand(ax,
                   verts: np.ndarray,
                   faces: np.ndarray,
                   contact: np.ndarray,
                   hand_label: str,
                   elev: float = 20,
                   azim: float = -60):
    """
    Draw one MANO hand on a 3D axes.

    Parameters
    ----------
    verts   : (778, 3)  vertex positions in world coords
    faces   : (F, 3)    triangle indices
    contact : (778,) or (N,) bool  — contact flag per vertex.
              If len(contact) < 778, aligned by index (truncated).
    """
    n = min(len(verts), len(contact))
    contact_aligned = contact[:n]
    verts_used      = verts[:n]

    # ── Mesh faces (thin transparent layer for topology) ──────────────────────
    # Only include faces where all 3 vertices are within bounds
    valid_faces = faces[np.all(faces < n, axis=1)]
    tri_verts   = verts[valid_faces]   # (F, 3, 3)

    poly = Poly3DCollection(
        tri_verts, alpha=ALPHA_FACE,
        facecolor=COL_FACE_BASE, edgecolor=COL_EDGE, linewidth=0.15
    )
    ax.add_collection3d(poly)

    # ── Scatter vertices coloured by contact ──────────────────────────────────
    in_contact  = contact_aligned
    not_contact = ~contact_aligned

    if not_contact.sum() > 0:
        ax.scatter(
            verts_used[not_contact, 0],
            verts_used[not_contact, 1],
            verts_used[not_contact, 2],
            c=COL_NO_CONTACT, s=8, linewidths=0,
            alpha=ALPHA_SCATTER, depthshade=True, zorder=2
        )
    if in_contact.sum() > 0:
        ax.scatter(
            verts_used[in_contact, 0],
            verts_used[in_contact, 1],
            verts_used[in_contact, 2],
            c=COL_CONTACT, s=20, linewidths=0,
            alpha=0.98, depthshade=False, zorder=3    # red on top
        )

    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio workaround for mpl 3D
    extents = np.array([
        [verts[:, 0].min(), verts[:, 0].max()],
        [verts[:, 1].min(), verts[:, 1].max()],
        [verts[:, 2].min(), verts[:, 2].max()],
    ])
    centres = extents.mean(axis=1)
    half    = (extents[:, 1] - extents[:, 0]).max() / 2.0 + 1e-4
    ax.set_xlim(centres[0] - half, centres[0] + half)
    ax.set_ylim(centres[1] - half, centres[1] + half)
    ax.set_zlim(centres[2] - half, centres[2] + half)

    # Legend annotation
    n_contact = int(in_contact.sum())
    ax.text2D(
        0.02, 0.97,
        f"{hand_label}\nContact: {n_contact}/{n}\n"
        f"({n_contact/n*100:.1f}%)",
        transform=ax.transAxes,
        color=TEXT_COLOUR, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                  edgecolor="#444", alpha=0.7)
    )


def make_contact_figure(
    frame_id: int,
    verts_right: np.ndarray,
    verts_left: np.ndarray,
    faces_right: np.ndarray,
    faces_left: np.ndarray,
    contact_right: np.ndarray,
    contact_left: np.ndarray,
    source_label: str,        # "GT" or "Pred"
    extra_info: str = "",
) -> plt.Figure:
    """
    Build a 2×2 figure:
      Col 0 = Right hand,  Col 1 = Left hand
      Row 0 = front view (elev=20, azim=-60)
      Row 1 = side  view (elev=10, azim=30)
    """
    fig = plt.figure(figsize=(14, 10), facecolor=BG_COLOUR)
    fig.patch.set_facecolor(BG_COLOUR)

    views = [
        ("Front view", 20,  -60),
        ("Side view",  10,   30),
    ]
    hands = [
        ("Right hand", verts_right, faces_right, contact_right),
        ("Left  hand", verts_left,  faces_left,  contact_left),
    ]

    for row, (view_label, elev, azim) in enumerate(views):
        for col, (hand_label, verts, faces, contact) in enumerate(hands):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            ax.set_facecolor(BG_COLOUR)
            _style_3d_ax(
                ax,
                title=f"{view_label} — {hand_label}",
                xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)"
            )
            draw_mano_hand(ax, verts, faces, contact,
                           hand_label=hand_label, elev=elev, azim=azim)

    # Colour legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_CONTACT,
               markersize=8, label="Contact"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_NO_CONTACT,
               markersize=8, label="No contact"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.4, labelcolor=TEXT_COLOUR,
               edgecolor="#555", facecolor="#1a1a2e",
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"MANO Contact Visualisation  [{source_label}]  |  Frame {frame_id}\n"
        f"{extra_info}",
        fontsize=11, color=TEXT_COLOUR, y=0.98
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PER-FRAME RENDER
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(
    frame_id: int,
    arr_idx: int,
    pv_data: dict,
    faces_right: np.ndarray,
    faces_left: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    args,
    out_dir: str,
):
    # ── Vertex positions ──────────────────────────────────────────────────────
    verts_right, verts_left = get_frame_verts(pv_data, arr_idx)

    # ── GT contact ────────────────────────────────────────────────────────────
    gt_path = os.path.join(args.gt_dir, f"frame_{frame_id:05d}.npz")
    if not os.path.exists(gt_path):
        print(f"  WARN frame {frame_id}: GT file not found, skipping.")
        return False

    gt_right, gt_left = load_gt_contact(gt_path)

    gt_info = (
        f"GT thr=0.003 m  |  "
        f"R contact: {int(gt_right.sum())}/{len(gt_right)}  "
        f"L contact: {int(gt_left.sum())}/{len(gt_left)}"
    )
    fig_gt = make_contact_figure(
        frame_id,
        verts_right, verts_left,
        faces_right, faces_left,
        gt_right, gt_left,
        source_label="GT",
        extra_info=gt_info,
    )
    gt_out = os.path.join(out_dir, f"frame_{frame_id:05d}_gt.png")
    fig_gt.savefig(gt_out, dpi=130, bbox_inches="tight",
                   facecolor=fig_gt.get_facecolor())
    plt.close(fig_gt)

    # ── Pred contact ──────────────────────────────────────────────────────────
    pattern  = os.path.join(args.preds_dir, f"*__{frame_id:05d}_hcontact_vertices.npz")
    matches  = glob.glob(pattern)
    if not matches:
        print(f"  WARN frame {frame_id}: prediction file not found, skipping pred.")
        return False
    pred_path = matches[0]

    try:
        pred_right, pred_left = load_pred_contact(
            pred_path, left_idx, right_idx, args.pred_threshold
        )
    except (KeyError, ValueError) as e:
        print(f"  WARN frame {frame_id}: {e}")
        return False

    pred_info = (
        f"Pred thr={args.pred_threshold}  |  "
        f"R contact: {int(pred_right.sum())}/{len(pred_right)}  "
        f"L contact: {int(pred_left.sum())}/{len(pred_left)}"
    )
    fig_pred = make_contact_figure(
        frame_id,
        verts_right, verts_left,
        faces_right, faces_left,
        pred_right, pred_left,
        source_label="Pred",
        extra_info=pred_info,
    )
    pred_out = os.path.join(out_dir, f"frame_{frame_id:05d}_pred.png")
    fig_pred.savefig(pred_out, dpi=130, bbox_inches="tight",
                     facecolor=fig_pred.get_facecolor())
    plt.close(fig_pred)

    return True


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-frame MANO contact visualisation (GT and Pred, separate PNGs)."
    )
    parser.add_argument("--preds_dir",        type=str, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--gt_dir",           type=str, default=DEFAULT_GT_DIR)
    parser.add_argument("--processed_verts",  type=str, default=DEFAULT_PROCESSED_VERTS,
                        help="ARCTIC processed_verts .npy for world-coord vertex positions.")
    parser.add_argument("--pkl_path",         type=str, default=DEFAULT_PKL_PATH,
                        help="MANO_SMPLX_vertex_ids.pkl for hand vertex mapping.")
    parser.add_argument("--mano_right_pkl",   type=str, default=DEFAULT_MANO_RIGHT_PKL)
    parser.add_argument("--mano_left_pkl",    type=str, default=DEFAULT_MANO_LEFT_PKL)
    parser.add_argument("--out_dir",          type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pred_threshold",   type=float, default=DEFAULT_PRED_THR)
    parser.add_argument("--frame_offset",     type=int, default=DEFAULT_FRAME_OFF,
                        help="Subtracted from frame_id to get array row. Default: 1.")
    # Frame selection — mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame_id", type=int,
                       help="Render a single frame.")
    group.add_argument("--frame_range", type=int, nargs=2, metavar=("START", "END"),
                       help="Render frames START..END inclusive.")
    group.add_argument("--all_frames", action="store_true",
                       help="Auto-discover all prediction frames and render all.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("MANO Contact Visualisation")
    print("=" * 65)
    print(f"  Output dir : {args.out_dir}")
    print(f"  Pred thr   : {args.pred_threshold}")

    # ── Load static assets ────────────────────────────────────────────────────
    print("\nLoading MANO face topology...")
    faces_right = load_mano_faces(args.mano_right_pkl)
    faces_left  = load_mano_faces(args.mano_left_pkl)
    print(f"  Right hand faces : {len(faces_right)}")
    print(f"  Left  hand faces : {len(faces_left)}")

    print("\nLoading hand vertex mapping...")
    left_idx, right_idx = load_hand_vertex_mapping(args.pkl_path)
    print(f"  Right indices : {len(right_idx)}  Left indices : {len(left_idx)}")

    print("\nLoading processed_verts...")
    pv_data  = load_processed_verts(args.processed_verts)
    n_frames = pv_data["world_coord"]["verts.right"].shape[0]
    print(f"  Total frames in array : {n_frames}")

    # ── Resolve frame list ────────────────────────────────────────────────────
    if args.frame_id is not None:
        frame_ids = [args.frame_id]
    elif args.frame_range is not None:
        frame_ids = list(range(args.frame_range[0], args.frame_range[1] + 1))
    else:  # --all_frames
        frame_ids = discover_frame_ids(args.preds_dir)
        if not frame_ids:
            print(f"ERROR: no prediction files found in {args.preds_dir}")
            return
        print(f"\nAuto-discovered {len(frame_ids)} frames: "
              f"{frame_ids[0]} … {frame_ids[-1]}")

    print(f"\nRendering {len(frame_ids)} frame(s)...")
    ok = 0
    for i, frame_id in enumerate(frame_ids):
        arr_idx = frame_id - args.frame_offset
        if arr_idx < 0 or arr_idx >= n_frames:
            print(f"  SKIP frame {frame_id}: array row {arr_idx} "
                  f"out of range [0, {n_frames - 1}] — check --frame_offset.")
            continue

        if i % 10 == 0 or len(frame_ids) <= 20:
            print(f"  [{i+1:>4}/{len(frame_ids)}] frame {frame_id}", flush=True)

        success = render_frame(
            frame_id, arr_idx, pv_data,
            faces_right, faces_left,
            left_idx, right_idx,
            args, args.out_dir,
        )
        if success:
            ok += 1

    print(f"\nDone. {ok}/{len(frame_ids)} frames rendered to: {args.out_dir}")
    print("  Each frame produces two files:")
    print("    frame_NNNNN_gt.png   — ground truth contact")
    print("    frame_NNNNN_pred.png — predicted contact")
    print("=" * 65)


if __name__ == "__main__":
    main()