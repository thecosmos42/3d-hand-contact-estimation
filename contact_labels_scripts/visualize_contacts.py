#!/usr/bin/env python3
"""
visualize_contacts.py
=====================
Render ARCTIC contact labels as an MP4 video — no display or GPU required.
Uses OpenCV if available (robust), otherwise imageio+FFmpeg.
"""

import argparse
import sys
import warnings
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    sys.exit("mpl_toolkits is required (comes with matplotlib).")

# ---------------------------------------------------------------------------
# Try to import OpenCV; fall back to imageio
# ---------------------------------------------------------------------------
try:
    importlib.import_module('cv2')
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    try:
        import imageio
    except ImportError:
        sys.exit(
            "Neither OpenCV nor imageio is available. Install one:\n"
            "  pip install opencv-python\n"
            "  pip install imageio imageio-ffmpeg"
        )

if not HAS_CV2:
    try:
        import imageio_ffmpeg
        _ = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_CONTACT    = np.array([1.00, 0.15, 0.15])
COLOR_NO_CONTACT = np.array([0.20, 0.40, 0.90])
COLOR_OBJECT     = np.array([0.65, 0.65, 0.65])
BG_COLOR         = (0.10, 0.10, 0.10)


# ---------------------------------------------------------------------------
# Data loading (same as before)
# ---------------------------------------------------------------------------
def load_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=False))


def load_full_sequence(seq_path: Path) -> dict:
    """
    Load the processed .npy sequence file.
    Expected: data['world_coord']['verts.left'], ['verts.right'], ['verts.object']
    Returns dict with keys: left_verts, right_verts, obj_verts (each may be None).
    """
    try:
        raw = np.load(seq_path, allow_pickle=True).item()
        wc = raw["world_coord"]
    except Exception as exc:
        sys.exit(f"Failed to load {seq_path}: {exc}")

    key_map = {
        "left_verts":  "verts.left",
        "right_verts": "verts.right",
        "obj_verts":   "verts.object",
    }
    result = {}
    for logical, actual in key_map.items():
        if actual in wc:
            result[logical] = wc[actual]
        else:
            warnings.warn(f"Key '{actual}' missing in world_coord of {seq_path}")
            result[logical] = None
    return result


def load_npz_list(label_dir: Path) -> list:
    files = sorted(label_dir.glob("frame_*.npz"))
    if not files:
        sys.exit(f"No frame_*.npz files found in {label_dir}.")
    return files


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def vertex_colors_for_hand(contact_mask: np.ndarray) -> np.ndarray:
    return np.where(contact_mask[:, None], COLOR_CONTACT, COLOR_NO_CONTACT)


def face_colors_from_vertex_colors(vertex_colors: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertex_colors[faces].mean(axis=1)


def compute_scene_bounds(arrays: list) -> tuple:
    pts = np.concatenate([a for a in arrays if a is not None], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    centre = (lo + hi) / 2.0
    half = max((hi - lo).max() / 2.0 * 1.15, 1e-3)
    return centre, half


# ---------------------------------------------------------------------------
# Single‑frame renderer (updated with buffer_rgba)
# ---------------------------------------------------------------------------
def render_frame_to_rgb(
    left_verts: np.ndarray,
    right_verts,
    obj_verts,
    left_contact: np.ndarray,
    right_contact,
    mano_faces: np.ndarray,
    show_object: bool,
    show_right: bool,
    frame_label: str,
    image_size: tuple,
    dpi: int,
    elev: float,
    azim: float,
) -> np.ndarray:
    fig_w = image_size[0] / dpi
    fig_h = image_size[1] / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    bound_arrays = [left_verts]
    if show_right and right_verts is not None:
        bound_arrays.append(right_verts)
    if show_object and obj_verts is not None:
        bound_arrays.append(obj_verts)
    centre, half = compute_scene_bounds(bound_arrays)

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)

    # Object point cloud
    if show_object and obj_verts is not None:
        pts = obj_verts
        if pts.shape[0] > 4000:
            idx = np.random.choice(pts.shape[0], 4000, replace=False)
            pts = pts[idx]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=[COLOR_OBJECT.tolist()],
            s=1.5,
            depthshade=True,
            linewidths=0,
            alpha=0.55,
        )

    def _draw_hand(verts, contact_mask):
        vc = vertex_colors_for_hand(contact_mask)
        fc = face_colors_from_vertex_colors(vc, mano_faces)
        triangles = verts[mano_faces]
        poly = Poly3DCollection(
            triangles,
            facecolors=fc,
            edgecolors="none",
            alpha=0.95,
        )
        ax.add_collection3d(poly)

    _draw_hand(left_verts, left_contact)

    if show_right and right_verts is not None and right_contact is not None:
        _draw_hand(right_verts, right_contact)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_CONTACT.tolist(),
                   markersize=8, label="Contact"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_NO_CONTACT.tolist(),
                   markersize=8, label="No contact"),
    ]
    if show_object and obj_verts is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=COLOR_OBJECT.tolist(),
                       markersize=8, label="Object")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        framealpha=0.35,
        labelcolor="white",
        facecolor="#1e1e1e",
    )

    fig.text(0.5, 0.02, frame_label,
             ha="center", va="bottom", fontsize=9,
             color="white", alpha=0.75)

    # Rasterise – use buffer_rgba() to avoid deprecation warning
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb = rgba[:, :, :3]   # drop alpha
    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# Frame‑list builder
# ---------------------------------------------------------------------------
def collect_frames(npz_files: list, seq_data) -> list:
    frames = []
    for npz_path in npz_files:
        stem = npz_path.stem
        try:
            frame_idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            warnings.warn(f"Cannot parse frame index from '{npz_path.name}'; skipping.")
            continue
        if seq_data is not None:
            verts = {}
            for k in ("left_verts", "right_verts", "obj_verts"):
                arr = seq_data.get(k)
                if arr is not None and frame_idx < arr.shape[0]:
                    verts[k] = arr[frame_idx]
                else:
                    verts[k] = None
        else:
            verts = None
        frames.append((npz_path, frame_idx, verts))
    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ARCTIC contact labels as an MP4 video."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz_path", type=Path, help="Single .npz contact label file.")
    src.add_argument("--contact_label_dir", type=Path, help="Directory of frame_*.npz files.")
    parser.add_argument("--processed_seq_path", type=Path, default=None,
                        help="Original processed .npy sequence (if vertices not in .npz).")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--mano_faces_path", required=True, type=Path,
                        help="Path to mano_faces.npy (shape (1538,3)).")
    parser.add_argument("--output_mp4", required=True, type=Path,
                        help="Output .mp4 file.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--rotate", action="store_true",
                        help="Rotate camera 360° across the video.")
    parser.add_argument("--show_object", action="store_true")
    parser.add_argument("--show_right", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"Loading MANO face indices from {args.mano_faces_path} …")
    mano_faces = np.load(args.mano_faces_path).astype(np.int32)
    if mano_faces.shape != (1538, 3):
        warnings.warn(f"Expected MANO faces shape (1538,3), got {mano_faces.shape}.")

    if args.npz_path is not None:
        npz_files = [args.npz_path]
    else:
        npz_files = load_npz_list(args.contact_label_dir)
    print(f"Found {len(npz_files)} frame(s) to render.")

    seq_data = None
    if args.processed_seq_path is not None:
        print(f"Loading full sequence from {args.processed_seq_path} …")
        seq_data = load_full_sequence(args.processed_seq_path)

    if args.npz_path is not None and seq_data is not None:
        fi = args.frame_index
        verts = {
            k: (seq_data[k][fi] if seq_data.get(k) is not None else None)
            for k in ("left_verts", "right_verts", "obj_verts")
        }
        frames_meta = [(args.npz_path, fi, verts)]
    else:
        frames_meta = collect_frames(npz_files, seq_data)

    n_frames = len(frames_meta)
    if n_frames == 0:
        sys.exit("No renderable frames found.")

    if args.rotate and n_frames > 1:
        azim_values = [args.azim + 360.0 * i / n_frames for i in range(n_frames)]
    else:
        azim_values = [args.azim] * n_frames

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    image_size = (args.width, args.height)

    # ==================================================================
    #  Video writer setup (OpenCV preferred, imageio fallback)
    # ==================================================================
    if HAS_CV2:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'avc1'
        video = cv2.VideoWriter(str(args.output_mp4), fourcc, args.fps, image_size)
        if not video.isOpened():
            sys.exit(
                "OpenCV VideoWriter failed to open. "
                "Try loading FFmpeg module: 'module load ffmpeg'"
            )
        writer_type = "opencv"
    else:
        # imageio fallback (requires ffmpeg)
        writer_kwargs = dict(fps=args.fps, format='ffmpeg',
                             output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
        video = imageio.get_writer(str(args.output_mp4), **writer_kwargs)
        writer_type = "imageio"

    print(
        f"Writing MP4: {args.output_mp4}  "
        f"({args.width}×{args.height} @ {args.fps} fps) using {writer_type}"
    )

    try:
        for i, (npz_path, frame_idx, verts) in enumerate(frames_meta):
            if args.verbose or i % 50 == 0 or i == n_frames - 1:
                print(f"  frame {i+1:>5}/{n_frames}  ({npz_path.name})")

            try:
                npz = load_npz(npz_path)
            except Exception as exc:
                warnings.warn(f"Failed to load {npz_path}: {exc}. Skipping.")
                continue

            left_contact  = npz.get("left_contact")
            right_contact = npz.get("right_contact")
            if left_contact is None:
                warnings.warn(f"'left_contact' missing in {npz_path}. Skipping.")
                continue

            if verts is not None:
                left_verts  = verts.get("left_verts")
                right_verts = verts.get("right_verts")
                obj_verts   = verts.get("obj_verts")
            else:
                left_verts  = npz.get("left_verts")
                right_verts = npz.get("right_verts")
                obj_verts   = npz.get("obj_verts")

            if left_verts is None:
                warnings.warn(
                    f"left_verts not available for frame {frame_idx}. "
                    "Provide --processed_seq_path or regenerate with --include_verts."
                )
                continue

            rgb = render_frame_to_rgb(
                left_verts=left_verts,
                right_verts=right_verts,
                obj_verts=obj_verts,
                left_contact=left_contact,
                right_contact=right_contact,
                mano_faces=mano_faces,
                show_object=args.show_object,
                show_right=args.show_right,
                frame_label=f"Frame {frame_idx:05d}",
                image_size=image_size,
                dpi=args.dpi,
                elev=args.elev,
                azim=azim_values[i],
            )

            if writer_type == "opencv":
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                video.write(rgb_bgr)
            else:
                video.append_data(rgb)

    finally:
        if writer_type == "opencv":
            video.release()
        else:
            video.close()

    print(f"\nDone — MP4 saved to: {args.output_mp4}")


if __name__ == "__main__":
    main()
