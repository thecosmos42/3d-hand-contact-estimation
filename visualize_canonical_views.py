"""
visualize_canonical_views.py
============================
Render four canonical views (palm, back, left, right) of the combined MANO
hand mesh used by InterFieldHands (configuration: 4MV-Z_MANO_Both).

BACKGROUND — why the raw mesh has overlapping hands
----------------------------------------------------
The canonical combined-hand mesh is NOT a spatially separated model of two
hands placed side by side.  It is a *vertex-index-based template*: the right
hand occupies indices 0–777 and the left hand occupies indices 778–1555, but
BOTH hands are placed at the exact same location in 3D space (centred at the
origin).  The left hand is the mirror-image of the right hand, reflected in X.

This is intentional for the InterFieldHands pipeline: during training and
inference the mesh is rendered from fixed camera angles and the 2-D pixel
positions are looked up in pre-computed tables (pixel_to_vertex_map,
bary_coords_map) to lift 2-D contact predictions back to 3-D vertex labels.
The two hands share the same geometric "shell"; they are distinguished only by
vertex index, not by position.  In the palm and back views this overlap is
invisible because the meshes are identical in silhouette.  In the left and
right side views you see the characteristic "butterfly" spread — both hands
superimposed — which is technically correct but visually confusing for reports.

THE --separate FLAG
-------------------
When --separate is set the script copies the vertex array and adds
[+0.25, 0.0, 0.0] to all left-hand vertices (indices 778–1555).

Why +0.25?
  Right-hand X spans [-0.079, +0.114] (width ≈ 0.193 m).
  Left-hand  X spans [-0.114, +0.079] (same width, mirrored).
  Shifting the left hand by +0.25 moves its X range to [+0.136, +0.329].
  The gap between the right-hand maximum (+0.114) and the shifted left-hand
  minimum (+0.136) is ≈ 0.022 m — a clean, visible separation without
  wasting excessive figure space.

This separation is purely cosmetic for report figures.  The lookup tables and
training pipeline always use the ORIGINAL unseparated vertex positions.

Usage
-----
  # Overlapping (canonical, as used in training):
  python visualize_canonical_views.py \\
      --verts canonical_verts.npy \\
      --faces canonical_faces.npy \\
      --output_dir renders/

  # Visually separated (for report figures):
  python visualize_canonical_views.py \\
      --verts canonical_verts.npy \\
      --faces canonical_faces.npy \\
      --output_dir renders/ \\
      --separate \\
      --dpi 200

Dependencies: numpy, matplotlib only.  No torch, pytorch3d, or GPU required.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive, safe for servers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
# InterFieldHands 4MV-Z_MANO_Both camera configuration.
# Elevation is 0° for all views (horizontal belt around the hand).
VIEWS = [
    ("palm",  0,   0),   # front / palmar surface
    ("back",  180, 0),   # dorsal surface
    ("left",  90,  0),   # lateral / thumb side
    ("right", 270, 0),   # medial / little-finger side
]

# Grid layout: (row, col) for each view in the 2×2 subplot
GRID_POS = {
    "palm":  (0, 0),
    "back":  (0, 1),
    "left":  (1, 0),
    "right": (1, 1),
}

# Skin-tone base colour (R, G, B) in [0, 1].
# A warm mid-tone that reads well on white backgrounds.
BASE_COLOUR = np.array([0.95, 0.75, 0.65])

# Ambient light fraction — prevents fully-dark back-faces.
AMBIENT = 0.30

# Specular highlight strength (simple Phong-style).
SPECULAR_STRENGTH = 0.18
SPECULAR_SHININESS = 12

# Left-hand vertex separation offset (X axis, metres).
SEPARATE_OFFSET = np.array([0.25, 0.0, 0.0], dtype=np.float32)


# ─────────────────────────────────────────────
#  Mesh utilities
# ─────────────────────────────────────────────

def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute one unit normal per triangle face.

    Parameters
    ----------
    verts : (V, 3) float32
    faces : (F, 3) int64

    Returns
    -------
    face_normals : (F, 3) float32
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                  # (F, 3)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)       # guard degenerate faces
    return (fn / norms).astype(np.float32)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray,
                           face_normals: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals by averaging the normals of all adjacent faces.
    Uses numpy.add.at for an accumulate-by-index operation (handles non-manifold
    geometry gracefully).

    Parameters
    ----------
    verts        : (V, 3)
    faces        : (F, 3)
    face_normals : (F, 3)

    Returns
    -------
    vertex_normals : (V, 3) float32, unit-length
    """
    vn = np.zeros_like(verts)
    for i in range(3):                                 # accumulate per corner
        np.add.at(vn, faces[:, i], face_normals)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (vn / norms).astype(np.float32)


def camera_light_direction(azim_deg: float, elev_deg: float) -> np.ndarray:
    """
    Convert Matplotlib's (azim, elev) camera angles to a unit light direction
    vector pointing FROM the scene TOWARD the camera (i.e. "light from the
    camera" — a simple head-light model that always illuminates visible faces).

    Matplotlib 3D places the camera at:
        x = r * cos(elev) * sin(azim)   (note: sin, not cos, for azim)
        y = r * cos(elev) * cos(azim)
        z = r * sin(elev)
    The light direction is the unit vector from origin to that point.

    Parameters
    ----------
    azim_deg, elev_deg : float — degrees

    Returns
    -------
    light_dir : (3,) float64, unit vector
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    d = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ])
    return d / np.linalg.norm(d)


def shade_faces(face_normals: np.ndarray,
                light_dir: np.ndarray) -> np.ndarray:
    """
    Compute per-face RGBA colours using a Phong-inspired shading model:

        colour = ambient * base
               + diffuse  * max(N·L, 0) * base
               + specular * max(N·L, 0)^shininess * white

    All faces use the same base skin colour (BASE_COLOUR).  The result is
    clamped to [0, 1] and returned as (F, 4) RGBA with alpha=1.

    Parameters
    ----------
    face_normals : (F, 3) — unit normals, world space
    light_dir    : (3,)   — unit vector toward the light (= toward camera)

    Returns
    -------
    colours : (F, 4) float32
    """
    # Diffuse: Lambert cosine term
    diffuse = np.clip(face_normals @ light_dir, 0.0, 1.0)   # (F,)

    # Specular: simplified Phong — reflection is N*2*(N·L) - L but for a
    # head-light the specular highlight is just (N·L)^shininess
    specular = diffuse ** SPECULAR_SHININESS                  # (F,)

    # Compose
    diff_fraction = 1.0 - AMBIENT
    rgb = (
        AMBIENT              * BASE_COLOUR[None, :]           # ambient
        + diff_fraction      * diffuse[:, None] * BASE_COLOUR # diffuse
        + SPECULAR_STRENGTH  * specular[:, None]               # specular (white)
    )                                                          # (F, 3)

    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.ones((len(rgb), 1), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1).astype(np.float32)


# ─────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────

def set_equal_aspect(ax, verts: np.ndarray) -> None:
    """
    Force equal aspect ratio on a 3-D Axes by computing the bounding sphere
    of the mesh and setting all three axis limits to that sphere's diameter.

    Matplotlib 3D does not support ax.set_aspect('equal') natively before
    version 3.6, and even then it can behave unexpectedly.  This approach
    works across versions.

    Parameters
    ----------
    ax    : Axes3D
    verts : (V, 3) — the (possibly translated) vertex array
    """
    centre = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    half   = (verts.max(axis=0) - verts.min(axis=0)).max() / 2.0
    # Add a small margin so the mesh doesn't clip the axes border
    half  *= 1.15

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)

    # set_box_aspect is available in matplotlib ≥ 3.3
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])


def render_view(ax, verts: np.ndarray, faces: np.ndarray,
                face_normals: np.ndarray,
                view_name: str, azim: float, elev: float) -> None:
    """
    Render the mesh into a single Axes3D at the given camera orientation.

    Steps
    -----
    1. Compute per-face colours from the head-light shading model.
    2. Build a Poly3DCollection from the triangle vertex triplets.
    3. Set camera orientation, equal aspect, and hide axes/panes.

    Parameters
    ----------
    ax           : Axes3D
    verts        : (V, 3) — possibly translated vertex array
    faces        : (F, 3) — face index array
    face_normals : (F, 3) — pre-computed unit face normals
    view_name    : str    — title label
    azim, elev   : float  — camera angles in degrees
    """
    light_dir = camera_light_direction(azim, elev)
    colours   = shade_faces(face_normals, light_dir)  # (F, 4)

    # Build triangle vertex triplets: (F, 3, 3)
    triangles = verts[faces]                           # (F, 3, 3)

    # Poly3DCollection renders a set of polygons with per-polygon colours.
    # zsort='average' sorts polygons by their centroid depth — correct for
    # convex-ish meshes without requiring a proper z-buffer.
    poly = Poly3DCollection(
        triangles,
        facecolors=colours,
        edgecolors="none",   # no wireframe — cleaner shaded appearance
        alpha=1.0,
        zsort="average",
    )
    ax.add_collection3d(poly)

    # Camera orientation
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio so the hand isn't distorted
    set_equal_aspect(ax, verts)

    # Remove all axis decorations
    ax.set_axis_off()

    # Remove grey background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)

    ax.set_title(view_name.capitalize(), fontsize=13, pad=4, fontweight="normal")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 4 canonical views of the combined MANO hand mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--verts", required=True,
                   help="Path to canonical_verts.npy  (1556, 3) float32")
    p.add_argument("--faces", required=True,
                   help="Path to canonical_faces.npy  (3076, 3) int64")
    p.add_argument("--output_dir", required=True,
                   help="Directory in which output JPEGs are saved")
    p.add_argument("--separate", action="store_true",
                   help=("Translate left-hand vertices (indices 778–1555) by "
                         "+0.25 in X so both hands are visually separated.  "
                         "For report figures only — does NOT affect the "
                         "training pipeline."))
    p.add_argument("--dpi", type=int, default=150,
                   help="Output resolution in dots-per-inch (default: 150)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load mesh ─────────────────────────────────────────────────────────
    print(f"Loading verts from : {args.verts}")
    print(f"Loading faces from : {args.faces}")
    verts = np.load(args.verts).astype(np.float32)   # (1556, 3)
    faces = np.load(args.faces).astype(np.int64)      # (3076, 3)

    if verts.shape != (1556, 3):
        sys.exit(f"[ERROR] Expected verts shape (1556, 3), got {verts.shape}")
    if faces.shape != (3076, 3):
        sys.exit(f"[ERROR] Expected faces shape (3076, 3), got {faces.shape}")
    if faces.max() > 1555 or faces.min() < 0:
        sys.exit(f"[ERROR] Face indices out of range [0, 1555]")

    print(f"Mesh loaded OK — {len(verts)} vertices, {len(faces)} faces")
    print(f"Right hand centroid (verts 0–777):    "
          f"{verts[:778].mean(axis=0).round(4)}")
    print(f"Left  hand centroid (verts 778–1555): "
          f"{verts[778:].mean(axis=0).round(4)}")

    # ── 2. Optional: separate hands ──────────────────────────────────────────
    if args.separate:
        # Work on a copy — never mutate the original array.
        verts = verts.copy()
        verts[778:] += SEPARATE_OFFSET
        print(f"--separate: left hand (778–1555) shifted by "
              f"{SEPARATE_OFFSET.tolist()} in world space")
        print(f"  Right hand X range: [{verts[:778,0].min():.3f}, "
              f"{verts[:778,0].max():.3f}]")
        print(f"  Left  hand X range: [{verts[778:,0].min():.3f}, "
              f"{verts[778:,0].max():.3f}]")
    else:
        print("--separate not set: both hands rendered at origin (overlapping)")

    # ── 3. Pre-compute normals (view-independent) ─────────────────────────────
    print("Computing face and vertex normals …")
    face_normals = compute_face_normals(verts, faces)   # (3076, 3)
    # vertex_normals computed but not used directly in Poly3DCollection
    # (per-face shading is used instead — it gives crisper edges on low-poly
    # meshes like MANO's 3076 faces)
    _ = compute_vertex_normals(verts, faces, face_normals)
    print("Normals ready.")

    # ── 4. Output directory ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 5. Combined 2×2 figure ───────────────────────────────────────────────
    print("Rendering 2×2 combined figure …")
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    for view_name, azim, elev in VIEWS:
        row, col = GRID_POS[view_name]
        # subplot_index is 1-based, row-major
        idx = row * 2 + col + 1
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("white")
        render_view(ax, verts, faces, face_normals, view_name, azim, elev)

    fig.suptitle(
        "Canonical MANO Hand Mesh — 4MV-Z_MANO_Both",
        fontsize=11,
        y=0.98,
        color="#444444",
    )
    if args.separate:
        fig.text(
            0.5, 0.01,
            "Left hand translated +0.25 in X for visual separation (report figure only)",
            ha="center", fontsize=8, color="#888888",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    combined_path = os.path.join(args.output_dir, "canonical_views_all.jpg")
    fig.savefig(combined_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", format="jpeg",
                pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    print(f"  Saved: {combined_path}")

    # ── 6. Four individual figures ───────────────────────────────────────────
    for view_name, azim, elev in VIEWS:
        print(f"Rendering individual view: {view_name} …")
        fig_single = plt.figure(figsize=(5, 4))
        fig_single.patch.set_facecolor("white")
        ax = fig_single.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        render_view(ax, verts, faces, face_normals, view_name, azim, elev)

        if args.separate:
            fig_single.text(
                0.5, 0.01,
                "Left hand +0.25 X offset (report figure only)",
                ha="center", fontsize=7.5, color="#888888",
            )

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out_path = os.path.join(args.output_dir, f"{view_name}_view.jpg")
        fig_single.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                           facecolor="white", format="jpeg",
                           pil_kwargs={"quality": 95, "subsampling": 0})
        plt.close(fig_single)
        print(f"  Saved: {out_path}")

    print("\nDone.  All renders saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
