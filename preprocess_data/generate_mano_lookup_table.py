import argparse
import inspect
import os

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
for _legacy, _builtin in [("bool", bool), ("int", int), ("float", float),
                          ("complex", complex), ("object", object),
                          ("unicode", str), ("str", str)]:
    if not hasattr(np, _legacy):
        setattr(np, _legacy, _builtin)

import torch
from PIL import Image
from smplx import build_layer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from render_mesh_utils import (
    compute_vertex_normals,
    project_vertices_and_create_mask,
    render_mesh,
)

NUM_VERTS_PER_HAND = 778
HAND_SEPARATION_M = 0.15
IMAGE_SIZE = (1024, 1024)

VIEWS = {
    "palm":  (0.5, 0.0,   0.0, 0.0, 0.0),
    "back":  (0.5, 0.0, 180.0, 0.0, 0.0),
    "left":  (0.5, 0.0,  90.0, 0.0, 0.0),
    "right": (0.5, 0.0, 270.0, 0.0, 0.0),
}


def load_mano_layer(mano_dir, is_rhand):
    return build_layer(
        mano_dir,
        model_type="mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    )


def build_canonical_two_hand_mesh(mano_dir, device):
    mano_r = load_mano_layer(mano_dir, is_rhand=True).to(device)
    mano_l = load_mano_layer(mano_dir, is_rhand=False).to(device)

    verts_r = mano_r().vertices[0]
    verts_l = mano_l().vertices[0]

    shift_r = torch.tensor([+HAND_SEPARATION_M / 2, 0.0, 0.0], device=device)
    shift_l = torch.tensor([-HAND_SEPARATION_M / 2, 0.0, 0.0], device=device)
    verts_r = verts_r + shift_r
    verts_l = verts_l + shift_l

    verts = torch.cat([verts_r, verts_l], dim=0)
    assert verts.shape == (NUM_VERTS_PER_HAND * 2, 3), verts.shape

    faces_r = torch.from_numpy(mano_r.faces.astype(np.int64)).to(device)
    faces_l = torch.from_numpy(mano_l.faces.astype(np.int64)).to(device) + NUM_VERTS_PER_HAND
    faces = torch.cat([faces_r, faces_l], dim=0)

    normals = compute_vertex_normals(verts, faces)
    vertex_colors = (normals + 1.0) / 2.0

    return Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=vertex_colors.unsqueeze(0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mano-dir", required=True,
                        help="MANO_LEFT.pkl + MANO_RIGHT.pkl")
    parser.add_argument("--out-dir", default="data/hcontact_mano_rest")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    mesh = build_canonical_two_hand_mesh(args.mano_dir, device)
    num_verts = mesh.verts_packed().shape[0]
    num_faces = mesh.faces_packed().shape[0]
    print(f"Canonical mesh: {num_verts} verts, {num_faces} faces")

    pixel_to_vertex = {}
    bary_coords = {}
    dummy_contact = np.array([], dtype=np.int64)

    for view_name, cam_params in VIEWS.items():
        _, p2v, bary = project_vertices_and_create_mask(
            mesh, cam_params, dummy_contact,
            image_size=IMAGE_SIZE, device=device,
        )
        pixel_to_vertex[view_name] = p2v.astype(np.int64)
        bary_coords[view_name] = bary.astype(np.float32)

        debug_img = render_mesh(
            mesh, cam_params, light_location=(0.0, 0.5, 1.0),
            image_size=IMAGE_SIZE, device=device,
        )
        Image.fromarray(debug_img).save(
            os.path.join(args.out_dir, f"debug_{view_name}.png")
        )
        Image.fromarray(debug_img).save(
            os.path.join(args.out_dir, f"body_render_norm_{view_name}.png")
        )

        valid = (p2v >= 0).any(axis=-1).sum()
        total = p2v.shape[0] * p2v.shape[1]
        print(f"  {view_name:6s}  coverage={valid}/{total} ({100*valid/total:.1f}%)")

    np.savez(os.path.join(args.out_dir, "pixel_to_vertex_map_1024.npz"), **pixel_to_vertex)
    np.savez(os.path.join(args.out_dir, "bary_coords_map_1024.npz"), **bary_coords)

    print(f"\nLookup table:{args.out_dir}/")
    print(f"  Vertices = {num_verts}")
    print(f"  HUMAN_VIEW_DICT entry:'{os.path.basename(args.out_dir)}'")


if __name__ == "__main__":
    main()
