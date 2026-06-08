#!/usr/bin/env python3
"""Save canonical combined MANO mesh (1556 vertices, 3076 faces) as .npy files."""
import argparse
import torch
import numpy as np
from pathlib import Path
from smplx import build_layer

NUM_VERTS_PER_HAND = 778
HAND_SEPARATION_M = 0.15

def load_mano_layer(mano_dir, is_rhand):
    return build_layer(
        model_path=str(mano_dir),
        model_type="mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mano_dir', required=True, type=Path, help='Path containing MANO_RIGHT.pkl, MANO_LEFT.pkl')
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    mano_r = load_mano_layer(str(args.mano_dir), is_rhand=True).to(device)
    mano_l = load_mano_layer(str(args.mano_dir), is_rhand=False).to(device)

    with torch.no_grad():
        verts_r = mano_r().vertices[0] + torch.tensor([+HAND_SEPARATION_M / 2, 0.0, 0.0])
        verts_l = mano_l().vertices[0] + torch.tensor([-HAND_SEPARATION_M / 2, 0.0, 0.0])

    verts = torch.cat([verts_r, verts_l], dim=0)
    assert verts.shape == (NUM_VERTS_PER_HAND * 2, 3), verts.shape

    faces_r = torch.from_numpy(mano_r.faces.astype(np.int64))
    faces_l = torch.from_numpy(mano_l.faces.astype(np.int64)) + NUM_VERTS_PER_HAND
    faces = torch.cat([faces_r, faces_l], dim=0)

    np.save(args.output_dir / 'canonical_verts.npy', verts.numpy().astype(np.float32))
    np.save(args.output_dir / 'canonical_faces.npy', faces.numpy().astype(np.int64))
    print(f"Saved canonical_verts.npy {verts.shape} and canonical_faces.npy {faces.shape}")

if __name__ == '__main__':
    main()
