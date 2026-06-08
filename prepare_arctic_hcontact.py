#!/usr/bin/env python3
"""
prepare_arctic_hcontact.py – convert per-frame ARCTIC contact labels into
InterFieldHands training format.
Uses precomputed canonical mesh (no MANO or smplx needed here).
"""

import argparse, sys, pickle, warnings, os
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

# ---------- Arg parsing ----------
parser = argparse.ArgumentParser()
parser.add_argument('--processed_verts_dir', required=True, type=Path)
parser.add_argument('--contact_labels_dir', required=True, type=Path)
parser.add_argument('--images_dir', required=True, type=Path,
                    help='Root of egocentric images (e.g., outputs/egocentric/)')
parser.add_argument('--output_dir', required=True, type=Path)
parser.add_argument('--split', choices=['train','test'], required=True)
parser.add_argument('--canonical_mesh_dir', required=True, type=Path,
                    help='Directory with canonical_verts.npy and canonical_faces.npy')
parser.add_argument('--segmentation_path', required=True, type=Path,
                    help='Path to mano_segmentation_combined.pkl')
parser.add_argument('--interfieldhands_repo', required=True, type=Path,
                    help='Root of InterFieldHands (contains preprocess_data/)')
parser.add_argument('--frame_stride', type=int, default=1)
args = parser.parse_args()

# Insert repo root so we can import preprocess_data
sys.path.insert(0, str(args.interfieldhands_repo))
try:
    from preprocess_data.render_mesh_utils import project_vertices_and_create_mask
except ImportError as e:
    sys.exit(f"Failed to import render_mesh_utils: {e}\nMake sure PyTorch3D is properly installed and the repo path is correct.")

# ---------- Canonical mesh ----------
canonical_verts = torch.tensor(np.load(args.canonical_mesh_dir / 'canonical_verts.npy')).float()
canonical_faces = torch.tensor(np.load(args.canonical_mesh_dir / 'canonical_faces.npy')).long()
device = torch.device("cpu")
canonical_verts = canonical_verts.to(device)
canonical_faces = canonical_faces.to(device)

# ---------- Segmentation ----------
with open(args.segmentation_path, 'rb') as f:
    combined_seg = pickle.load(f)   # dict: part_name -> list of combined vertex indices

# ---------- Views ----------
VIEWS = {
    "palm":  (0.5, 0.0,   0.0, 0.0, 0.0),
    "back":  (0.5, 0.0, 180.0, 0.0, 0.0),
    "left":  (0.5, 0.0,  90.0, 0.0, 0.0),
    "right": (0.5, 0.0, 270.0, 0.0, 0.0),
}

def render_masks(contact_indices, obj_name, out_dir, base_name):
    from pytorch3d.structures import Meshes
    mesh = Meshes(verts=[canonical_verts], faces=[canonical_faces])
    contact_set = set(contact_indices)
    for view_name, cam_params in VIEWS.items():
        mask, _, _ = project_vertices_and_create_mask(
            mesh,
            cam_params,
            contact_set,
            image_size=(1024, 1024),
            min_vertices=1,
            device=device
        )
        save_path = out_dir / obj_name / f"{base_name}_{view_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), mask)

def get_obj_name(seq_stem):
    return seq_stem.split('_use_')[0]

# ---------- Main ----------
import os as _os; processed_npy_files = sorted(Path(r) / f for r, _, fs in _os.walk(args.processed_verts_dir, followlinks=True) for f in fs if f.endswith(".npy"))
if not processed_npy_files:
    sys.exit("No processed .npy files found.")

output_split = args.output_dir / args.split
images_out = output_split / "images"
images_out.mkdir(parents=True, exist_ok=True)

imgname_list = []
contact_list = []
body_parts_dict = {}

seq_info = []
for npy_path in processed_npy_files:
    data = np.load(npy_path, allow_pickle=True).item()
    wc = data['world_coord']
    num_frames = wc['verts.left'].shape[0]
    seq_stem = npy_path.stem
    seq_dir = args.contact_labels_dir / npy_path.relative_to(args.processed_verts_dir).parent / seq_stem
    seq_info.append((npy_path, seq_dir, seq_stem, num_frames))

total_frames = sum(s[-1] for s in seq_info)
pbar = tqdm(total=total_frames, desc=f"Processing {args.split}")

for npy_path, seq_dir, seq_stem, num_frames in seq_info:
    obj_name = get_obj_name(seq_stem)
    if not seq_dir.exists():
        pbar.update(num_frames)
        continue

    for frame_idx in range(0, num_frames, args.frame_stride):
        npz_path = seq_dir / f"frame_{frame_idx:05d}.npz"
        if not npz_path.exists():
            pbar.update(1)
            continue

        npz = np.load(npz_path)
        left_contact = npz["left_contact"]
        right_contact = npz["right_contact"]

        right_idx = np.where(right_contact)[0]
        left_idx = np.where(left_contact)[0] + 778
        combined_idx = np.concatenate([right_idx, left_idx]).tolist()

        # Image naming: ensure unique basenames
        img_base = f"frame_{frame_idx:05d}"
        unique_name = f"{seq_stem}_{img_base}.jpg"
        dst_img = images_out / unique_name
        subject = npy_path.relative_to(args.processed_verts_dir).parent
        src_img = args.images_dir / subject / seq_stem / "0" / f"{frame_idx+1:05d}.jpg"
        if src_img.exists():
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), cv2.imread(str(src_img)))
        else:
            warnings.warn(f"Image not found for {img_base}, using placeholder.")
        imgname_list.append(unique_name)

        # Contact entry
        contact_list.append({obj_name: combined_idx})

        # Body parts
        base_name_no_ext = os.path.splitext(unique_name)[0]
        loader_key = f"{base_name_no_ext}_{obj_name}"
        parts_touching = []
        for part, part_vertices in combined_seg.items():
            part_set = set(part_vertices)
            contact_set = set(combined_idx)
            intersection = len(part_set.intersection(contact_set))
            coverage = intersection / len(part_set) if part_set else 0
            if coverage >= 0.1:
                parts_touching.append(part)
        body_parts_dict[loader_key] = parts_touching

        # Render 4-view masks
        render_masks(combined_idx, obj_name, output_split / "hcontact_mano_rest", base_name_no_ext)

        pbar.update(1)

pbar.close()

# Save outputs
np.save(str(output_split / "imgname.npy"), np.array(imgname_list), allow_pickle=True)
with open(output_split / "contact_label_objectwise.pkl", 'wb') as f:
    pickle.dump(contact_list, f)
with open(output_split / "body_parts_objectwise.pkl", 'wb') as f:
    pickle.dump(body_parts_dict, f)

print(f"Done! Data written to {output_split}")
