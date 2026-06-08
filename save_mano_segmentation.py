#!/usr/bin/env python3
"""Save combined MANO vertex part segmentation (thumb, index, middle, ring, pinky, palm)."""
import argparse, pickle
import torch
import numpy as np
from pathlib import Path
from smplx import build_layer
from collections import defaultdict

NUM_VERTS_PER_HAND = 778

def load_mano_layer(mano_dir, is_rhand):
    return build_layer(
        model_path=str(mano_dir),
        model_type="mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    )

def create_one_hand_seg(mano_layer):
    with torch.no_grad():
        out = mano_layer()
        verts = out.vertices[0].cpu().numpy()   # (778, 3)
        joints = out.joints[0].cpu().numpy()    # (16, 3)  -- 16 MANO joints
    # Finger joint groups for 16-joint MANO:
    # 0: wrist
    # 1-3: thumb  (CMC, MCP, IP)
    # 4-6: index  (MCP, PIP, DIP)
    # 7-9: middle (MCP, PIP, DIP)
    # 10-12: ring (MCP, PIP, DIP)
    # 13-15: pinky (MCP, PIP, DIP)
    finger_joints = {
        "thumb":   [1,2,3],
        "index":   [4,5,6],
        "middle":  [7,8,9],
        "ring":    [10,11,12],
        "pinky":   [13,14,15],
    }
    dists = np.zeros((verts.shape[0], 5))
    for i, (finger, j_idxs) in enumerate(finger_joints.items()):
        fp = joints[j_idxs]               # (3,3)
        d = np.linalg.norm(verts[:, None, :] - fp[None, :, :], axis=2).min(axis=1)
        dists[:, i] = d
    wrist_d = np.linalg.norm(verts - joints[0], axis=1)
    part_ids = np.argmin(dists, axis=1)
    min_finger_dist = dists.min(axis=1)
    # assign palm where wrist is closer than any finger
    part_ids[wrist_d < min_finger_dist] = 5
    part_names = ["thumb", "index", "middle", "ring", "pinky"]
    seg = defaultdict(list)
    for i, pid in enumerate(part_ids):
        if pid == 5:
            seg["palm"].append(i)
        else:
            seg[part_names[pid]].append(i)
    return dict(seg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mano_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    mano_r = load_mano_layer(str(args.mano_dir), is_rhand=True).to(device)
    mano_l = load_mano_layer(str(args.mano_dir), is_rhand=False).to(device)

    seg_r = create_one_hand_seg(mano_r)
    seg_l = create_one_hand_seg(mano_l)

    # merge into combined indices (right first, left offset by 778)
    combined = {}
    for part in seg_r.keys():
        combined[part] = seg_r[part].copy()
        if part in seg_l:
            combined[part].extend([idx + NUM_VERTS_PER_HAND for idx in seg_l[part]])

    out_path = args.output_dir / 'mano_segmentation_combined.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"Segmentation saved to {out_path}")

if __name__ == '__main__':
    main()
