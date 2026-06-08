# 3D Hand Contact Estimation from 2D Foundation Models

This is the code for our project where we adapted [InteractVLM](https://interactvlm.is.tue.mpg.de/) so that it predicts hand-object contact on the MANO hand mesh instead of full-body contact on SMPL. We fine-tuned it on the [ARCTIC](https://arctic.is.tue.mpg.de/) dataset.

![Demo](assets/demo.gif)

## Requirements

- Python 3.10 or newer with CUDA 12.1
- `pip install -r requirements.txt`, plus PyTorch3D and DeepSpeed
- The MANO models from <https://mano.is.tue.mpg.de>. Put `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `mano_v1_2/models/mano/`.
- The ARCTIC dataset from <https://arctic.is.tue.mpg.de>.
- The LISA-13B checkpoint from HuggingFace (`xinlai/LISA-13B-llama2-v1`).
- The SAM ViT-H checkpoint, which you can [download here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Put it in `data/sam_vit_h_4b8939.pth`.

## Running

### 1. Process ARCTIC
First run ARCTIC's `process_seqs.py` script to get the MANO and object vertices for every frame. We had to patch it a bit because it expects MANO to return 21 joints but the original MANO only gives 16. The fix is to add the five fingertip vertices to the output.

### 2. Make the canonical mesh and lookup table
```bash
python save_mano_canonical_mesh.py
python preprocess_data/generate_mano_lookup_table.py
```
This builds the combined left+right MANO mesh and the lookup table that we use to go from 2D mask predictions back to 3D vertex labels.

### 3. Generating contact labels
```bash
python contact_labels_scripts/generate_contact_labels.py \
    --processed_verts_dir <ARCTIC_processed_verts> \
    --output_dir <CONTACT_LABELS_OUT>
```
A vertex is marked as in contact if it sits within 3 mm of the nearest object vertex.

### 4. Training data preparation
```bash
python prepare_arctic_hcontact.py \
    --processed_verts_dir <ARCTIC_processed_verts> \
    --contact_labels_dir  <CONTACT_LABELS_OUT> \
    --images_dir          <ARCTIC_cropped_images> \
    --output_dir          <HCONTACT_ARCTIC_TRAIN> \
    --split train \
    --canonical_mesh_dir  data/mano_canonical \
    --segmentation_path   data/mano_canonical/mano_segmentation_combined.pkl \
    --interfieldhands_repo .
```
Then symlink the output to `data/arctic/train`, and do the same for `data/arctic/test` with your held-out split.

### 5. Train
```bash
bash scripts/run_train.sh hcontact-arctic
```
This uses the same settings as in the report: LoRA with r=8 and α=16, batch size 8, learning rate 3e-4 with cosine annealing, 30 epochs of 500 steps each.

### 6. Baseline and figures
`final_baseline/eval.py` gives you the baseline numbers in Table 1, and `final_baseline/viz.py` is what we used to make the qualitative figures (Figures 2 and 3).

## Changes from the original InteractVLM

- `scripts/run_train.sh` – added the `arctic_hcontact` config
- `datasets/hcontact_3d.py` – added the ARCTIC dataset loader
- `preprocess_data/constants.py` – added the `4MV-Z_MANO_Both` view
- `save_mano_canonical_mesh.py` – builds the combined left+right MANO mesh with a small gap between the two hands
- `prepare_arctic_hcontact.py` – new script for the ARCTIC training data
- `contact_labels_scripts/generate_contact_labels.py` – new script for the 3 mm contact labels
- `utils/eval_utils.py` – stubbed the geodesic distance metric, since the SMPL distance matrix doesn't transfer to MANO

## Credits

Built on top of [InteractVLM](https://github.com/saidwivedi/InteractVLM) by Dwivedi et al. and the [ARCTIC](https://arctic.is.tue.mpg.de/) dataset by Fan et al.
