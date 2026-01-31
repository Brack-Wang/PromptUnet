import os
import zarr
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import random
import json
from datetime import datetime

try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    # 新版没有 skeletonize_3d，用 skeletonize 替代（对3D兼容较弱，但能跑通）
    from skimage.morphology import skeletonize as skeletonize_3d


# ========== Configuration ==========
input_dir = "/data/wangfeiran/code/brainbow/datasets/fisbe/combined_data_500"
output_base_dir = "/data/wangfeiran/code/brainbow/datasets/fisbe/nnunet/nnUNet_rawdata/Dataset010_nninteractive8C_reordered"
output_image_dir = os.path.join(output_base_dir, "imagesTr")
output_label_dir = os.path.join(output_base_dir, "labelsTr")
progress_dir = os.path.join(output_base_dir, "_progress")

number_of_samples = 500
min_voxels_per_inst = 50
prev_mask_use_prob = 0.60
prev_mask_coverage_range = (0.10, 0.70)
add_bg_noise = True
noise_intensity = 5.0

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

# ========== Helper functions ==========

def sample_points(mask, n_points):
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = np.random.choice(coords.shape[0], min(n_points, coords.shape[0]), replace=False)
    pts = np.zeros_like(mask, dtype=np.uint8)
    for p in coords[idx]:
        pts[tuple(p)] = 1
    return pts

def skeleton_points(mask, n_range=(2, 4)):
    try:
        sk = skeletonize_3d(mask.astype(bool))
        coords = np.argwhere(sk)
        if coords.size == 0:
            return sample_points(mask, random.randint(*n_range))
        n = min(random.randint(*n_range), len(coords))
        idx = np.random.choice(len(coords), n, replace=False)
        pts = np.zeros_like(mask, dtype=np.uint8)
        for p in coords[idx]:
            pts[tuple(p)] = 1
        return pts
    except Exception:
        return sample_points(mask, random.randint(*n_range))

def mixed_fg_points(inst_mask):
    r = random.random()
    if r < 0.4:
        return skeleton_points(inst_mask, (15, 30))
    elif r < 0.8:
        return skeleton_points(inst_mask, (8, 15))
    elif r < 0.95:
        return sample_points(inst_mask, random.randint(5, 10))
    else:
        try:
            cen = binary_erosion(inst_mask, iterations=2)
            if cen.sum() > 0:
                return sample_points(cen, random.randint(8, 15))
        except Exception:
            pass
        return sample_points(inst_mask, random.randint(5, 10))

def strategic_bg_points(inst_mask, all_instances):
    other = (all_instances > 0) & (inst_mask == 0)
    empty = (all_instances == 0)
    weight = other.astype(float) * 0.8 + empty.astype(float) * 0.2
    n_bg = random.randint(2, 5)
    if weight.sum() == 0:
        return np.zeros_like(inst_mask, dtype=np.uint8)
    coords = np.argwhere(weight > 0)
    probs = weight[weight > 0].astype(np.float64)
    probs = probs / probs.sum()
    n = min(n_bg, len(coords))
    idx = np.random.choice(len(coords), n, replace=False, p=probs)
    pts = np.zeros_like(inst_mask, dtype=np.uint8)
    for i in idx:
        pts[tuple(coords[i])] = 1
    return pts

def dist_to_softmap(d):
    return (1.0 / (1.0 + d)).astype(np.float32)

def gen_point_prompt_maps(inst_mask, all_instances):
    fg_pts = mixed_fg_points(inst_mask)
    bg_pts = strategic_bg_points(inst_mask, all_instances)
    fg_soft = dist_to_softmap(distance_transform_edt(1 - fg_pts))
    bg_soft = dist_to_softmap(distance_transform_edt(1 - bg_pts))
    return fg_soft, bg_soft

def make_coarse_prev_mask(gt_mask, coverage_range=(0.5, 0.67)):
    if gt_mask.sum() == 0:
        return gt_mask.astype(np.float32)
    dist_in = distance_transform_edt(gt_mask > 0)
    total = (gt_mask > 0).sum()
    target_ratio = random.uniform(*coverage_range)
    k = int(max(1, round(total * target_ratio)))
    flat_idx = np.argsort(dist_in.ravel())[::-1][:k]
    coarse = np.zeros_like(gt_mask, dtype=np.uint8)
    coarse.ravel()[flat_idx] = 1
    return coarse.astype(np.float32)

# ========== Main ==========

sample_counter = 0
for idx in tqdm(range(number_of_samples), desc="Processing zarr files"):
    zarr_name = f"combined_{idx:03d}.zarr"
    zarr_path = os.path.join(input_dir, zarr_name, "volumes")
    raw_path = os.path.join(zarr_path, "raw")
    gt_path = os.path.join(zarr_path, "gt_instances")

    try:
        raw = zarr.load(raw_path).astype(np.float32)
        gt_instance = zarr.load(gt_path).astype(np.uint16)
    except Exception as e:
        print(f"[Error] {zarr_name}: {e}")
        continue

    if raw.ndim == 4:
        raw = raw[0]
    if gt_instance.ndim == 4:
        gt_merged = np.zeros_like(gt_instance[0], dtype=np.uint16)
        cur_id = 1
        for i in range(gt_instance.shape[0]):
            m = gt_instance[i] > 0
            gt_merged[m] = cur_id
            cur_id += 1
    elif gt_instance.ndim == 3:
        gt_merged = gt_instance
    else:
        continue

    raw_used = raw.copy() if add_bg_noise else raw
    if add_bg_noise:
        bg = (gt_merged == 0)
        noise = np.random.normal(0, noise_intensity, raw.shape).astype(np.float32)
        raw_used[bg] = np.clip(raw_used[bg] + noise[bg], raw.min(), raw.max())

    instance_ids = np.unique(gt_merged)
    instance_ids = instance_ids[instance_ids != 0]

    for inst_id in instance_ids:
        if sample_counter >= number_of_samples:
            break

        inst_mask = (gt_merged == inst_id).astype(np.uint8)
        if inst_mask.sum() < min_voxels_per_inst:
            continue

        fg_soft, bg_soft = gen_point_prompt_maps(inst_mask, gt_merged)
        zeros = np.zeros_like(inst_mask, dtype=np.float32)
        prev_mask = make_coarse_prev_mask(inst_mask, prev_mask_coverage_range) \
            if random.random() < prev_mask_use_prob else zeros

        # === New 8-channel order ===
        ch = [
            raw_used.astype(np.float32),  # 0 image
            prev_mask.astype(np.float32),  # 1 coarse
            zeros, zeros,                  # 2,3 box+/-
            fg_soft.astype(np.float32),    # 4 point+
            bg_soft.astype(np.float32),    # 5 point-
            zeros, zeros                   # 6,7 scribble+/-
        ]

        label_data = inst_mask.astype(np.uint8)
        case_id = f"fisbe_{sample_counter:05d}"

        for c_idx, arr in enumerate(ch):
            nib.save(nib.Nifti1Image(arr, affine=np.eye(4)),
                     os.path.join(output_image_dir, f"{case_id}_{c_idx:04d}.nii.gz"))
        nib.save(nib.Nifti1Image(label_data, affine=np.eye(4)),
                 os.path.join(output_label_dir, f"{case_id}.nii.gz"))

        sample_counter += 1
    if sample_counter >= number_of_samples:
        break

print(f"✅ Done! Generated {sample_counter} reordered 8-channel samples.")
print(f"Images -> {output_image_dir}")
print(f"Labels -> {output_label_dir}")

# ========== Dataset.json ==========
dataset_json = {
    "channel_names": {
        "0": "image",
        "1": "coarse_segmentation",
        "2": "bbox_pos",
        "3": "bbox_neg",
        "4": "point_pos",
        "5": "point_neg",
        "6": "scribble_pos",
        "7": "scribble_neg"
    },
    "labels": {"background": 0, "neuron": 1},
    "numTraining": sample_counter,
    "file_ending": ".nii.gz"
}

with open(os.path.join(output_base_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print("✅ dataset.json written.")
