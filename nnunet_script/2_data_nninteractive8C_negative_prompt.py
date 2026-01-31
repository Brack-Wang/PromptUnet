"""
nnUNet Interactive Segmentation Data Preparation - Improved Version
====================================================================

BASE SCRIPT: Please refer to the original script which already provide code to 
transform data from raw images to data for nnunet with 8 channels:
    /data/wangfeiran/code/brainbow/new/nnunet_script/2_data_nninteractive8C3.py


IMPROVEMENT OBJECTIVES:
    
    1. NEGATIVE PROMPT SAMPLING
       - Target: 100% from other neurons only (exclude empty background entirely)
       - Rationale: Simulates realistic user annotation where negative clicks 
         indicate "not this neuron" rather than clicking empty space
    
    2. POSITIVE PROMPT CONCENTRATION
       - Current: Points distributed across entire neuron skeleton (dispersed)
       - Target: Points spatially clustered in 1-3 local regions per neuron
       - Rationale: Mimics real human annotation behavior where annotators 
         click densely in local areas rather than uniformly across structure
       - Expected outcome: Model learns to infer full structure from localized prompts
    
    3. PROMPT QUANTITY
       - Current: 15-30 positive points, 2-5 negative points
       - Target: ~20-35 positive points (clustered), ~10-20 negative points
       - Note: Maintain positive > negative ratio to prioritize target identification

IMPLEMENTATION NOTES:
    - Handle edge cases: small neurons, insufficient neighbor neurons
    - Preserve all other functionality from base script
    - Keep the same 8-channel output format (channels 2-3, 6-7 remain zeros)
    
Author: Scott
Date: 11/18 - 11/19
"""

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
output_base_dir = "/data/wangfeiran/code/brainbow/datasets/fisbe/nnunet/nnUNet_rawdata/Dataset011_nninteractive8C_negative_prompt"
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

def get_clustered_points(mask, n_points, n_clusters=None):
    coords = np.argwhere(mask)
    pts = np.zeros_like(mask, dtype=np.uint8)

    if coords.shape[0] == 0 or n_points <= 0:
        return pts

    n_points = min(n_points, coords.shape[0])

    if n_clusters is None:
        n_clusters = random.randint(1, 3)
    n_clusters = min(n_clusters, n_points, coords.shape[0])

    points_left = n_points

    seed_indices = np.random.choice(coords.shape[0], n_clusters, replace=False)
    seeds = coords[seed_indices]

    for i, seed in enumerate(seeds):
        if points_left <= 0:
            break

        clusters_left = n_clusters - i
        k_base = max(1, points_left // clusters_left)

        dists = np.linalg.norm(coords - seed, axis=1)

        candidate_pool_size = min(k_base * 3, coords.shape[0])
        nearest_indices = np.argsort(dists)[:candidate_pool_size]

        k = min(k_base, candidate_pool_size, points_left)
        if k <= 0:
            continue

        chosen = np.random.choice(nearest_indices, k, replace=False)

        for idx in chosen:
            coord = tuple(coords[idx])
            if pts[coord] == 0:
                pts[coord] = 1
                points_left -= 1
                if points_left <= 0:
                    break

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
    n_points = random.randint(20, 35)
    n_clusters = random.randint(1, 3)
    return get_clustered_points(inst_mask > 0, n_points, n_clusters)

def strategic_bg_points(inst_mask, all_instances,
                        total_points_range=(10, 20)):
    neg_region = (all_instances > 0) & (inst_mask == 0)
    other_ids = np.unique(all_instances[neg_region])
    other_ids = other_ids[other_ids != 0]

    pts = np.zeros_like(inst_mask, dtype=np.uint8)

    if len(other_ids) == 0:
        return pts

    total_target = random.randint(*total_points_range)

    id_to_coords, id_to_size, id_to_score = compute_instance_danger_scores(
        inst_mask, all_instances, other_ids
    )

    valid_ids = [i for i in other_ids if i in id_to_coords]
    if len(valid_ids) == 0:
        return pts

    other_ids = np.array(valid_ids)

    scores = np.array([id_to_score[i] for i in other_ids], dtype=np.float32)
    sort_idx = np.argsort(-scores)
    other_ids_sorted = other_ids[sort_idx]

    max_covered_neighbors = min(len(other_ids_sorted), total_target)
    covered_ids = other_ids_sorted[:max_covered_neighbors]

    for inst_id in covered_ids:
        coords = id_to_coords[inst_id]
        idx = np.random.choice(coords.shape[0], 1, replace=False)
        coord = tuple(coords[idx[0]])
        pts[coord] = 1

    used_points = int(pts.sum())
    points_left = total_target - used_points

    if points_left <= 0:
        return pts

    all_coords = []
    all_weights = []

    for inst_id in other_ids_sorted:
        coords = id_to_coords[inst_id]
        danger_score = id_to_score[inst_id]
        if danger_score <= 0:
            continue
        for c in coords:
            all_coords.append(c)
            all_weights.append(danger_score)

    if len(all_coords) == 0:
        return pts

    all_coords = np.array(all_coords)
    all_weights = np.array(all_weights, dtype=np.float64)

    already = pts[tuple(all_coords.T)] > 0
    all_coords = all_coords[~already]
    all_weights = all_weights[~already]

    if all_coords.shape[0] == 0 or points_left <= 0:
        return pts

    probs = all_weights / all_weights.sum()
    n_extra = min(points_left, all_coords.shape[0])

    extra_idx = np.random.choice(all_coords.shape[0], n_extra, replace=False, p=probs)
    for i in extra_idx:
        coord = tuple(all_coords[i])
        pts[coord] = 1

    return pts

def compute_instance_danger_scores(inst_mask, all_instances, other_ids):
    target_coords = np.argwhere(inst_mask > 0)
    if target_coords.shape[0] == 0:
        target_center = np.array([0.0, 0.0, 0.0])
    else:
        target_center = target_coords.mean(axis=0)

    id_to_coords = {}
    id_to_size = {}
    id_to_score = {}

    for inst_id in other_ids:
        coords = np.argwhere(all_instances == inst_id)
        if coords.shape[0] == 0:
            continue

        size = float(coords.shape[0])
        center = coords.mean(axis=0)

        size_factor = 1.0 / (size + 1e-6)
        dist = np.linalg.norm(center - target_center)
        dist_factor = 1.0 / (1.0 + dist)

        danger_score = size_factor * dist_factor

        id_to_coords[inst_id] = coords
        id_to_size[inst_id] = size
        id_to_score[inst_id] = danger_score

    return id_to_coords, id_to_size, id_to_score

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
