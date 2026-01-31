# -*- coding: utf-8 -*-
"""
NeuronCombiner — 组合单神经元为多实例体积（不保存TIFF）
I/O 与示例保持一致：
- 读：volumes/raw -> (C,Z,Y,X)，volumes/gt_instances -> (Z,Y,X) 或 (1,Z,Y,X)
- 写：volumes/raw -> (3,Z,Y,X) uint8
      volumes/gt_instances -> (N,Z,Y,X) uint8
- 策略：
    - strategy="center": 等比缩放（必要时），居中放置；不裁剪越界
    - strategy="random": 等比缩放（必要时），随机放置；允许越界并裁剪
- 数据划分：
    - single_neurons 下按文件名排序：前 480 个为训练池，剩余为验证池
    - 训练样本写到 OUTPUT_DIR/train，验证样本写到 OUTPUT_DIR/eval
"""
import os
import random
import traceback
from typing import List, Tuple, Dict, Optional

import numpy as np
import zarr
import numcodecs
from scipy.ndimage import zoom


# =========================
# zarr v2/v3 兼容工具
# =========================
def _zarr_version_major() -> int:
    try:
        return int(zarr.__version__.split('.')[0])
    except Exception:
        return 2


def _create_array_compat(group, name, data, chunks, compressor_instance: Optional[numcodecs.abc.Codec]):
    """兼容 zarr v2/v3 的数组创建"""
    v = _zarr_version_major()
    if v >= 3:
        compressors = [compressor_instance] if compressor_instance is not None else None
        return group.create_array(name=name, data=data, chunks=chunks, compressors=compressors)
    else:
        return group.create_dataset(name, data=data, chunks=chunks, compressor=compressor_instance)


class NeuronCombiner:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 target_size: Tuple[int, int, int] = (400, 680, 680),  # (Z, Y, X)
                 scale_small: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.scale_small = scale_small

        os.makedirs(output_dir, exist_ok=True)

        # Blosc 实例
        self.compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # ---------- 读取 ----------
    def load_neuron(self, zarr_path: str) -> Optional[Dict]:
        try:
            zroot = zarr.open(zarr_path, mode='r')
            raw = zroot['volumes/raw'][:]         # (C,Z,Y,X)
            gt = zroot['volumes/gt_instances'][:] # (Z,Y,X) 或 (1,Z,Y,X)
            if gt.ndim == 4 and gt.shape[0] == 1:
                gt = gt[0]
            if raw.ndim != 4 or gt.ndim != 3:
                print(f"❌ 形状异常: raw={raw.shape}, gt={gt.shape} in {zarr_path}")
                return None
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8)
            if gt.dtype != np.uint8:
                gt = (gt > 0).astype(np.uint8)
            return {'raw': raw, 'gt': gt}
        except Exception as e:
            print(f"❌ 加载失败: {zarr_path}, 错误: {e}")
            return None

    # ---------- 裁剪 ----------
    def crop_to_minimal_bbox(self, raw: np.ndarray, gt: np.ndarray) -> Optional[Dict]:
        coords = np.argwhere(gt > 0)
        if coords.shape[0] == 0:
            return None
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        cropped_gt = gt[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
        cropped_raw = raw[:, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
        return {'raw': cropped_raw, 'gt': cropped_gt}

    # ---------- 缩放 ----------
    def _scale_to_fit(self, raw_c: np.ndarray, gt_c: np.ndarray):
        tz, ty, tx = self.target_size
        nz, ny, nx = gt_c.shape
        factors = (tz / nz, ty / ny, tx / nx)
        if self.scale_small:
            s = min(factors)  # 小的也放大
        else:
            if nz <= tz and ny <= ty and nx <= tx:
                return raw_c, gt_c, 1.0
            s = min(1.0, min(factors))  # 只缩小不放大
        gt_scaled = zoom(gt_c, s, order=0)
        raw_scaled = np.stack([zoom(raw_c[c], s, order=1) for c in range(raw_c.shape[0])]).astype(np.uint8)
        return raw_scaled, gt_scaled.astype(np.uint8), s

    # ---------- 放置位置 ----------
    def _center_start(self, nz, ny, nx):
        tz, ty, tx = self.target_size
        return ((tz - nz) // 2, (ty - ny) // 2, (tx - nx) // 2)

    def _random_start_allow_crop(self, nz, ny, nx):
        tz, ty, tx = self.target_size
        sz = random.randint(-(nz - 1), tz - 1)
        sy = random.randint(-(ny - 1), ty - 1)
        sx = random.randint(-(nx - 1), tx - 1)
        return (sz, sy, sx)

    # ---------- 粘贴（允许裁剪越界部分） ----------
    def _paste_with_crop(self, combined_raw, combined_gt, src_raw, src_gt, start, neuron_index):
        tz, ty, tx = self.target_size
        nz, ny, nx = src_gt.shape
        sz, sy, sx = start
        # 目标交集
        tz0, ty0, tx0 = max(0, sz), max(0, sy), max(0, sx)
        tz1, ty1, tx1 = min(tz, sz + nz), min(ty, sy + ny), min(tx, sx + nx)
        if tz1 <= tz0 or ty1 <= ty0 or tx1 <= tx0:
            return False
        # 源交集
        sz0, sy0, sx0 = max(0, -sz), max(0, -sy), max(0, -sx)
        sz1, sy1, sx1 = sz0 + (tz1 - tz0), sy0 + (ty1 - ty0), sx0 + (tx1 - tx0)
        # 粘贴
        mask = src_gt[sz0:sz1, sy0:sy1, sx0:sx1] > 0
        for c in range(3):
            tview = combined_raw[c, tz0:tz1, ty0:ty1, tx0:tx1]
            sview = src_raw[c, sz0:sz1, sy0:sy1, sx0:sx1]
            np.maximum(tview, sview, out=tview, where=mask)
        tgt_gt = combined_gt[neuron_index, tz0:tz1, ty0:ty1, tx0:tx1]
        tgt_gt[mask] = 1
        return True

    # ---------- 主逻辑 ----------
    def combine_neurons(self, n_neurons: int, output_name: Optional[str] = None,
                        strategy: str = "center", seed: Optional[int] = None,
                        files_subset: Optional[List[str]] = None) -> Dict:
        """
        files_subset: 若提供，则仅从该列表（文件名，非完整路径）中采样；否则使用 input_dir 中全部 .zarr
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        assert strategy in ("center", "random")

        if files_subset is None:
            all_files = [f for f in os.listdir(self.input_dir) if f.endswith('.zarr')]
        else:
            all_files = [f for f in files_subset if f.endswith('.zarr')]

        if len(all_files) < n_neurons:
            print(f"⚠️ 可用神经元({len(all_files)})少于请求({n_neurons})，将使用全部。")
            n_neurons = len(all_files)
        if n_neurons == 0:
            return {'status': 'error', 'error': '采样池为空（没有 .zarr 文件）'}

        selected = random.sample(all_files, n_neurons)
        print(f"🔄 策略: {strategy} | 合成 {n_neurons} 个神经元 | 池大小: {len(all_files)}")
        tz, ty, tx = self.target_size
        combined_raw = np.zeros((3, tz, ty, tx), dtype=np.uint8)
        combined_gt = np.zeros((n_neurons, tz, ty, tx), dtype=np.uint8)
        info = []
        placed = 0

        for i, fname in enumerate(selected):
            zpath = os.path.join(self.input_dir, fname)
            data = self.load_neuron(zpath)
            if not data:
                continue
            crop = self.crop_to_minimal_bbox(data['raw'], data['gt'])
            if not crop:
                continue
            fraw, fgt, scale = self._scale_to_fit(crop['raw'], crop['gt'])
            nz, ny, nx = fgt.shape
            if strategy == "center":
                start = self._center_start(nz, ny, nx)
            else:
                start = self._random_start_allow_crop(nz, ny, nx)
            ok = self._paste_with_crop(combined_raw, combined_gt, fraw, fgt, start, placed)
            if not ok:
                continue
            placed += 1
            info.append({'id': placed, 'file': fname,
                         'orig_size': crop['gt'].shape,
                         'final_size': fgt.shape,
                         'scale': float(scale),
                         'strategy': strategy,
                         'start': tuple(int(x) for x in start)})

        if placed == 0:
            return {'status': 'error', 'error': '没有神经元被成功处理'}
        if placed < n_neurons:
            combined_gt = combined_gt[:placed]
        out_name = output_name or f"combined_{strategy}_{placed}"
        save = self.save_combined_data(combined_raw, combined_gt, out_name, info, seed)
        return {'status': save.get('status', 'error'),
                'n_neurons': placed,
                'output_name': out_name,
                **save}

    # ---------- 保存 ----------
    def save_combined_data(self, combined_raw, combined_gt, output_name, processing_info, seed):
        try:
            zarr_path = os.path.join(self.output_dir, output_name + ".zarr")
            os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
            zroot = zarr.open(zarr_path, mode="w")
            g = zroot.create_group("volumes")
            _create_array_compat(g, "raw", combined_raw, (1, 64, 128, 128), self.compressor)
            _create_array_compat(g, "gt_instances", combined_gt, (1, 64, 128, 128), self.compressor)
            zroot.attrs['combination_info'] = {'n_neurons': len(processing_info),
                                               'target_size': self.target_size,
                                               'seed': seed,
                                               'neurons': processing_info}
            return {'status': 'success', 'zarr_path': zarr_path}
        except Exception as e:
            print("🔥 保存阶段异常:\n" + traceback.format_exc())
            return {'status': 'error', 'error': str(e)}


# =============================
# === 主程序：按文件名切分池 ===
# =============================
if __name__ == "__main__":
    INPUT_DIR = "/data/wangfeiran/code/brainbow/datasets/fisbe/single_neurons"
    OUTPUT_DIR = "/data/wangfeiran/code/brainbow/datasets/fisbe/synthetic_data"
    TARGET_SIZE = (200, 250, 250)
    SCALE_SMALL = False  # 如需放大小神经元则改为 True

    # 想各生成多少个样本
    TRAIN_SAMPLES = 1000
    EVAL_SAMPLES = 200

    # 每个样本里随机合成的神经元数量范围
    N_NEURONS_MIN, N_NEURONS_MAX = 2, 6

    # 按照你要求的目录结构：synthetic_data/train 和 synthetic_data/eval
    TRAIN_OUT = os.path.join(OUTPUT_DIR, "train")
    EVAL_OUT  = os.path.join(OUTPUT_DIR, "eval")
    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(EVAL_OUT, exist_ok=True)

    # 1) 读取 single_neurons 下的 .zarr，按文件名排序
    all_files_sorted = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".zarr")])

    # 2) 前 480 个作为训练池，其余作为验证池
    train_pool = all_files_sorted[:480]
    eval_pool  = all_files_sorted[480:]
    print(f"📦 单神经元总数: {len(all_files_sorted)} | 训练池: {len(train_pool)} | 验证池: {len(eval_pool)}")

    # 3) 不同输出目录各一个 combiner
    combiner_train = NeuronCombiner(INPUT_DIR, TRAIN_OUT, TARGET_SIZE, scale_small=SCALE_SMALL)
    combiner_eval  = NeuronCombiner(INPUT_DIR, EVAL_OUT,  TARGET_SIZE, scale_small=SCALE_SMALL)

    # 生成训练集
    for i in range(TRAIN_SAMPLES):
        n = random.randint(N_NEURONS_MIN, N_NEURONS_MAX)
        out_name = f"combined_{i:03d}"   # 训练目录里 000,001,...
        seed = 101 + i
        result = combiner_train.combine_neurons(
            n_neurons=n,
            output_name=out_name,
            seed=seed,
            files_subset=train_pool
        )
        if result['status'] == "success":
            print(f"✅ [TRAIN] {out_name}.zarr | n={result['n_neurons']}")
        else:
            print(f"❌ [TRAIN] 生成失败: {result.get('error')}")

    # 生成验证集
    for j in range(EVAL_SAMPLES):
        n = random.randint(N_NEURONS_MIN, N_NEURONS_MAX)
        out_name = f"combined_{j:03d}"   # 验证目录里 000,001,...
        seed = 202 + j
        result = combiner_eval.combine_neurons(
            n_neurons=n,
            output_name=out_name,
            seed=seed,
            files_subset=eval_pool
        )
        if result['status'] == "success":
            print(f"✅ [EVAL] {out_name}.zarr | n={result['n_neurons']}")
        else:
            print(f"❌ [EVAL] 生成失败: {result.get('error')}")
