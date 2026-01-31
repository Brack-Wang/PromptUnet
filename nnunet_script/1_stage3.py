# -*- coding: utf-8 -*-
"""
Stage 3 数据生成器 - 紧密排列的神经元（基于生物学现实）

生物学背景：
- 神经元之间有细胞膜边界，不会大面积重合
- 光学显微镜下看起来重合，实际有边界
- 神经元可以贴得很近，但重叠面积很小

设计原则：
- 神经元靠得近（质心距离小）
- 最大重叠率 15%（模拟光学分辨率限制）
- 约 30% 的神经元对有轻微接触
- 约 70% 的神经元对是独立的

数据划分：只使用前 500 个单神经元（400 训练 + 100 验证）
"""
import os
import json
import random
import traceback
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import zarr
import numcodecs
from scipy.ndimage import zoom, center_of_mass, map_coordinates, gaussian_filter


# =========================
# 配置
# =========================
@dataclass 
class Stage3Config:
    """Stage 3 配置"""
    # 路径
    input_dir: str = "/data/wangfeiran/code/brainbow/datasets/fisbe/single_neurons"
    output_dir: str = "/data/wangfeiran/code/brainbow/datasets/fisbe/curriculum_data/stage3"
    
    # 目标体积大小
    target_size: Tuple[int, int, int] = (200, 250, 250)
    
    # 数据划分（只使用前 500 个）
    total_neurons_to_use: int = 500
    train_pool_size: int = 400
    
    # 神经元数量和缩放
    n_neurons_range: Tuple[int, int] = (3, 6)
    neuron_scale_range: Tuple[float, float] = (0.45, 0.7)
    
    # 核心参数：紧密但不大面积重叠
    max_overlap_ratio: float = 0.15        # 最大重叠 15%
    touching_pair_ratio: float = 0.30      # 约 30% 的神经元对会接触
    min_centroid_distance: int = 25        # 最小质心距离（允许靠近）
    
    # 放置参数
    max_placement_attempts: int = 150
    min_visibility: float = 0.5            # 至少 50% 可见
    
    # 样本数量
    train_samples: int = 600
    eval_samples: int = 120
    
    # 数据增强
    enable_augmentation: bool = True
    flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    elastic_deform_prob: float = 0.3
    elastic_alpha: float = 15.0
    elastic_sigma: float = 3.0
    
    # 其他
    random_seed: int = 42
    save_visualization: bool = True


# =========================
# zarr 兼容工具
# =========================
def _zarr_version_major() -> int:
    try:
        return int(zarr.__version__.split('.')[0])
    except Exception:
        return 2


def _create_array_compat(group, name, data, chunks, compressor_instance):
    v = _zarr_version_major()
    if v >= 3:
        compressors = [compressor_instance] if compressor_instance is not None else None
        return group.create_array(name=name, data=data, chunks=chunks, compressors=compressors)
    else:
        return group.create_dataset(name, data=data, chunks=chunks, compressor=compressor_instance)


# =========================
# 数据增强器
# =========================
class Augmentor:
    """数据增强器"""
    
    def __init__(self, config: Stage3Config):
        self.config = config
    
    def augment(self, raw: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if not self.config.enable_augmentation:
            return raw, gt, {'augmentation': 'disabled'}
        
        aug_info = {'flips': [], 'rotation': 0, 'elastic': False}
        
        # 随机翻转
        for axis, axis_name in [(1, 'Z'), (2, 'Y'), (3, 'X')]:
            if random.random() < self.config.flip_prob:
                raw = np.flip(raw, axis=axis)
                gt = np.flip(gt, axis=axis)
                aug_info['flips'].append(axis_name)
        
        # 随机 90° 旋转
        if random.random() < self.config.rotate90_prob:
            k = random.randint(1, 3)
            raw = np.rot90(raw, k=k, axes=(2, 3))
            gt = np.rot90(gt, k=k, axes=(2, 3))
            aug_info['rotation'] = k * 90
        
        # 弹性变形
        if random.random() < self.config.elastic_deform_prob:
            raw, gt = self._elastic_deform(raw, gt)
            aug_info['elastic'] = True
        
        return np.ascontiguousarray(raw), np.ascontiguousarray(gt), aug_info
    
    def _elastic_deform(self, raw: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, dz, dy, dx = raw.shape
        alpha = self.config.elastic_alpha
        sigma = self.config.elastic_sigma
        
        dz_field = gaussian_filter(np.random.randn(dz, dy, dx), sigma) * alpha
        dy_field = gaussian_filter(np.random.randn(dz, dy, dx), sigma) * alpha
        dx_field = gaussian_filter(np.random.randn(dz, dy, dx), sigma) * alpha
        
        z, y, x = np.meshgrid(np.arange(dz), np.arange(dy), np.arange(dx), indexing='ij')
        
        indices = [
            np.clip(z + dz_field, 0, dz - 1),
            np.clip(y + dy_field, 0, dy - 1),
            np.clip(x + dx_field, 0, dx - 1),
        ]
        
        raw_deformed = np.zeros_like(raw)
        for c in range(raw.shape[0]):
            raw_deformed[c] = map_coordinates(raw[c], indices, order=1, mode='reflect')
        
        gt_deformed = np.zeros_like(gt)
        for n in range(gt.shape[0]):
            gt_deformed[n] = map_coordinates(gt[n], indices, order=0, mode='reflect')
        
        return raw_deformed.astype(np.uint8), gt_deformed.astype(np.uint8)


# =========================
# Stage 3 生成器
# =========================
class Stage3Generator:
    """Stage 3 数据生成器 - 紧密排列但不大面积重叠"""
    
    def __init__(self, config: Stage3Config):
        self.config = config
        self.compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        self.augmentor = Augmentor(config)
        
        # 创建目录
        self.train_dir = os.path.join(config.output_dir, "train")
        self.eval_dir = os.path.join(config.output_dir, "eval")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        if config.save_visualization:
            self.vis_dir = os.path.join(config.output_dir, "visualizations")
            os.makedirs(self.vis_dir, exist_ok=True)
        
        # 加载神经元池
        self._load_neuron_pools()
        self._save_config()
    
    def _load_neuron_pools(self):
        all_files = sorted([f for f in os.listdir(self.config.input_dir) if f.endswith('.zarr')])
        available = all_files[:self.config.total_neurons_to_use]
        
        self.train_pool = available[:self.config.train_pool_size]
        self.eval_pool = available[self.config.train_pool_size:self.config.total_neurons_to_use]
        
        print(f"📦 神经元池划分:")
        print(f"   - 总数: {len(all_files)}")
        print(f"   - 使用: 前 {self.config.total_neurons_to_use} 个")
        print(f"   - 训练池: {len(self.train_pool)}")
        print(f"   - 验证池: {len(self.eval_pool)}")
    
    def _save_config(self):
        config_path = os.path.join(self.config.output_dir, "generation_config.json")
        config_dict = asdict(self.config)
        config_dict['generation_time'] = datetime.now().isoformat()
        config_dict['description'] = "Stage 3: 紧密排列但不大面积重叠（基于生物学现实）"
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"📝 配置已保存: {config_path}")
    
    # ---------- 神经元加载 ----------
    def load_neuron(self, zarr_path: str) -> Optional[Dict]:
        try:
            zroot = zarr.open(zarr_path, mode='r')
            raw = zroot['volumes/raw'][:]
            gt = zroot['volumes/gt_instances'][:]
            if gt.ndim == 4 and gt.shape[0] == 1:
                gt = gt[0]
            if raw.ndim != 4 or gt.ndim != 3:
                return None
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8)
            if gt.dtype != np.uint8:
                gt = (gt > 0).astype(np.uint8)
            return {'raw': raw, 'gt': gt}
        except Exception:
            return None
    
    def crop_to_bbox(self, raw: np.ndarray, gt: np.ndarray) -> Optional[Dict]:
        coords = np.argwhere(gt > 0)
        if coords.shape[0] == 0:
            return None
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        return {
            'raw': raw[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1],
            'gt': gt[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        }
    
    def scale_neuron(self, raw: np.ndarray, gt: np.ndarray, 
                     target_scale: float) -> Tuple[np.ndarray, np.ndarray, float]:
        tz, ty, tx = self.config.target_size
        nz, ny, nx = gt.shape
        
        max_scale = min(tz / nz, ty / ny, tx / nx) * 0.95
        final_scale = min(target_scale, max_scale, 1.0)
        
        if nz > tz or ny > ty or nx > tx:
            force_scale = min(tz / nz, ty / ny, tx / nx) * 0.85
            final_scale = min(final_scale, force_scale)
        
        if final_scale >= 0.99:
            return raw, gt, 1.0
        
        gt_scaled = zoom(gt, final_scale, order=0)
        raw_scaled = np.stack([zoom(raw[c], final_scale, order=1) 
                               for c in range(raw.shape[0])]).astype(np.uint8)
        return raw_scaled, gt_scaled.astype(np.uint8), final_scale
    
    # ---------- 混合放置策略 ----------
    def _compute_placement(self, existing_centroids: List[Tuple],
                           neuron_size: Tuple[int, int, int],
                           placed_count: int, n_neurons: int,
                           should_touch: bool) -> Tuple[int, int, int]:
        """
        混合放置策略：
        - 如果 should_touch=True: 靠近已有神经元
        - 如果 should_touch=False: 分散放置
        """
        nz, ny, nx = neuron_size
        tz, ty, tx = self.config.target_size
        
        if len(existing_centroids) == 0:
            # 第一个神经元：放在中心区域
            z_min = max(0, (tz - nz) // 4)
            z_max = max(z_min + 1, 3 * (tz - nz) // 4)
            y_min = max(0, (ty - ny) // 4)
            y_max = max(y_min + 1, 3 * (ty - ny) // 4)
            x_min = max(0, (tx - nx) // 4)
            x_max = max(x_min + 1, 3 * (tx - nx) // 4)
            
            return (
                random.randint(z_min, z_max),
                random.randint(y_min, y_max),
                random.randint(x_min, x_max)
            )
        
        if should_touch:
            # 靠近已有神经元（但不重叠太多）
            anchor = random.choice(existing_centroids)
            
            # 使用中等 spread，让神经元靠近但不完全重叠
            spread = 40  # 适中的扩散
            
            offset_z = np.random.normal(0, spread)
            offset_y = np.random.normal(0, spread)
            offset_x = np.random.normal(0, spread)
            
            sz = int(anchor[0] + offset_z - nz // 2)
            sy = int(anchor[1] + offset_y - ny // 2)
            sx = int(anchor[2] + offset_x - nx // 2)
        else:
            # 分散放置：使用 Grid 策略
            if n_neurons <= 2:
                divisions = (1, 1, 2)
            elif n_neurons <= 4:
                divisions = (1, 2, 2)
            else:
                divisions = (2, 2, 2)
            
            dz, dy, dx = divisions
            iz = (placed_count // (dy * dx)) % dz
            iy = (placed_count // dx) % dy
            ix = placed_count % dx
            
            cell_z, cell_y, cell_x = tz // dz, ty // dy, tx // dx
            
            # 网格中心 + 小抖动
            center_z = iz * cell_z + cell_z // 2
            center_y = iy * cell_y + cell_y // 2
            center_x = ix * cell_x + cell_x // 2
            
            jitter = 15
            sz = int(center_z + np.random.uniform(-jitter, jitter) - nz // 2)
            sy = int(center_y + np.random.uniform(-jitter, jitter) - ny // 2)
            sx = int(center_x + np.random.uniform(-jitter, jitter) - nx // 2)
        
        # 限制范围
        sz = max(0, min(sz, tz - nz))
        sy = max(0, min(sy, ty - ny))
        sx = max(0, min(sx, tx - nx))
        
        return (sz, sy, sx)
    
    # ---------- 重叠计算 ----------
    def _compute_overlap(self, mask: np.ndarray, gt: np.ndarray, 
                         start: Tuple[int, int, int]) -> float:
        tz, ty, tx = self.config.target_size
        nz, ny, nx = gt.shape
        sz, sy, sx = start
        
        # 边界检查
        if sz < 0 or sy < 0 or sx < 0:
            return 1.0
        if sz + nz > tz or sy + ny > ty or sx + nx > tx:
            return 1.0
        
        existing = mask[sz:sz+nz, sy:sy+ny, sx:sx+nx] > 0
        new_region = gt > 0
        new_count = np.sum(new_region)
        
        if new_count == 0:
            return 0.0
        
        return np.sum(new_region & existing) / new_count
    
    # ---------- 质心距离检查 ----------
    def _check_centroid_distance(self, centroids: List[Tuple], 
                                  gt: np.ndarray, start: Tuple[int, int, int]) -> bool:
        if len(centroids) == 0:
            return True
        
        nz, ny, nx = gt.shape
        local = center_of_mass(gt)
        global_c = (
            start[0] + local[0],
            start[1] + local[1],
            start[2] + local[2]
        )
        
        for c in centroids:
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(global_c, c)))
            if dist < self.config.min_centroid_distance:
                return False
        return True
    
    # ---------- 粘贴 ----------
    def _paste(self, combined_raw: np.ndarray, combined_gt: np.ndarray,
               combined_mask: np.ndarray, src_raw: np.ndarray,
               src_gt: np.ndarray, start: Tuple[int, int, int],
               neuron_idx: int) -> bool:
        tz, ty, tx = self.config.target_size
        nz, ny, nx = src_gt.shape
        sz, sy, sx = start
        
        if sz < 0 or sy < 0 or sx < 0:
            return False
        if sz + nz > tz or sy + ny > ty or sx + nx > tx:
            return False
        
        mask = src_gt > 0
        n_ch = min(3, src_raw.shape[0])
        
        for c in range(n_ch):
            target = combined_raw[c, sz:sz+nz, sy:sy+ny, sx:sx+nx]
            np.maximum(target, src_raw[c], out=target, where=mask)
        
        for c in range(n_ch, 3):
            target = combined_raw[c, sz:sz+nz, sy:sy+ny, sx:sx+nx]
            np.maximum(target, src_raw[0], out=target, where=mask)
        
        combined_gt[neuron_idx, sz:sz+nz, sy:sy+ny, sx:sx+nx][mask] = 1
        combined_mask[sz:sz+nz, sy:sy+ny, sx:sx+nx][mask] = 1
        
        return True
    
    # ---------- 生成单个样本 ----------
    def generate_sample(self, output_path: str, neuron_pool: List[str], seed: int) -> Dict:
        random.seed(seed)
        np.random.seed(seed)
        
        n_neurons = random.randint(*self.config.n_neurons_range)
        candidates = random.sample(neuron_pool, min(n_neurons * 5, len(neuron_pool)))
        
        # 决定哪些神经元对会"接触"
        # 约 30% 的神经元会尝试靠近其他神经元
        touching_count = max(1, int(n_neurons * self.config.touching_pair_ratio))
        touching_indices = set(random.sample(range(1, n_neurons), min(touching_count, n_neurons - 1)))
        
        tz, ty, tx = self.config.target_size
        combined_raw = np.zeros((3, tz, ty, tx), dtype=np.uint8)
        combined_gt = np.zeros((n_neurons, tz, ty, tx), dtype=np.uint8)
        combined_mask = np.zeros((tz, ty, tx), dtype=np.uint8)
        
        centroids = []
        neurons_info = []
        placed = 0
        
        for fname in candidates:
            if placed >= n_neurons:
                break
            
            data = self.load_neuron(os.path.join(self.config.input_dir, fname))
            if data is None:
                continue
            
            cropped = self.crop_to_bbox(data['raw'], data['gt'])
            if cropped is None:
                continue
            
            scale = random.uniform(*self.config.neuron_scale_range)
            raw_s, gt_s, actual_scale = self.scale_neuron(cropped['raw'], cropped['gt'], scale)
            
            nz, ny, nx = gt_s.shape
            if nz > tz or ny > ty or nx > tx:
                continue
            
            # 决定这个神经元是否应该靠近其他神经元
            should_touch = placed in touching_indices
            
            # 尝试放置
            best_start = None
            best_overlap = None
            
            for attempt in range(self.config.max_placement_attempts):
                start = self._compute_placement(centroids, (nz, ny, nx), placed, n_neurons, should_touch)
                
                overlap = self._compute_overlap(combined_mask, gt_s, start)
                
                # 检查重叠是否在允许范围内
                if overlap > self.config.max_overlap_ratio:
                    continue
                
                # 检查质心距离
                if not self._check_centroid_distance(centroids, gt_s, start):
                    continue
                
                best_start = start
                best_overlap = overlap
                break
            
            if best_start is None:
                continue
            
            # 粘贴
            if not self._paste(combined_raw, combined_gt, combined_mask, raw_s, gt_s, best_start, placed):
                continue
            
            # 记录
            local = center_of_mass(gt_s)
            global_c = (
                best_start[0] + local[0],
                best_start[1] + local[1],
                best_start[2] + local[2]
            )
            centroids.append(global_c)
            
            neurons_info.append({
                'id': placed + 1,
                'file': fname,
                'scale': float(actual_scale),
                'start': best_start,
                'centroid': tuple(float(x) for x in global_c),
                'overlap': float(best_overlap) if best_overlap else 0.0,
                'touching': should_touch
            })
            placed += 1
        
        if placed < 2:
            return {'status': 'error', 'error': f'只放置了 {placed} 个神经元'}
        
        if placed < n_neurons:
            combined_gt = combined_gt[:placed]
        
        # 数据增强
        combined_raw, combined_gt, aug_info = self.augmentor.augment(combined_raw, combined_gt)
        
        # 计算统计
        touching_neurons = sum(1 for info in neurons_info if info.get('touching', False))
        avg_overlap = np.mean([info['overlap'] for info in neurons_info if info['overlap'] > 0]) if any(info['overlap'] > 0 for info in neurons_info) else 0.0
        
        return self._save_sample(combined_raw, combined_gt, output_path, neurons_info, aug_info, avg_overlap, touching_neurons, seed)
    
    # ---------- 保存 ----------
    def _save_sample(self, raw: np.ndarray, gt: np.ndarray, output_path: str,
                     neurons_info: List[Dict], aug_info: Dict, 
                     avg_overlap: float, touching_count: int, seed: int) -> Dict:
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            zroot = zarr.open(output_path, mode='w')
            g = zroot.create_group('volumes')
            _create_array_compat(g, 'raw', raw, (1, 64, 128, 128), self.compressor)
            _create_array_compat(g, 'gt_instances', gt, (1, 64, 128, 128), self.compressor)
            
            zroot.attrs['info'] = {
                'stage': 3,
                'n_neurons': len(neurons_info),
                'touching_neurons': touching_count,
                'avg_overlap': avg_overlap,
                'max_overlap_allowed': self.config.max_overlap_ratio,
                'augmentation': aug_info,
                'seed': seed,
                'neurons': neurons_info
            }
            
            return {
                'status': 'success',
                'path': output_path,
                'n_neurons': len(neurons_info),
                'touching': touching_count,
                'avg_overlap': avg_overlap
            }
        except Exception as e:
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    # ---------- 可视化 ----------
    def save_visualization(self, zarr_path: str, output_prefix: str):
        try:
            zroot = zarr.open(zarr_path, mode='r')
            raw = zroot['volumes/raw'][:]
            gt = zroot['volumes/gt_instances'][:]
            
            raw_mip = np.max(raw, axis=1)
            
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 128, 0], [128, 0, 128], [0, 128, 128]
            ]
            
            gt_colored = np.zeros((3, gt.shape[2], gt.shape[3]), dtype=np.uint8)
            for i in range(min(gt.shape[0], len(colors))):
                gt_mip = np.max(gt[i], axis=0)
                color = colors[i]
                for c in range(3):
                    gt_colored[c][gt_mip > 0] = color[c]
            
            np.save(f"{output_prefix}_raw_mip.npy", raw_mip)
            np.save(f"{output_prefix}_gt_mip.npy", gt_colored)
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")
    
    # ---------- 主流程 ----------
    def generate_all(self):
        random.seed(self.config.random_seed)
        
        print("\n" + "=" * 70)
        print("🚀 Stage 3 数据生成（紧密排列，基于生物学现实）")
        print("=" * 70)
        print(f"📊 配置:")
        print(f"   - 神经元数量: {self.config.n_neurons_range}")
        print(f"   - 最大重叠率: {self.config.max_overlap_ratio * 100}%")
        print(f"   - 接触神经元比例: {self.config.touching_pair_ratio * 100}%")
        print(f"   - 最小质心距离: {self.config.min_centroid_distance}")
        print(f"   - 训练样本: {self.config.train_samples}")
        print(f"   - 验证样本: {self.config.eval_samples}")
        print()
        
        stats = {'train': [], 'eval': []}
        seed_counter = 3000
        
        # 训练集
        print("📁 生成训练集...")
        for i in range(self.config.train_samples):
            path = os.path.join(self.train_dir, f"stage3_{i:04d}.zarr")
            result = self.generate_sample(path, self.train_pool, seed_counter)
            seed_counter += 1
            
            if result['status'] == 'success':
                stats['train'].append({
                    'n_neurons': result['n_neurons'],
                    'touching': result['touching'],
                    'overlap': result['avg_overlap']
                })
                if (i + 1) % 100 == 0:
                    print(f"   ✅ {i + 1}/{self.config.train_samples}")
            else:
                if (i + 1) % 50 == 0:
                    print(f"   ⚠️ [{i}] {result.get('error')}")
        
        # 验证集
        print("\n📁 生成验证集...")
        for i in range(self.config.eval_samples):
            path = os.path.join(self.eval_dir, f"stage3_{i:04d}.zarr")
            result = self.generate_sample(path, self.eval_pool, seed_counter)
            seed_counter += 1
            
            if result['status'] == 'success':
                stats['eval'].append({
                    'n_neurons': result['n_neurons'],
                    'touching': result['touching'],
                    'overlap': result['avg_overlap']
                })
                if (i + 1) % 50 == 0:
                    print(f"   ✅ {i + 1}/{self.config.eval_samples}")
        
        # 可视化
        if self.config.save_visualization:
            print("\n📸 生成可视化...")
            for i in range(min(5, len(stats['train']))):
                zarr_path = os.path.join(self.train_dir, f"stage3_{i:04d}.zarr")
                vis_prefix = os.path.join(self.vis_dir, f"train_{i:04d}")
                self.save_visualization(zarr_path, vis_prefix)
            print(f"   保存至: {self.vis_dir}")
        
        # 统计
        print("\n" + "=" * 70)
        print("📊 生成统计")
        print("=" * 70)
        for split in ['train', 'eval']:
            data = stats[split]
            if data:
                neurons = [d['n_neurons'] for d in data]
                touching = [d['touching'] for d in data]
                overlaps = [d['overlap'] for d in data if d['overlap'] > 0]
                
                print(f"\n   [{split}] 成功: {len(data)} 样本")
                print(f"   - 神经元数: {np.mean(neurons):.1f} ± {np.std(neurons):.1f}")
                print(f"   - 接触神经元: {np.mean(touching):.1f} ± {np.std(touching):.1f}")
                if overlaps:
                    print(f"   - 平均重叠率: {np.mean(overlaps)*100:.1f}% ± {np.std(overlaps)*100:.1f}%")
        
        print(f"\n📂 输出目录: {self.config.output_dir}")
        print("✨ 完成!")


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    config = Stage3Config(
        # === 路径 ===
        input_dir="/data/wangfeiran/code/brainbow/datasets/fisbe/single_neurons",
        output_dir="/data/wangfeiran/code/brainbow/datasets/fisbe/curriculum_data/stage3",
        
        # === 数据划分 ===
        total_neurons_to_use=500,
        train_pool_size=400,
        
        # === 体积大小 ===
        target_size=(200, 250, 250),
        
        # === 神经元参数 ===
        n_neurons_range=(3, 6),
        neuron_scale_range=(0.45, 0.7),
        
        # === 核心参数（基于生物学现实）===
        max_overlap_ratio=0.15,         # 最大 15% 重叠
        touching_pair_ratio=0.30,       # 30% 的神经元会接触
        min_centroid_distance=25,       # 允许靠近
        
        # === 样本数量 ===
        train_samples=600,
        eval_samples=120,
        
        # === 数据增强 ===
        enable_augmentation=True,
        flip_prob=0.5,
        rotate90_prob=0.5,
        elastic_deform_prob=0.3,
        
        # === 其他 ===
        random_seed=42,
        save_visualization=True,
    )
    
    generator = Stage3Generator(config)
    generator.generate_all()