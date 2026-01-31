# -*- coding: utf-8 -*-
"""
Stage 2 数据生成器 - 多个分离的神经元
改进版本：
- 数据划分：前 500 个单神经元（400 训练 + 100 验证），其余留作测试
- 数据增强：随机翻转、90° 旋转
- 元数据记录：保存完整配置信息
- 可视化检查：生成最大投影图
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
from scipy.ndimage import zoom, center_of_mass


# =========================
# 配置
# =========================
@dataclass
class Stage2Config:
    """Stage 2 配置参数"""
    # 路径
    input_dir: str = "/data/wangfeiran/code/brainbow/datasets/fisbe/single_neurons"
    output_dir: str = "/data/wangfeiran/code/brainbow/datasets/fisbe/curriculum_data/stage2"
    
    # 目标体积大小
    target_size: Tuple[int, int, int] = (200, 250, 250)
    
    # 数据划分（只使用前 500 个）
    total_neurons_to_use: int = 500     # 只使用前 500 个
    train_pool_size: int = 400          # 前 400 个用于训练
    # 剩余 100 个用于验证
    
    # Stage 2 参数
    n_neurons_range: Tuple[int, int] = (2, 4)
    max_overlap_ratio: float = 0.05
    min_centroid_distance: int = 40
    neuron_scale_range: Tuple[float, float] = (0.5, 0.75)
    max_placement_attempts: int = 100
    
    # 样本数量
    train_samples: int = 800
    eval_samples: int = 160
    
    # 数据增强
    enable_augmentation: bool = True
    flip_prob: float = 0.5          # 翻转概率
    rotate90_prob: float = 0.5      # 90° 旋转概率
    
    # 其他
    random_seed: int = 42
    save_visualization: bool = True  # 是否保存可视化


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
# 数据增强
# =========================
class Augmentor:
    """数据增强器"""
    
    def __init__(self, flip_prob: float = 0.5, rotate90_prob: float = 0.5):
        self.flip_prob = flip_prob
        self.rotate90_prob = rotate90_prob
    
    def augment(self, raw: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        对 raw (C, Z, Y, X) 和 gt (N, Z, Y, X) 进行相同的增强
        返回增强后的数据和增强记录
        """
        aug_info = {'flips': [], 'rotations': 0}
        
        # 随机翻转（沿 Z, Y, X 轴）
        for axis, axis_name in [(1, 'Z'), (2, 'Y'), (3, 'X')]:
            if random.random() < self.flip_prob:
                raw = np.flip(raw, axis=axis)
                gt = np.flip(gt, axis=axis)
                aug_info['flips'].append(axis_name)
        
        # 随机 90° 旋转（在 Y-X 平面）
        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)  # 旋转 90°, 180°, 或 270°
            raw = np.rot90(raw, k=k, axes=(2, 3))
            gt = np.rot90(gt, k=k, axes=(2, 3))
            aug_info['rotations'] = k * 90
        
        # 确保内存连续
        raw = np.ascontiguousarray(raw)
        gt = np.ascontiguousarray(gt)
        
        return raw, gt, aug_info


# =========================
# Stage 2 生成器
# =========================
class Stage2Generator:
    """Stage 2 数据生成器"""
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        
        # 数据增强器
        self.augmentor = Augmentor(
            flip_prob=config.flip_prob if config.enable_augmentation else 0,
            rotate90_prob=config.rotate90_prob if config.enable_augmentation else 0
        )
        
        # 创建输出目录
        self.train_dir = os.path.join(config.output_dir, "train")
        self.eval_dir = os.path.join(config.output_dir, "eval")
        self.vis_dir = os.path.join(config.output_dir, "visualizations")
        
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        if config.save_visualization:
            os.makedirs(self.vis_dir, exist_ok=True)
        
        # 加载神经元池
        self._load_neuron_pools()
        
        # 保存配置
        self._save_config()
    
    def _load_neuron_pools(self):
        """加载并划分神经元池"""
        all_files = sorted([f for f in os.listdir(self.config.input_dir) if f.endswith('.zarr')])
        
        # 只使用前 500 个
        available_files = all_files[:self.config.total_neurons_to_use]
        
        # 划分训练和验证
        self.train_pool = available_files[:self.config.train_pool_size]
        self.eval_pool = available_files[self.config.train_pool_size:self.config.total_neurons_to_use]
        self.test_pool = all_files[self.config.total_neurons_to_use:]  # 留作测试
        
        print(f"📦 神经元池划分:")
        print(f"   - 总数: {len(all_files)}")
        print(f"   - 训练池: {len(self.train_pool)} (索引 0-{self.config.train_pool_size - 1})")
        print(f"   - 验证池: {len(self.eval_pool)} (索引 {self.config.train_pool_size}-{self.config.total_neurons_to_use - 1})")
        print(f"   - 测试池（保留）: {len(self.test_pool)} (索引 {self.config.total_neurons_to_use}+)")
    
    def _save_config(self):
        """保存配置到 JSON"""
        config_path = os.path.join(self.config.output_dir, "generation_config.json")
        config_dict = asdict(self.config)
        config_dict['generation_time'] = datetime.now().isoformat()
        config_dict['train_pool_files'] = self.train_pool
        config_dict['eval_pool_files'] = self.eval_pool
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"📝 配置已保存: {config_path}")
    
    # ---------- 神经元加载与处理 ----------
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
        except Exception as e:
            print(f"❌ 加载失败: {zarr_path}, 错误: {e}")
            return None
    
    def crop_to_bbox(self, raw: np.ndarray, gt: np.ndarray) -> Optional[Dict]:
        coords = np.argwhere(gt > 0)
        if coords.shape[0] == 0:
            return None
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        cropped_gt = gt[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_raw = raw[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        return {'raw': cropped_raw, 'gt': cropped_gt}
    
    def scale_neuron(self, raw: np.ndarray, gt: np.ndarray, 
                     target_scale: float) -> Tuple[np.ndarray, np.ndarray, float]:
        tz, ty, tx = self.config.target_size
        nz, ny, nx = gt.shape
        
        max_scale = min(tz / nz, ty / ny, tx / nx)
        final_scale = min(target_scale, max_scale, 1.0)
        
        if final_scale >= 0.99:
            return raw, gt, 1.0
        
        gt_scaled = zoom(gt, final_scale, order=0)
        raw_scaled = np.stack([zoom(raw[c], final_scale, order=1) 
                               for c in range(raw.shape[0])]).astype(np.uint8)
        return raw_scaled, gt_scaled.astype(np.uint8), final_scale
    
    # ---------- 放置策略 ----------
    def _get_grid_divisions(self, n_neurons: int) -> Tuple[int, int, int]:
        if n_neurons <= 2:
            return (1, 1, 2)
        elif n_neurons <= 4:
            return (1, 2, 2)
        else:
            return (2, 2, 2)
    
    def _grid_start(self, nz: int, ny: int, nx: int, 
                    grid_index: int, n_neurons: int) -> Tuple[int, int, int]:
        tz, ty, tx = self.config.target_size
        dz, dy, dx = self._get_grid_divisions(n_neurons)
        
        iz = (grid_index // (dy * dx)) % dz
        iy = (grid_index // dx) % dy
        ix = grid_index % dx
        
        cell_z, cell_y, cell_x = tz // dz, ty // dy, tx // dx
        
        z_start, y_start, x_start = iz * cell_z, iy * cell_y, ix * cell_x
        z_end = min((iz + 1) * cell_z, tz) - nz
        y_end = min((iy + 1) * cell_y, ty) - ny
        x_end = min((ix + 1) * cell_x, tx) - nx
        
        sz = random.randint(z_start, max(z_start, z_end))
        sy = random.randint(y_start, max(y_start, y_end))
        sx = random.randint(x_start, max(x_start, x_end))
        
        return (sz, sy, sx)
    
    def _random_start(self, nz: int, ny: int, nx: int) -> Tuple[int, int, int]:
        tz, ty, tx = self.config.target_size
        return (
            random.randint(0, max(0, tz - nz)),
            random.randint(0, max(0, ty - ny)),
            random.randint(0, max(0, tx - nx))
        )
    
    # ---------- 碰撞检测 ----------
    def _compute_overlap(self, mask: np.ndarray, gt: np.ndarray, 
                         start: Tuple[int, int, int]) -> float:
        tz, ty, tx = self.config.target_size
        nz, ny, nx = gt.shape
        sz, sy, sx = start
        
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
    
    def _check_distance(self, centroids: List[Tuple], gt: np.ndarray, 
                        start: Tuple[int, int, int]) -> bool:
        if len(centroids) == 0:
            return True
        
        local = center_of_mass(gt)
        global_c = (start[0] + local[0], start[1] + local[1], start[2] + local[2])
        
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
    def generate_sample(self, n_neurons: int, output_path: str,
                        neuron_pool: List[str], seed: int) -> Dict:
        random.seed(seed)
        np.random.seed(seed)
        
        candidates = random.sample(neuron_pool, min(n_neurons * 4, len(neuron_pool)))
        
        tz, ty, tx = self.config.target_size
        combined_raw = np.zeros((3, tz, ty, tx), dtype=np.uint8)
        combined_gt = np.zeros((n_neurons, tz, ty, tx), dtype=np.uint8)
        combined_mask = np.zeros((tz, ty, tx), dtype=np.uint8)
        
        centroids = []
        info = []
        placed = 0
        
        for fname in candidates:
            if placed >= n_neurons:
                break
            
            # 加载
            data = self.load_neuron(os.path.join(self.config.input_dir, fname))
            if data is None:
                continue
            
            cropped = self.crop_to_bbox(data['raw'], data['gt'])
            if cropped is None:
                continue
            
            # 缩放
            scale = random.uniform(*self.config.neuron_scale_range)
            raw_s, gt_s, actual_scale = self.scale_neuron(cropped['raw'], cropped['gt'], scale)
            
            nz, ny, nx = gt_s.shape
            if nz > tz or ny > ty or nx > tx:
                continue
            
            # 放置
            success = False
            for attempt in range(self.config.max_placement_attempts):
                if attempt < self.config.max_placement_attempts // 2:
                    start = self._grid_start(nz, ny, nx, placed, n_neurons)
                else:
                    start = self._random_start(nz, ny, nx)
                
                if self._compute_overlap(combined_mask, gt_s, start) > self.config.max_overlap_ratio:
                    continue
                if not self._check_distance(centroids, gt_s, start):
                    continue
                
                success = True
                break
            
            if not success:
                continue
            
            # 粘贴
            if not self._paste(combined_raw, combined_gt, combined_mask, raw_s, gt_s, start, placed):
                continue
            
            # 记录
            local = center_of_mass(gt_s)
            global_c = (start[0] + local[0], start[1] + local[1], start[2] + local[2])
            centroids.append(global_c)
            
            info.append({
                'id': placed + 1,
                'file': fname,
                'scale': float(actual_scale),
                'start': start,
                'centroid': tuple(float(x) for x in global_c)
            })
            placed += 1
        
        if placed == 0:
            return {'status': 'error', 'error': '无法放置任何神经元'}
        
        if placed < n_neurons:
            combined_gt = combined_gt[:placed]
        
        # 数据增强
        combined_raw, combined_gt, aug_info = self.augmentor.augment(combined_raw, combined_gt)
        
        # 保存
        return self._save_sample(combined_raw, combined_gt, output_path, info, aug_info, seed)
    
    # ---------- 保存 ----------
    def _save_sample(self, raw: np.ndarray, gt: np.ndarray, output_path: str,
                     neurons_info: List[Dict], aug_info: Dict, seed: int) -> Dict:
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            zroot = zarr.open(output_path, mode='w')
            g = zroot.create_group('volumes')
            _create_array_compat(g, 'raw', raw, (1, 64, 128, 128), self.compressor)
            _create_array_compat(g, 'gt_instances', gt, (1, 64, 128, 128), self.compressor)
            
            zroot.attrs['info'] = {
                'stage': 2,
                'n_neurons': len(neurons_info),
                'target_size': self.config.target_size,
                'max_overlap_ratio': self.config.max_overlap_ratio,
                'min_centroid_distance': self.config.min_centroid_distance,
                'augmentation': aug_info,
                'seed': seed,
                'neurons': neurons_info
            }
            
            return {'status': 'success', 'path': output_path, 'n_neurons': len(neurons_info)}
        except Exception as e:
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    # ---------- 可视化 ----------
    def save_visualization(self, zarr_path: str, output_prefix: str):
        """生成最大投影可视化"""
        try:
            zroot = zarr.open(zarr_path, mode='r')
            raw = zroot['volumes/raw'][:]
            gt = zroot['volumes/gt_instances'][:]
            
            # 最大投影
            raw_mip = np.max(raw, axis=1)  # (3, Y, X)
            
            # GT 合并并着色
            n_neurons = gt.shape[0]
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 0, 0], [0, 128, 0], [0, 0, 128]
            ]
            
            gt_colored = np.zeros((3, gt.shape[2], gt.shape[3]), dtype=np.uint8)
            for i in range(n_neurons):
                gt_mip = np.max(gt[i], axis=0)
                color = colors[i % len(colors)]
                for c in range(3):
                    gt_colored[c][gt_mip > 0] = color[c]
            
            # 保存为 .npy（可以用其他工具转换为图片）
            np.save(f"{output_prefix}_raw_mip.npy", raw_mip)
            np.save(f"{output_prefix}_gt_mip.npy", gt_colored)
            
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")
    
    # ---------- 主流程 ----------
    def generate_all(self):
        """生成所有数据"""
        random.seed(self.config.random_seed)
        
        print("\n" + "=" * 60)
        print("🚀 Stage 2 数据生成")
        print("=" * 60)
        print(f"📊 配置:")
        print(f"   - 神经元数量: {self.config.n_neurons_range}")
        print(f"   - 最大重叠率: {self.config.max_overlap_ratio * 100}%")
        print(f"   - 质心距离: {self.config.min_centroid_distance}")
        print(f"   - 缩放范围: {self.config.neuron_scale_range}")
        print(f"   - 数据增强: {'开启' if self.config.enable_augmentation else '关闭'}")
        print()
        
        stats = {'train': [], 'eval': []}
        
        # 训练集
        print("📁 生成训练集...")
        for i in range(self.config.train_samples):
            n = random.randint(*self.config.n_neurons_range)
            path = os.path.join(self.train_dir, f"stage2_{i:04d}.zarr")
            result = self.generate_sample(n, path, self.train_pool, seed=2000 + i)
            
            if result['status'] == 'success':
                stats['train'].append(result['n_neurons'])
                if (i + 1) % 100 == 0:
                    print(f"   ✅ {i + 1}/{self.config.train_samples}")
            else:
                print(f"   ❌ [{i}] {result.get('error')}")
        
        # 验证集
        print("\n📁 生成验证集...")
        for i in range(self.config.eval_samples):
            n = random.randint(*self.config.n_neurons_range)
            path = os.path.join(self.eval_dir, f"stage2_{i:04d}.zarr")
            result = self.generate_sample(n, path, self.eval_pool, seed=3000 + i)
            
            if result['status'] == 'success':
                stats['eval'].append(result['n_neurons'])
                if (i + 1) % 50 == 0:
                    print(f"   ✅ {i + 1}/{self.config.eval_samples}")
            else:
                print(f"   ❌ [{i}] {result.get('error')}")
        
        # 可视化（前几个样本）
        if self.config.save_visualization:
            print("\n📸 生成可视化...")
            for i in range(min(5, len(stats['train']))):
                zarr_path = os.path.join(self.train_dir, f"stage2_{i:04d}.zarr")
                vis_prefix = os.path.join(self.vis_dir, f"train_{i:04d}")
                self.save_visualization(zarr_path, vis_prefix)
            print(f"   保存至: {self.vis_dir}")
        
        # 统计
        print("\n" + "=" * 60)
        print("📊 生成统计")
        print("=" * 60)
        for split in ['train', 'eval']:
            counts = stats[split]
            if counts:
                print(f"   [{split}]")
                print(f"   - 成功: {len(counts)} 样本")
                print(f"   - 神经元数: min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.2f}")
        
        print(f"\n📂 输出目录: {self.config.output_dir}")
        print("✨ 完成!")


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    config = Stage2Config(
        # === 路径 ===
        input_dir="/data/wangfeiran/code/brainbow/datasets/fisbe/single_neurons",
        output_dir="/data/wangfeiran/code/brainbow/datasets/fisbe/curriculum_data/stage2",
        
        # === 数据划分（只用前500个）===
        total_neurons_to_use=500,   # 只使用前 500 个
        train_pool_size=400,        # 400 训练 + 100 验证
        
        # === 体积大小 ===
        target_size=(200, 250, 250),
        
        # === Stage 2 参数 ===
        n_neurons_range=(2, 4),
        max_overlap_ratio=0.05,
        min_centroid_distance=40,
        neuron_scale_range=(0.5, 0.75),
        max_placement_attempts=100,
        
        # === 样本数量 ===
        train_samples=800,
        eval_samples=160,
        
        # === 数据增强 ===
        enable_augmentation=True,
        flip_prob=0.5,
        rotate90_prob=0.5,
        
        # === 其他 ===
        random_seed=42,
        save_visualization=True,
    )
    
    generator = Stage2Generator(config)
    generator.generate_all()