"""
基于归一化后的足模/鞋垫网格，构建 occupancy 训练数据集。

流程概述：
1) 足模点云采样：从足模网格表面采样固定数量的点（默认 4096）。
2) 鞋垫体积采样：分区策略（默认：近表面50% + 内部25% + 外部25%）。
   使用 Open3D RaycastingScene 计算点位于网格内部(1)或外部(0)的占据标签。
3) 数据配对与打包：(foot_points, sample_points, occupancy) 按样本对保存为 .npz。
4) 数据集划分：按比例划分 train/val/test，并可选将各 split 聚合保存为 .npz。
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import open3d as o3d
from datetime import datetime


LOG_FILE_PATH: Optional[Path] = None


def _append_log(line: str) -> None:
    if LOG_FILE_PATH is None:
        return
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_info(message: str) -> None:
    """打印信息级日志，同时写入运行日志文件。"""
    line = f"[INFO] {message}"
    print(line)
    _append_log(line)


def log_warn(message: str) -> None:
    """打印警告级日志，同时写入运行日志文件。"""
    line = f"[WARN] {message}"
    print(line)
    _append_log(line)


def ensure_dir(path: Path) -> None:
    """确保目录存在（若不存在则创建）。"""
    path.mkdir(parents=True, exist_ok=True)


def _parse_foot_key(stem: str) -> Optional[Tuple[str, str]]:
    """解析足模文件名中的 (prefix, side) 格式：xxx-foot-L/R 或 xxx_foot_L/R。"""
    m = re.match(r"^(?P<prefix>.+)[-_]foot[-_](?P<side>[lr])$", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group("prefix"), m.group("side").upper()


def _parse_insole_key(stem: str) -> Optional[Tuple[str, str]]:
    """解析鞋垫文件名中的 (prefix, side) 格式：xxx-insole-L/R 或 xxx_insole_L/R。"""
    m = re.match(r"^(?P<prefix>.+)[-_]insole[-_](?P<side>[lr])$", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group("prefix"), m.group("side").upper()


def list_normalized_pairs(feet_dir: Path, insoles_dir: Path) -> List[Tuple[Path, Path]]:
    """按“相同 prefix + 相同 L/R”匹配归一化足模与鞋垫（xxx-foot-L/R 对应 xxx-insole-L/R）。"""
    foot_files = sorted(feet_dir.glob("*.stl"))
    insole_files = sorted(insoles_dir.glob("*.stl"))

    insole_index: Dict[Tuple[str, str], Path] = {}
    for p in insole_files:
        key = _parse_insole_key(p.stem)
        if key is not None:
            insole_index[key] = p

    pairs: List[Tuple[Path, Path]] = []
    for foot in foot_files:
        key = _parse_foot_key(foot.stem)
        if key is None:
            log_warn(f"无法解析足模命名 (期望 xxx-foot-L/R): {foot.name}")
            continue
        ins = insole_index.get(key)
        if ins is None:
            log_warn(f"缺少鞋垫 (prefix/side={key}), 跳过: {foot.name}")
            continue
        pairs.append((foot, ins))
    return pairs


def load_o3d_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """加载 STL 网格并计算法线。"""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"空 mesh: {path}")
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def sample_surface_points(mesh: o3d.geometry.TriangleMesh, num_points: int, method: str = "uniform") -> np.ndarray:
    """从网格表面采样点云（uniform 或 poisson）。"""
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=int(num_points))
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=int(num_points))
    return np.asarray(pcd.points, dtype=np.float32)


def aabb_with_margin(mesh: o3d.geometry.TriangleMesh, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """获取 AABB 并按比例 margin 进行对称扩展。"""
    aabb = mesh.get_axis_aligned_bounding_box()
    mn = np.asarray(aabb.get_min_bound(), dtype=np.float32)
    mx = np.asarray(aabb.get_max_bound(), dtype=np.float32)
    center = (mn + mx) / 2.0
    half = (mx - mn) / 2.0
    half = half * (1.0 + float(margin))
    mn2 = center - half
    mx2 = center + half
    return mn2, mx2


def uniform_points_in_aabb(min_xyz: np.ndarray, max_xyz: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """在给定 AABB 内均匀随机采样 3D 点。"""
    low = min_xyz.reshape(1, 3)
    high = max_xyz.reshape(1, 3)
    u = rng.random((int(num_points), 3), dtype=np.float32)
    pts = low + (high - low) * u
    return pts.astype(np.float32)


def occupancy_open3d(mesh: o3d.geometry.TriangleMesh, query_points: np.ndarray) -> np.ndarray:
    """使用 Open3D RaycastingScene 计算占据标签。

    优先调用 compute_occupancy；若不可用则回退到 compute_signed_distance。"""
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(t_mesh)
    pts = o3d.core.Tensor(query_points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    try:
        occ = scene.compute_occupancy(pts).numpy().astype(np.uint8)
    except Exception:
        # 回退：用有符号距离 SDF 判定内部(<=0)为占据
        sdf = scene.compute_signed_distance(pts).numpy()
        occ = (sdf <= 0.0).astype(np.uint8)
    return occ


def sample_near_surface(mesh: o3d.geometry.TriangleMesh, count: int, noise_std: float, method: str, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    base = sample_surface_points(mesh, int(count), method=method)
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=base.shape).astype(np.float32)
    return (base + noise).astype(np.float32)


def select_interior_exterior(mesh: o3d.geometry.TriangleMesh,
                             mn: np.ndarray,
                             mx: np.ndarray,
                             interior_count: int,
                             exterior_count: int,
                             rng: np.random.Generator,
                             pool_multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    target_in = max(0, int(interior_count))
    target_ex = max(0, int(exterior_count))
    if (target_in + target_ex) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    selected_in: List[np.ndarray] = []
    selected_ex: List[np.ndarray] = []
    max_rounds = 6
    remaining_in = target_in
    remaining_ex = target_ex
    for _ in range(max_rounds):
        need = remaining_in + remaining_ex
        if need <= 0:
            break
        pool_size = int(max(need * float(pool_multiplier), need))
        pool_pts = uniform_points_in_aabb(mn, mx, pool_size, rng)
        occ = occupancy_open3d(mesh, pool_pts)
        in_idx = np.where(occ == 1)[0]
        ex_idx = np.where(occ == 0)[0]

        if remaining_in > 0 and in_idx.size > 0:
            take = min(remaining_in, in_idx.size)
            selected_in.append(pool_pts[in_idx[:take]])
            remaining_in -= take
        if remaining_ex > 0 and ex_idx.size > 0:
            take = min(remaining_ex, ex_idx.size)
            selected_ex.append(pool_pts[ex_idx[:take]])
            remaining_ex -= take

    ins = np.concatenate(selected_in, axis=0) if selected_in else np.zeros((0, 3), dtype=np.float32)
    exs = np.concatenate(selected_ex, axis=0) if selected_ex else np.zeros((0, 3), dtype=np.float32)

    # 若仍不足，回退用均匀点补齐
    if ins.shape[0] < target_in:
        extra = uniform_points_in_aabb(mn, mx, target_in - ins.shape[0], rng)
        ins = np.concatenate([ins, extra], axis=0)
    if exs.shape[0] < target_ex:
        extra = uniform_points_in_aabb(mn, mx, target_ex - exs.shape[0], rng)
        exs = np.concatenate([exs, extra], axis=0)

    return ins.astype(np.float32), exs.astype(np.float32)


def build_single_sample(foot_path: Path,
                        insole_path: Path,
                        num_foot_points: int,
                        num_occ_points: int,
                        bbox_margin: float,
                        sampler: str,
                        rng: np.random.Generator,
                        near_surface_frac: float = 0.5,
                        interior_frac: float = 0.25,
                        exterior_frac: float = 0.25,
                        near_surface_noise: float = 0.002,
                        pool_multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建单个样本：(足模点云, 体积分布点, 占据标签)。"""
    foot_mesh = load_o3d_mesh(foot_path)
    insole_mesh = load_o3d_mesh(insole_path)

    # 1) 足模表面采样
    foot_points = sample_surface_points(foot_mesh, num_foot_points, method=sampler)

    # 2) 在鞋垫 AABB(+margin) 内采样 3D 点并计算占据标签
    mn, mx = aabb_with_margin(insole_mesh, bbox_margin)

    # 分区采样
    ns = max(0, int(num_occ_points * float(near_surface_frac)))
    remain = max(0, num_occ_points - ns)
    interior_target = max(0, int(remain * float(interior_frac / max(1e-6, (interior_frac + exterior_frac)))))
    exterior_target = max(0, remain - interior_target)

    near_surface_pts = sample_near_surface(insole_mesh, ns, near_surface_noise, sampler, rng)
    interior_pts, exterior_pts = select_interior_exterior(
        insole_mesh, mn, mx, interior_target, exterior_target, rng, pool_multiplier=pool_multiplier
    )
    samples = np.concatenate([near_surface_pts, interior_pts, exterior_pts], axis=0)
    # 计算占据标签
    occ = occupancy_open3d(insole_mesh, samples)

    return foot_points.astype(np.float32), samples.astype(np.float32), occ.astype(np.uint8)


def save_sample_npz(out_dir: Path,
                    stem: str,
                    foot_points: np.ndarray,
                    sample_points: np.ndarray,
                    occupancy: np.ndarray) -> Path:
    """将单个样本保存为 .npz。"""
    ensure_dir(out_dir)
    out_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_path,
        foot_points=foot_points,
        sample_points=sample_points,
        occupancy=occupancy,
    )
    return out_path


def split_indices(stems: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """根据比例与随机种子划分 train/val/test 的文件名列表。"""
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def write_split_files(out_root: Path, train: List[str], val: List[str], test: List[str]) -> None:
    """将划分结果写入 out_root 下的 train/val/test 文本索引文件。"""
    ensure_dir(out_root)
    for name, lst in ("train.txt", train), ("val.txt", val), ("test.txt", test):
        with open(out_root / name, "w", encoding="utf-8") as f:
            for s in lst:
                f.write(s + "\n")


def aggregate_split_npz(samples_dir: Path, stems: List[str], out_path: Path) -> None:
    """将某个 split 的样本聚合为批量 .npz，便于快速加载。"""
    ensure_dir(out_path.parent)
    foot_list: List[np.ndarray] = []
    sample_list: List[np.ndarray] = []
    occ_list: List[np.ndarray] = []
    for stem in stems:
        p = samples_dir / f"{stem}.npz"
        if not p.exists():
            log_warn(f"缺少样本文件: {p}")
            continue
        with np.load(p) as data:
            foot_list.append(data["foot_points"]) 
            sample_list.append(data["sample_points"]) 
            occ_list.append(data["occupancy"]) 
    if len(foot_list) == 0:
        log_warn(f"聚合空 split，跳过: {out_path}")
        return
    foot_arr = np.stack(foot_list, axis=0)
    sample_arr = np.stack(sample_list, axis=0)
    occ_arr = np.stack(occ_list, axis=0)
    np.savez_compressed(out_path, foot_points=foot_arr, sample_points=sample_arr, occupancy=occ_arr)


def main() -> None:
    # 命令行参数：采样规模、AABB margin、划分比例、输出路径等
    parser = argparse.ArgumentParser(description="基于归一化网格构建 occupancy 数据集")
    parser.add_argument("--normalized-root", type=str, default=str(Path("data") / "normalize"), help="归一化数据根目录")
    parser.add_argument("--num-foot-points", type=int, default=4096)
    parser.add_argument("--num-occupancy", type=int, default=65536)
    parser.add_argument("--bbox-margin", type=float, default=0.15, help="AABB 扩展比例")
    parser.add_argument("--sampler", choices=["uniform", "poisson"], default="uniform")
    # 分区采样参数
    parser.add_argument("--near-surface-frac", type=float, default=0.5)
    parser.add_argument("--interior-frac", type=float, default=0.25)
    parser.add_argument("--exterior-frac", type=float, default=0.25)
    parser.add_argument("--near-surface-noise", type=float, default=0.002)
    parser.add_argument("--pool-multiplier", type=float, default=3.0, help="内外部筛选池倍数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-root", type=str, default=str(Path("data") / "dataset"))
    parser.add_argument("--aggregate", action="store_true", help="同时输出按 split 聚合的 .npz")
    parser.add_argument("--log-dir", type=str, default=str(Path("log")), help="日志输出目录")
    args = parser.parse_args()

    # 初始化运行日志文件
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_dir / f"build_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    normalized_root = Path(args.normalized_root)
    feet_dir = normalized_root / "feet"
    insoles_dir = normalized_root / "insoles"
    if not feet_dir.exists() or not insoles_dir.exists():
        raise FileNotFoundError("未找到归一化数据目录: feet 或 insoles 不存在")

    out_root = Path(args.out_root)
    samples_dir = out_root / "samples"
    ensure_dir(samples_dir)

    # 读取样本对（按文件名匹配）
    pairs = list_normalized_pairs(feet_dir, insoles_dir)
    log_info(f"有效样本对: {len(pairs)}")

    np_rng = np.random.default_rng(args.seed)

    stems: List[str] = []
    for foot_p, insole_p in pairs:
        stem = foot_p.stem
        try:
            # 构建并保存单个样本 .npz
            current_occ = int(args.num_occupancy)
            while True:
                try:
                    foot_points, sample_points, occupancy = build_single_sample(
                        foot_p,
                        insole_p,
                        num_foot_points=args.num_foot_points,
                        num_occ_points=current_occ,
                        bbox_margin=args.bbox_margin,
                        sampler=args.sampler,
                        rng=np_rng,
                        near_surface_frac=args.near_surface_frac,
                        interior_frac=args.interior_frac,
                        exterior_frac=args.exterior_frac,
                        near_surface_noise=args.near_surface_noise,
                        pool_multiplier=args.pool_multiplier,
                    )
                    break
                except Exception as e_inner:
                    msg = str(e_inner).lower()
                    if ("out of memory" in msg) or isinstance(e_inner, MemoryError):
                        if current_occ <= 4096:
                            raise
                        new_occ = max(4096, current_occ // 2)
                        log_warn(f"{stem}: OOM at num_occupancy={current_occ}, retry with {new_occ}")
                        current_occ = new_occ
                        continue
                    else:
                        raise
            save_sample_npz(samples_dir, stem, foot_points, sample_points, occupancy)
            stems.append(stem)
        except Exception as e:
            log_warn(f"构建样本失败 {stem}: {e}")

    if len(stems) == 0:
        log_warn("未生成任何样本，结束")
        return

    # 写入划分文件并可选聚合输出
    train, val, test = split_indices(stems, args.train_ratio, args.val_ratio, args.seed)
    write_split_files(out_root, train, val, test)
    log_info(f"划分完成: train={len(train)}, val={len(val)}, test={len(test)}")

    if args.aggregate:
        log_info("生成聚合 .npz ...")
        aggregate_split_npz(samples_dir, train, out_root / "train.npz")
        aggregate_split_npz(samples_dir, val, out_root / "val.npz")
        aggregate_split_npz(samples_dir, test, out_root / "test.npz")
        log_info("聚合完成")


if __name__ == "__main__":
    main()


