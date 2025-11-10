"""
将原始足模与鞋垫 STL 批量进行基本清理、可选对齐、可选高度裁剪，并统一到全局坐标范围内，输出到
data/normalize 下的 feet/、insoles/ 与 meta/。meta 中记录了全局 shift 与 scale 等信息。

主要步骤：
1) 基础几何清理：退化/重复/非流形要素移除与法线重建。
2) 连通组件保留：仅保留面积最大的三角片区，剔除离群碎片（足模）。
3) 可选 ICP 对齐：将足模对齐到鞋垫坐标（或反之），减少相对位姿差异。
4) 可选高度裁剪：以足模的最小 z 为 0，保留 (z-min_z)<=H 的顶点与相关三角形。
5) 全局归一化：按全局 AABB 计算 shift 与 scale，使整体落入约 [-1,1] 范围。
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import numpy as np
import open3d as o3d


# ========== 基础工具函数 ==========
LOG_FILE_PATH: Optional[Path] = None


def _append_log(line: str) -> None:
    global LOG_FILE_PATH
    if LOG_FILE_PATH is None:
        return
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_info(msg: str):
    line = f"[INFO] {msg}"
    print(line)
    _append_log(line)


def log_warn(msg: str):
    line = f"[WARN] {msg}"
    print(line)
    _append_log(line)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """读取 STL 网格并计算法线。"""
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertices():
        raise ValueError(f"空 mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str):
    """保存网格到给定路径，并确保父目录存在。"""
    ensure_dir(Path(path).parent)
    o3d.io.write_triangle_mesh(path, mesh)


def basic_mesh_cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    基础几何清理：去除退化/重复/非流形等，并移除未被引用的顶点。
    """
    try:
        mesh.remove_degenerate_triangles()
    except Exception:
        pass
    try:
        mesh.remove_duplicated_triangles()
    except Exception:
        pass
    try:
        mesh.remove_duplicated_vertices()
    except Exception:
        pass
    try:
        mesh.remove_non_manifold_edges()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    mesh.compute_vertex_normals()
    return mesh


def keep_largest_connected_component_o3d(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    仅保留最大的连通三角片区（按面积），用于剔除远离主体的小碎片（离群组件）。
    参考 stl2pointcloud_withpreprocess.py 中的组件清理策略。
    """
    try:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        areas = np.asarray(cluster_area)
        if areas.size == 0:
            return mesh
        keep_idx = int(np.argmax(areas))
        clusters = np.asarray(triangle_clusters)
        remove_mask = clusters != keep_idx
        if remove_mask.size != len(mesh.triangles):
            # 对齐长度异常时放弃清理
            return mesh
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        # cluster 失败则返回原 mesh
        return mesh


def clip_mesh_by_height_from_min_z(mesh: o3d.geometry.TriangleMesh, max_height: float) -> o3d.geometry.TriangleMesh:
    """
    以当前网格的顶点最小 z 为 0 基准，移除所有 (z - min_z) > max_height 的顶点及相关三角形。
    注意：此处仅用于粗略高度裁剪，不对相交三角形做几何切割。
    """
    if max_height is None or max_height <= 0:
        return mesh
    try:
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            return mesh
        min_z = float(np.min(verts[:, 2]))
        keep = (verts[:, 2] - min_z) <= float(max_height)
        # 若全部被移除或移除过多，则跳过
        if not np.any(keep) or np.sum(keep) < 3:
            return mesh
        remove_mask = np.logical_not(keep)
        mesh.remove_vertices_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return mesh


def save_meta(meta: dict, out_path: Path, fmt: str = "json"):
    """将元信息保存为 json 或 npz。"""
    ensure_dir(out_path.parent)
    if fmt == "json":
        with open(str(out_path) + ".json", "w") as f:
            json.dump(meta, f, indent=2)
    elif fmt == "npz":
        np.savez(str(out_path) + ".npz", **meta)
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        raise ValueError(f"Unsupported meta format: {fmt}")


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


def match_pairs_by_stem(feet_dir: Path, insoles_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """按“相同 prefix + 相同 L/R”匹配，其中足模名为 xxx-foot-L/R，鞋垫名为 xxx-insole-L/R。"""
    foot_files = list(feet_dir.glob("*.stl"))
    insole_files = list(insoles_dir.glob("*.stl"))

    insole_index: Dict[Tuple[str, str], Path] = {}
    for p in insole_files:
        key = _parse_insole_key(p.stem)
        if key is None:
            continue
        insole_index[key] = p

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for foot in foot_files:
        fkey = _parse_foot_key(foot.stem)
        if fkey is None:
            log_warn(f"无法解析足模命名 (期望 xxx-foot-L/R): {foot.name}")
            pairs.append((foot, None))
            continue
        insole = insole_index.get(fkey, None)
        if insole is None:
            log_warn(f"未找到匹配鞋垫: prefix/side={fkey} for {foot.name}")
        pairs.append((foot, insole))
    return pairs


# ========== 对齐函数（可选） ==========
def compute_icp_transform(source: o3d.geometry.TriangleMesh,
                          target: o3d.geometry.TriangleMesh,
                          voxel_size: float = 2.0,
                          max_points: int = 5000) -> np.ndarray:
    """
    用 ICP 估计两个 mesh 的对齐矩阵
    """
    src = source.sample_points_uniformly(max_points)
    tgt = target.sample_points_uniformly(max_points)

    threshold = voxel_size * 1.5
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation


# ========== 全局归一化 ==========
def compute_global_normalization(feet_dir: Path, insoles_dir: Path, voxel_size: float = 0.0):
    """
    扫描所有 mesh，得到全局 shift 和 scale。
    确保足模和鞋垫使用相同的中心点进行归一化。
    """
    min_bound = np.array([np.inf, np.inf, np.inf])
    max_bound = np.array([-np.inf, -np.inf, -np.inf])

    # 先扫描所有足模和鞋垫，计算全局边界
    for d in [feet_dir, insoles_dir]:
        if d is None or not d.exists():
            continue
        for file in d.glob("*.stl"):
            mesh = load_mesh(str(file))
            if voxel_size > 0:
                mesh = mesh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
            min_bound = np.minimum(min_bound, mesh.get_min_bound())
            max_bound = np.maximum(max_bound, mesh.get_max_bound())

    # 计算全局中心（基于所有足模和鞋垫的联合边界）
    center = (min_bound + max_bound) / 2
    scale = 2.0 / np.max(max_bound - min_bound)  # 缩放到 [-1,1]

    # 返回 shift = -center，这样归一化后所有 mesh 的中心都在原点附近
    return -center, scale


def normalize_mesh_global(mesh: o3d.geometry.TriangleMesh, shift: np.ndarray, scale: float):
    """将 mesh 顶点做仿射变换 (v+shift)*scale 并重建法线。"""
    vertices = np.asarray(mesh.vertices)
    vertices = (vertices + shift) * scale
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        mesh.triangles
    )
    mesh.compute_vertex_normals()
    return mesh


# ========== 核心处理函数 ==========
def process_single_pair(foot_path: Path,
                        insole_path: Optional[Path],
                        out_foot_path: Path,
                        out_insole_path: Optional[Path],
                        meta_out_no_ext: Path,
                        global_shift: np.ndarray,
                        global_scale: float,
                        align: bool = False,
                        clip_enabled: bool = False,
                        clip_height: Optional[float] = None,
                        template_path: Optional[Path] = None,
                        meta_format: str = "json",
                        save_rotation: bool = False,
                        icp_max_points: int = 5000,
                        icp_voxel_size: float = 2.0,
                        also_save_to_normalize_root: Optional[Path] = None,
                        save_per_sample_meta: bool = False):
    """
    处理一个样本对
    """
    log_info(f"处理 {foot_path.stem}")

    foot = load_mesh(str(foot_path))

    # === 基础清理 + 连通域离群组件剔除（足模） ===
    tri_before = len(foot.triangles)
    foot = basic_mesh_cleanup(foot)
    foot = keep_largest_connected_component_o3d(foot)
    tri_after = len(foot.triangles)
    if tri_after < tri_before:
        log_info(f"足模连通域清理: 三角形 {tri_before} -> {tri_after}")

    # 预加载并清理鞋垫（若存在）
    insole: Optional[o3d.geometry.TriangleMesh] = None
    if insole_path is not None:
        insole = load_mesh(str(insole_path))
        tri_b = len(insole.triangles)
        insole = basic_mesh_cleanup(insole)
        # 鞋垫同样只保留最大的连通组件，剔除离群碎片
        insole = keep_largest_connected_component_o3d(insole)
        tri_a = len(insole.triangles)
        if tri_a < tri_b:
            log_info(f"鞋垫基础清理: 三角形 {tri_b} -> {tri_a}")

    # === 对齐（可选） ===
    if align and insole is not None:
        trans = compute_icp_transform(foot, insole, voxel_size=icp_voxel_size, max_points=icp_max_points)
        foot.transform(trans)

    # === 高度裁剪（仅对足模生效） ===
    if clip_enabled and (clip_height is not None) and clip_height > 0:
        tri_b = len(foot.triangles)
        foot = clip_mesh_by_height_from_min_z(foot, float(clip_height))
        tri_a = len(foot.triangles)
        log_info(f"足模高度裁剪(H={clip_height}): 三角形 {tri_b} -> {tri_a}")

    # === 确保足模和鞋垫中心对齐（归一化前） ===
    if insole_path is not None and insole is not None:
        foot_center = foot.get_center()
        insole_center = insole.get_center()
        offset = foot_center - insole_center
        if np.linalg.norm(offset) > 1e-6:
            # 将鞋垫中心对齐到足模中心
            insole.translate(offset)
            log_info(f"鞋垫中心对齐: 偏移 {np.linalg.norm(offset):.3f}")

    # === 全局归一化 ===
    normalized_foot = normalize_mesh_global(foot, global_shift, global_scale)
    save_mesh(normalized_foot, str(out_foot_path))

    if insole_path is not None:
        # 若之前已加载并清理，则直接使用；否则兜底加载
        if insole is None:
            insole = load_mesh(str(insole_path))
            insole = basic_mesh_cleanup(insole)
            insole = keep_largest_connected_component_o3d(insole)
            # 对齐中心
            foot_center = foot.get_center()
            insole_center = insole.get_center()
            offset = foot_center - insole_center
            if np.linalg.norm(offset) > 1e-6:
                insole.translate(offset)
        normalized_insole = normalize_mesh_global(insole, global_shift, global_scale)
        save_mesh(normalized_insole, str(out_insole_path))

    # === per-sample meta（按需保存；默认关闭，仅保留全局 meta） ===
    if save_per_sample_meta:
        meta = {
            "global_shift": global_shift.tolist(),
            "global_scale": global_scale,
            "foot_file": str(out_foot_path),
            "insole_file": str(out_insole_path) if insole_path else None,
            "clip_enabled": bool(clip_enabled),
            "clip_height": float(clip_height) if (clip_height is not None) else None,
        }
        save_meta(meta, meta_out_no_ext, meta_format)


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="足模+鞋垫 统一归一化处理")
    parser.add_argument("--align", action="store_true", help="是否进行 ICP 对齐")
    parser.add_argument("--meta-format", choices=["json", "npz"], default="json")
    parser.add_argument("--save-rotation", action="store_true")
    parser.add_argument("--icp-max-points", type=int, default=5000)
    parser.add_argument("--icp-voxel-size", type=float, default=2.0)
    parser.add_argument("--clip-height", type=float, default=None, help="足模高度裁剪阈值H，按 (z-min_z)<=H 保留。")
    parser.add_argument("--no-clip", action="store_true", help="关闭足模高度裁剪。")
    args = parser.parse_args()

    # 初始化运行日志文件
    log_dir = Path("log")
    ensure_dir(log_dir)
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_dir / f"normalize_align_stl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    feet_dir = Path("data") / "raw" / "feet"
    insoles_dir = Path("data") / "raw" / "insoles"
    out_root = Path("data") / "normalize"

    ensure_dir(out_root / "feet")
    ensure_dir(out_root / "insoles")
    ensure_dir(out_root / "meta")

    # === 先计算全局归一化参数 ===
    log_info("计算全局归一化参数 ...")
    global_shift, global_scale = compute_global_normalization(feet_dir, insoles_dir, voxel_size=args.icp_voxel_size)
    log_info(f"全局 shift={global_shift}, scale={global_scale}")
    # 仅保存一份全局归一化参数，供后续反归一化使用
    save_meta({
        "global_shift": global_shift.tolist(),
        "global_scale": global_scale,
    }, out_root / "meta" / "global_normalization", args.meta_format)

    pairs = match_pairs_by_stem(feet_dir, insoles_dir)
    log_info(f"共匹配到 {len(pairs)} 个样本")

    processed_insole_names = set()
    for foot_path, insole_path in pairs:
        out_foot = out_root / "feet" / foot_path.name
        out_insole = out_root / "insoles" / insole_path.name if insole_path else None
        meta_out = out_root / "meta" / foot_path.stem  # 不再使用，仅兼容保留变量

        try:
            process_single_pair(
                foot_path=foot_path,
                insole_path=insole_path,
                out_foot_path=out_foot,
                out_insole_path=out_insole,
                meta_out_no_ext=meta_out,
                global_shift=global_shift,
                global_scale=global_scale,
                align=args.align,
                clip_enabled=(False if args.no_clip else (args.clip_height is not None and args.clip_height > 0)),
                clip_height=(None if args.no_clip else args.clip_height),
                meta_format=args.meta_format,
                save_rotation=args.save_rotation,
                icp_max_points=args.icp_max_points,
                icp_voxel_size=args.icp_voxel_size,
                also_save_to_normalize_root=out_root,
                save_per_sample_meta=False
            )
            if insole_path is not None:
                processed_insole_names.add(insole_path.name)
        except Exception as e:
            log_warn(f"处理 {foot_path} 失败: {e}")

    # 第二遍：处理未匹配到足模的鞋垫（仅做清理与全局归一化，不做对齐/裁剪）
    remaining_insoles = [p for p in insoles_dir.glob("*.stl") if p.name not in processed_insole_names]
    if len(remaining_insoles) > 0:
        log_info(f"额外处理未匹配足模的鞋垫: {len(remaining_insoles)}")
        for insole_path in remaining_insoles:
            try:
                insole = load_mesh(str(insole_path))
                insole = basic_mesh_cleanup(insole)
                insole = keep_largest_connected_component_o3d(insole)
                normalized_insole = normalize_mesh_global(insole, global_shift, global_scale)
                out_insole = out_root / "insoles" / insole_path.name
                save_mesh(normalized_insole, str(out_insole))
            except Exception as e:
                log_warn(f"处理鞋垫 {insole_path} 失败: {e}")

    log_info("全部处理完成。")


if __name__ == "__main__":
    main()
