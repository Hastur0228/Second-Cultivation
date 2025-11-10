import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def visualize_npz(npz_path: Path,
                  save_dir: Optional[Path] = None,
                  no_window: bool = False,
                  downsample_ratio: float = 1.0) -> None:
    with np.load(npz_path) as data:
        foot_points = data["foot_points"].astype(np.float64)
        sample_points = data["sample_points"].astype(np.float64)
        occupancy = data["occupancy"].astype(np.uint8)

    if 0.0 < downsample_ratio < 1.0:
        num_fp = foot_points.shape[0]
        num_sp = sample_points.shape[0]
        fp_take = max(1, int(num_fp * downsample_ratio))
        sp_take = max(1, int(num_sp * downsample_ratio))
        fp_idx = np.random.default_rng(42).choice(num_fp, size=fp_take, replace=False)
        sp_idx = np.random.default_rng(43).choice(num_sp, size=sp_take, replace=False)
        foot_points = foot_points[fp_idx]
        sample_points = sample_points[sp_idx]
        occupancy = occupancy[sp_idx]

    foot_pcd = o3d.geometry.PointCloud()
    foot_pcd.points = o3d.utility.Vector3dVector(foot_points)
    foot_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    inside = sample_points[occupancy == 1]
    outside = sample_points[occupancy == 0]

    inside_pcd = o3d.geometry.PointCloud()
    inside_pcd.points = o3d.utility.Vector3dVector(inside)
    inside_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    outside_pcd = o3d.geometry.PointCloud()
    outside_pcd.points = o3d.utility.Vector3dVector(outside)
    outside_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    if not no_window:
        try:
            o3d.visualization.draw_geometries([foot_pcd, inside_pcd, outside_pcd])
        except Exception:
            pass

    if save_dir is not None:
        ensure_dir(save_dir)
        stem = npz_path.stem
        o3d.io.write_point_cloud(str(save_dir / f"{stem}_foot.ply"), foot_pcd)
        o3d.io.write_point_cloud(str(save_dir / f"{stem}_inside.ply"), inside_pcd)
        o3d.io.write_point_cloud(str(save_dir / f"{stem}_outside.ply"), outside_pcd)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 .npz 样本的足模点云与占据采样点")
    parser.add_argument("--save-dir", type=str, default=None, help="将点云导出为 PLY 的目录（可选）")
    parser.add_argument("--no-window", action="store_true", help="禁用窗口，仅落盘 PLY")
    parser.add_argument("--downsample-ratio", type=float, default=1.0, help="可选下采样比例 (0,1]")
    args = parser.parse_args()

    # 交互式输入文件路径
    npz_input = input("请输入要可视化的 .npz 文件路径: ").strip()
    if not npz_input:
        print("[ERROR] 未输入文件路径")
        return
    
    npz_path = Path(npz_input)
    if not npz_path.exists():
        print(f"[ERROR] 文件不存在: {npz_path}")
        return
    
    if not npz_path.suffix == ".npz":
        print(f"[WARN] 文件扩展名不是 .npz: {npz_path}")
    
    save_dir = Path(args.save_dir) if args.save_dir else None
    visualize_npz(npz_path, save_dir=save_dir, no_window=bool(args.no_window), downsample_ratio=float(args.downsample_ratio))


if __name__ == "__main__":
    main()


