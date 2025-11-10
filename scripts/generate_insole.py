"""
基于已训练的 Occupancy Network，从足模点云条件生成鞋垫网格：
1) 读取并采样足模点云（默认 4096 点）
2) 在 AABB(+margin) 三维网格上评估占据概率
3) 用 Marching Cubes 提取 0.5 等值面并保存为 STL
4) 输入足模若未归一化，则依据全局参数做归一化；输出默认反归一化回原坐标
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import marching_cubes
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


def log_info(msg: str) -> None:
    line = f"[INFO] {msg}"
    print(line)
    _append_log(line)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_global_meta(meta_file: Path) -> Tuple[np.ndarray, float]:
    if meta_file.suffix == ".json":
        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        shift = np.array(meta["global_shift"], dtype=np.float32)
        scale = float(meta["global_scale"])
        return shift, scale
    elif meta_file.suffix == ".npz":
        data = np.load(meta_file)
        shift = data["global_shift"].astype(np.float32)
        scale = float(data["global_scale"])
        return shift, scale
    else:
        raise ValueError(f"Unsupported meta file: {meta_file}")


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"空 mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_surface_points(mesh: o3d.geometry.TriangleMesh, num_points: int) -> np.ndarray:
    pcd = mesh.sample_points_uniformly(number_of_points=int(num_points))
    return np.asarray(pcd.points, dtype=np.float32)


def aabb_with_margin(mesh: o3d.geometry.TriangleMesh, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    aabb = mesh.get_axis_aligned_bounding_box()
    mn = np.asarray(aabb.get_min_bound(), dtype=np.float32)
    mx = np.asarray(aabb.get_max_bound(), dtype=np.float32)
    center = (mn + mx) / 2.0
    half = (mx - mn) / 2.0
    half = half * (1.0 + float(margin))
    mn2 = center - half
    mx2 = center + half
    return mn2, mx2


class SimplePointNetEncoder(nn.Module):
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.mlp(points)  # (B,P,F)
        x = torch.max(x, dim=1).values  # (B,F)
        return x


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def _get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    B, C, N = x.size()
    idx = _knn(x, k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(B * N, C)[idx, :].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNNEncoder(nn.Module):
    def __init__(self, feat_dim: int = 256, k: int = 20):
        super().__init__()
        self.k = int(k)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(2, 1).contiguous()
        x1 = _get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = _get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = _get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = _get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_feat = self.conv5(x_cat)
        x_global = torch.max(x_feat, dim=2)[0]
        out = self.fc(x_global)
        return out
class OccupancyDecoder(nn.Module):
    def __init__(self, feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_points: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        B, Q, _ = query_points.shape
        z = global_feat.unsqueeze(1).expand(B, Q, -1)
        x = torch.cat([query_points, z], dim=-1)
        logits = self.net(x)
        return logits.squeeze(-1)


class OccupancyNetwork(nn.Module):
    def __init__(self, feat_dim: int = 256, hidden: int = 256, encoder: str = "dgcnn", dgcnn_k: int = 20):
        super().__init__()
        if encoder == "dgcnn":
            self.encoder = DGCNNEncoder(feat_dim=feat_dim, k=dgcnn_k)
        else:
            self.encoder = SimplePointNetEncoder(feat_dim=feat_dim)
        self.decoder = OccupancyDecoder(feat_dim=feat_dim, hidden=hidden)

    def forward(self, foot_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        z = self.encoder(foot_points)
        logits = self.decoder(query_points, z)
        return logits


def grid_query(min_xyz: np.ndarray, max_xyz: np.ndarray, res: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    nx, ny, nz = int(res[0]), int(res[1]), int(res[2])
    xs = np.linspace(min_xyz[0], max_xyz[0], nx, dtype=np.float32)
    ys = np.linspace(min_xyz[1], max_xyz[1], ny, dtype=np.float32)
    zs = np.linspace(min_xyz[2], max_xyz[2], nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, (xs, ys, zs)


def volume_from_probs(probs: np.ndarray, res: Tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = res
    vol = probs.reshape(ny, nx, nz).transpose(2, 0, 1)  # -> (nz, ny, nx)
    return vol


def verts_to_world(verts_zyx: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    vz, vy, vx = verts_zyx.T
    dx = (xs[-1] - xs[0]) / max(1, len(xs) - 1)
    dy = (ys[-1] - ys[0]) / max(1, len(ys) - 1)
    dz = (zs[-1] - zs[0]) / max(1, len(zs) - 1)
    xw = xs[0] + vx * dx
    yw = ys[0] + vy * dy
    zw = zs[0] + vz * dz
    return np.stack([xw, yw, zw], axis=-1).astype(np.float32)


def denormalize_vertices(verts: np.ndarray, shift: np.ndarray, scale: float) -> np.ndarray:
    # 训练归一化在 normalize 脚本中是: x_norm = (x + shift) * scale
    # 因此反归一化为: x = x_norm / scale - shift
    return (verts / float(scale) - shift.astype(np.float32)).astype(np.float32)


def zero_mean_unit_norm(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    if scale <= 0:
        return points.astype(np.float32), center.astype(np.float32), 1.0
    return ((points - center) / scale).astype(np.float32), center.astype(np.float32), scale


def main():
    parser = argparse.ArgumentParser(description="Occupancy Network 生成鞋垫网格")
    # checkpoint：可单一或按左右分别给出；若未给 --checkpoint，则使用默认左右 checkpoint
    parser.add_argument("--checkpoint", type=str, default=None, help="单一 checkpoint（若提供则覆盖左右）")
    parser.add_argument("--ckpt-left", type=str, default=str(Path("outputs") / "occnet_both" / "L" / "checkpoints" / "best.pt"), help="左脚 checkpoint 默认路径")
    parser.add_argument("--ckpt-right", type=str, default=str(Path("outputs") / "occnet_both" / "R" / "checkpoints" / "best.pt"), help="右脚 checkpoint 默认路径")
    # 输入：单文件或目录（二选一；若未提供 foot-file，则默认用目录 test/foot 下全部 .stl）
    parser.add_argument("--foot-file", type=str, default=None, help="归一化后的足模 STL 路径（单文件）")
    parser.add_argument("--foot-dir", type=str, default=str(Path("test") / "foot"), help="批量模式的足模目录")
    parser.add_argument("--num-foot-points", type=int, default=4096)
    parser.add_argument("--grid-res", type=int, nargs="*", default=[160], help="体素网格分辨率：单值或 3 值 Nx Ny Nz")
    parser.add_argument("--bbox-margin", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--encoder", type=str, choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--dgcnn-k", type=int, default=20)
    # 输出：单文件或目录（若为目录，默认 test/output）
    parser.add_argument("--out-mesh", type=str, default=None, help="单文件输出路径（仅在 --foot-file 模式下使用）")
    parser.add_argument("--out-dir", type=str, default=str(Path("test") / "output"), help="批量模式输出目录")
    # 反归一化：默认开启，可用 --no-denorm 关闭；并可按侧别分别使用不同 checkpoint
    parser.add_argument("--denorm", dest="denorm", action="store_true")
    parser.add_argument("--no-denorm", dest="denorm", action="store_false")
    parser.set_defaults(denorm=True)
    parser.add_argument("--meta-file", type=str, default=str(Path("data") / "normalize" / "meta" / "global_normalization.json"))
    parser.add_argument("--side", type=str, choices=["L", "R", "auto"], default="auto", help="选择侧别或自动从文件名推断")
    # 可选坐标修正（如出现镜像/轴方向反）：在最终输出坐标系中翻转对应轴
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    # 归一化一致性检查/可选零均值单位球归一化（与训练保持一致）
    parser.add_argument("--check-norm", action="store_true", help="打印输入是否近似归一化")
    parser.add_argument("--force-zero-mean-unit", action="store_true", help="对输入点云强制零均值单位球归一化")
    # Marching Cubes 等值面量化分位（更稳定的阈值选择）
    parser.add_argument("--mc-quantile", type=float, default=0.3, help="体素值分位阈值 (0~1)，默认 0.3")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(Path("log")), help="日志输出目录")
    args = parser.parse_args()

    # 初始化运行日志文件
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_dir / f"generate_insole_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log_info(f"Using device: {device}")

    # 模型缓存：按侧别加载对应 checkpoint
    def load_model_for_side(side: str) -> OccupancyNetwork:
        ckpt_path = args.checkpoint if args.checkpoint is not None else (args.ckpt_left if side == "L" else args.ckpt_right)
        net = OccupancyNetwork(feat_dim=args.feat_dim, hidden=args.hidden, encoder=args.encoder, dgcnn_k=args.dgcnn_k).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        net.load_state_dict(state, strict=False)
        net.eval()
        return net

    model_cache = {}

    # 加载全局参数：用于输入归一化与（可选）输出反归一化
    shift: Optional[np.ndarray] = None
    scale: Optional[float] = None
    shift, scale = load_global_meta(Path(args.meta_file))

    def parse_side_from_stem(stem: str) -> Optional[str]:
        import re as _re
        m = _re.match(r"^(?P<prefix>.+)(?P<sep>[-_])foot(?P=sep)(?P<side>[lrLR])$", stem)
        if m:
            return m.group("side").upper()
        return None

    def map_foot_to_insole_stem(stem: str) -> str:
        import re as _re
        m = _re.match(r"^(?P<prefix>.+)(?P<sep>[-_])foot(?P=sep)(?P<side>[lrLR])$", stem)
        if m:
            prefix = m.group("prefix")
            sep = m.group("sep")
            side = m.group("side").upper()
            return f"{prefix}{sep}insole{sep}{side}"
        return f"{stem}_insole"

    def process_one(foot_file: Path, out_file: Optional[Path] = None):
        # 加载并采样足模点云
        foot_mesh = load_mesh(foot_file)
        # 若输入为原始坐标，则先做归一化
        # 统一归一化：始终应用 (x + shift) * scale 到输入足模
        verts = np.asarray(foot_mesh.vertices, dtype=np.float32)
        verts_norm = (verts + shift.astype(np.float32)) * float(scale)
        foot_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts_norm.astype(np.float64)),
            triangles=foot_mesh.triangles,
        )
        foot_mesh.compute_vertex_normals()
        foot_pts = sample_surface_points(foot_mesh, args.num_foot_points)
        if args.force_zero_mean_unit:
            foot_pts, _, _ = zero_mean_unit_norm(foot_pts)
        elif args.check_norm:
            max_abs = float(np.max(np.abs(foot_pts)))
            if max_abs > 1.5:
                log_info(f"[WARN] 输入足模点云似乎未归一化，max_abs={max_abs:.2f}")

        # 评估网格范围：在标准归一化空间内覆盖完整体积（基于训练规范）
        half = float(1.0 + args.bbox_margin)
        mn = np.array([-half, -half, -half], dtype=np.float32)
        mx = np.array([ half,  half,  half], dtype=np.float32)

        # 构建查询网格
        if len(args.grid_res) == 1:
            res = (args.grid_res[0], args.grid_res[0], args.grid_res[0])
        elif len(args.grid_res) == 3:
            res = (args.grid_res[0], args.grid_res[1], args.grid_res[2])
        else:
            raise ValueError("--grid-res 需为 1 值或 3 值")
        query_pts, (xs, ys, zs) = grid_query(mn, mx, res)

        # 侧别解析与模型选择
        inferred_side = parse_side_from_stem(foot_file.stem)
        eff_side = None
        if args.side in ("L", "R"):
            eff_side = args.side
        else:
            if inferred_side is None:
                raise ValueError(f"无法从文件名解析侧别（期望 *_foot_L/_R）: {foot_file}")
            eff_side = inferred_side
        if eff_side not in model_cache:
            model_cache[eff_side] = load_model_for_side(eff_side)
        model = model_cache[eff_side]
        # 调试：点云归一化后分布
        try:
            log_info(f"foot_pts stats: min {float(foot_pts.min()):.3f}, max {float(foot_pts.max()):.3f}, mean {float(foot_pts.mean()):.3f}")
        except Exception:
            pass
        foot_tensor = torch.from_numpy(foot_pts).unsqueeze(0).to(device)
        probs_all = []
        with torch.no_grad():
            for i in range(0, query_pts.shape[0], args.chunk_size):
                chunk = query_pts[i:i + args.chunk_size]
                q = torch.from_numpy(chunk).unsqueeze(0).to(device)
                logits = model(foot_tensor, q)
                probs = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()
                probs_all.append(probs)
        probs_all = np.concatenate(probs_all, axis=0)
        try:
            med = float(np.median(probs_all))
            frac = float((probs_all > 0.5).mean())
            log_info(f"probs stats: min {float(probs_all.min()):.3f}, max {float(probs_all.max()):.3f}, mean {float(probs_all.mean()):.3f}, med {med:.3f}, frac>0.5 {frac:.3f}")
        except Exception:
            pass

        # 体素体积与 Marching Cubes
        vol = volume_from_probs(probs_all, res)
        vol_min, vol_max = float(np.min(vol)), float(np.max(vol))
        log_info(f"Volume range: [{vol_min:.3f}, {vol_max:.3f}]")
        if vol_min == vol_max:
            log_info("[ERROR] Volume is constant; no isosurface can be extracted. Skipping file.")
            return
        # 使用更稳健的分位阈值（默认 0.3），避免 0.5 阈值导致残缺
        q = float(np.clip(args.mc_quantile, 0.0, 1.0))
        level = float(np.quantile(vol, q))
        try:
            verts_zyx, faces, _, _ = marching_cubes(vol, level=level)
        except ValueError as e:
            log_info(f"[ERROR] Marching cubes failed: {e}. Skipping {foot_file}")
            return  # Skip this file
        verts = verts_to_world(verts_zyx, xs, ys, zs)

        # 可选反归一化
        if args.denorm and shift is not None and scale is not None:
            verts = denormalize_vertices(verts, shift, scale)

        # 坐标可选翻转（在最终坐标系中进行）
        if args.flip_x or args.flip_y or args.flip_z:
            sx = -1.0 if args.flip_x else 1.0
            sy = -1.0 if args.flip_y else 1.0
            sz = -1.0 if args.flip_z else 1.0
            verts = verts * np.array([sx, sy, sz], dtype=np.float32)

        # 输出路径
        if out_file is None:
            out_dir = Path(args.out_dir)
            ensure_dir(out_dir)
            out_stem = map_foot_to_insole_stem(foot_file.stem)
            out_path = out_dir / f"{out_stem}.stl"
        else:
            out_path = out_file

        # 保存 STL
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
        )
        mesh.compute_vertex_normals()
        ensure_dir(out_path.parent)
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        log_info(f"Saved mesh -> {out_path}")

    # 模式选择：单文件或目录批量
    if args.foot_file is not None:
        out_file = Path(args.out_mesh) if args.out_mesh is not None else None
        process_one(Path(args.foot_file), out_file)
    else:
        in_dir = Path(args.foot_dir)
        files = sorted(in_dir.glob("*.stl"))
        if args.side in ("L", "R"):
            files = [f for f in files if parse_side_from_stem(f.stem) == args.side]
        if len(files) == 0:
            log_info(f"未在目录中发现匹配的 STL: {in_dir}")
        for f in files:
            process_one(f, None)


if __name__ == "__main__":
    main()


