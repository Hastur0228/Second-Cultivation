import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


class FootInsoleDataset(Dataset):
    """读取 data/dataset 下的样本与划分索引，按样本对提供训练数据。

    每条样本：
    - foot_points: (P, 3)
    - sample_points: (N, 3)
    - occupancy: (N,)
    本数据集在 __getitem__ 内部做查询点的均衡/随机子采样。
    """

    def __init__(self,
                 root: Path,
                 split: str,
                 num_queries: int = 4096,
                 balance_pos_fraction: float = 0.5,
                 seed: int = 42,
                 side_filter: Optional[str] = None,
                 renorm_zero_mean_unit: bool = False,
                 norm_check_only: bool = True):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.split = split
        self.num_queries = int(num_queries)
        self.balance_pos_fraction = float(balance_pos_fraction)
        self.rng = np.random.default_rng(seed)
        self.renorm_zero_mean_unit = bool(renorm_zero_mean_unit)
        self.norm_check_only = bool(norm_check_only)

        index_file = self.root / f"{split}.txt"
        with open(index_file, "r", encoding="utf-8") as f:
            stems = [line.strip() for line in f if line.strip()]
        if side_filter is not None:
            side = side_filter.upper()
            stems = [s for s in stems if _stem_has_side(s, side)]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def _balanced_subsample(self, pts: np.ndarray, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = pts.shape[0]
        q = min(self.num_queries, n)
        # 目标正样本数
        target_pos = int(q * self.balance_pos_fraction)
        pos_idx = np.where(occ == 1)[0]
        neg_idx = np.where(occ == 0)[0]
        # 随机采样（如数量不足则有放回采样）
        if pos_idx.size > 0 and target_pos > 0:
            sel_pos = self.rng.choice(pos_idx, size=min(target_pos, pos_idx.size), replace=False)
        else:
            sel_pos = np.array([], dtype=np.int64)
        remaining = q - sel_pos.size
        if neg_idx.size > 0 and remaining > 0:
            sel_neg = self.rng.choice(neg_idx, size=min(remaining, neg_idx.size), replace=False)
        else:
            sel_neg = np.array([], dtype=np.int64)
        sel = np.concatenate([sel_pos, sel_neg])
        # 若仍不足 q，则从全体补齐
        if sel.size < q:
            extra = self.rng.choice(np.arange(n), size=q - sel.size, replace=False)
            sel = np.concatenate([sel, extra])
        return pts[sel], occ[sel]

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        path = self.samples_dir / f"{stem}.npz"
        with np.load(path) as data:
            foot_points = data["foot_points"].astype(np.float32)  # (P,3)
            sample_points = data["sample_points"].astype(np.float32)  # (N,3)
            occupancy = data["occupancy"].astype(np.uint8)  # (N,)
        # 归一化一致性检查/可选重新归一化
        if self.norm_check_only or self.renorm_zero_mean_unit:
            # 检查是否大致落在 [-1,1]
            max_abs = float(np.max(np.abs(np.concatenate([foot_points, sample_points], axis=0))))
            if max_abs > 1.5 and not self.renorm_zero_mean_unit:
                # 仅检查模式下，报告一次明显越界
                print(f"[WARN] Sample {stem} appears not normalized (max_abs={max_abs:.2f}).")
            if self.renorm_zero_mean_unit:
                foot_points, sample_points = _renorm_zero_mean_unit(foot_points, sample_points)
        # 子采样查询点
        sample_points, occupancy = self._balanced_subsample(sample_points, occupancy)
        return (
            torch.from_numpy(foot_points),
            torch.from_numpy(sample_points),
            torch.from_numpy(occupancy.astype(np.float32)),
            stem,
        )


class SimplePointNetEncoder(nn.Module):
    """简易 PointNet 编码器：点特征 MLP + 全局 max pooling。"""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B,P,3)
        x = self.mlp(points)  # (B,P,F)
        x = torch.max(x, dim=1).values  # (B,F)
        return x


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B, C, N) -> idx: (B, N, K)
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B,N,N)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B,1,N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B,N,N)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,K)
    return idx


def _get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B, C, N) -> (B, 2C, N, K)
    B, C, N = x.size()
    idx = _knn(x, k)  # (B,N,K)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()  # (B,N,C)
    feature = x.view(B * N, C)[idx, :].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B,2C,N,K)
    return feature


class DGCNNEncoder(nn.Module):
    """DGCNN encoder to extract global feature from point clouds.

    Produces a global feature of size feat_dim via EdgeConv blocks and global pooling.
    """

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
        # points: (B,P,3)
        x = points.transpose(2, 1).contiguous()  # (B,3,N)
        x1 = _get_graph_feature(x, k=self.k)  # (B,6,N,K)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # (B,64,N)

        x2 = _get_graph_feature(x1, k=self.k)  # (B,128,N,K)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (B,64,N)

        x3 = _get_graph_feature(x2, k=self.k)  # (B,128,N,K)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # (B,128,N)

        x4 = _get_graph_feature(x3, k=self.k)  # (B,256,N,K)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # (B,256,N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B,64+64+128+256,N)
        x_feat = self.conv5(x_cat)  # (B,1024,N)
        x_global = torch.max(x_feat, dim=2)[0]  # (B,1024)
        out = self.fc(x_global)  # (B,feat_dim)
        return out
class OccupancyDecoder(nn.Module):
    """MLP 解码器：输入 query (x,y,z) 与全局特征 z，输出占据概率。"""

    def __init__(self, feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_points: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # query_points: (B,Q,3)  global_feat: (B,F)
        B, Q, _ = query_points.shape
        z = global_feat.unsqueeze(1).expand(B, Q, -1)
        x = torch.cat([query_points, z], dim=-1)
        logits = self.net(x)  # (B,Q,1)
        return logits.squeeze(-1)  # (B,Q)


class OccupancyNetwork(nn.Module):
    def __init__(self, feat_dim: int = 256, hidden: int = 256, encoder: str = "dgcnn", dgcnn_k: int = 20):
        super().__init__()
        if encoder == "dgcnn":
            self.encoder = DGCNNEncoder(feat_dim=feat_dim, k=dgcnn_k)
        else:
            self.encoder = SimplePointNetEncoder(feat_dim=feat_dim)
        self.decoder = OccupancyDecoder(feat_dim=feat_dim, hidden=hidden)

    def forward(self, foot_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        # foot_points: (B,P,3), query_points: (B,Q,3)
        z = self.encoder(foot_points)
        logits = self.decoder(query_points, z)
        return logits  # (B,Q)


def collate_batch(batch):
    foot_list, query_list, occ_list, stems = zip(*batch)
    foot = torch.stack(foot_list, dim=0)  # (B,P,3)
    query = torch.stack(query_list, dim=0)  # (B,Q,3)
    occ = torch.stack(occ_list, dim=0)  # (B,Q)
    return foot, query, occ, stems


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    bce = nn.BCEWithLogitsLoss()
    for foot, query, occ, _ in loader:
        foot = foot.to(device)
        query = query.to(device)
        occ = occ.to(device)
        logits = model(foot, query)
        loss = bce(logits, occ)
        total_loss += float(loss.item()) * occ.numel()
        pred = (torch.sigmoid(logits) >= 0.5).to(occ.dtype)
        total_correct += int((pred == occ).sum().item())
        total_count += int(occ.numel())
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def _stem_has_side(stem: str, side: str) -> bool:
    import re
    m = re.match(r"^(?P<prefix>.+)(?P<sep>[-_])foot(?P=sep)(?P<side>[lrLR])$", stem)
    if not m:
        return False
    return m.group("side").upper() == side.upper()


def _renorm_zero_mean_unit(foot_points_np: np.ndarray, sample_points_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.mean(foot_points_np, axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(foot_points_np - center, axis=1)))
    if scale <= 0:
        return foot_points_np, sample_points_np
    foot_norm = (foot_points_np - center) / scale
    sample_norm = (sample_points_np - center) / scale
    return foot_norm.astype(np.float32), sample_norm.astype(np.float32)


def train(args, side_override: Optional[str] = None, out_dir_override: Optional[Path] = None):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # 初始化运行日志文件
    log_dir = Path(args.out_dir) / "logs"
    ensure_dir(log_dir)
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_dir / f"train_occupancy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_info(f"Using device: {device}")

    dataset_root = Path(args.dataset_root)
    side_for_data = side_override if side_override is not None else args.side
    train_set = FootInsoleDataset(dataset_root, "train", num_queries=args.num_queries,
                                  balance_pos_fraction=args.pos_fraction, seed=args.seed,
                                  side_filter=side_for_data,
                                  renorm_zero_mean_unit=args.renorm_zero_mean_unit,
                                  norm_check_only=True)
    val_set = FootInsoleDataset(dataset_root, "val", num_queries=args.num_queries,
                                balance_pos_fraction=args.pos_fraction, seed=args.seed + 1,
                                side_filter=side_for_data,
                                renorm_zero_mean_unit=args.renorm_zero_mean_unit,
                                norm_check_only=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_batch)

    model = OccupancyNetwork(feat_dim=args.feat_dim, hidden=args.hidden, encoder=args.encoder, dgcnn_k=args.dgcnn_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor(float(args.pos_weight), device=device) if args.pos_weight is not None else None
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    out_dir = Path(args.out_dir) if out_dir_override is None else Path(out_dir_override)
    ensure_dir(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_correct = 0
        running_count = 0
        running_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for foot, query, occ, _ in pbar:
            foot = foot.to(device)
            query = query.to(device)
            occ = occ.to(device)

            query.requires_grad_(args.lambda_smooth > 0.0)
            logits = model(foot, query)
            loss = bce(logits, occ)
            if args.lambda_smooth > 0.0:
                grad = torch.autograd.grad(outputs=logits, inputs=query,
                                           grad_outputs=torch.ones_like(logits),
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                reg = grad.norm(dim=-1).mean()
                loss = loss + float(args.lambda_smooth) * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).to(occ.dtype)
                acc = (pred == occ).float().mean().item()
                running_correct += int((pred == occ).sum().item())
                running_count += int(occ.numel())
                running_loss_sum += float(loss.item()) * int(occ.numel())
                avg_acc = running_correct / max(1, running_count)
                avg_loss = running_loss_sum / max(1, running_count)
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
                "avg_acc": f"{avg_acc:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })

            global_step += 1

        # 验证（使用与训练相同的 pos_weight）
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for foot, query, occ, _ in val_loader:
                foot = foot.to(device)
                query = query.to(device)
                occ = occ.to(device)
                logits = model(foot, query)
                loss = bce(logits, occ)
                total_loss += float(loss.item()) * occ.numel()
                pred = (torch.sigmoid(logits) >= 0.5).to(occ.dtype)
                total_correct += int((pred == occ).sum().item())
                total_count += int(occ.numel())
        val_loss = total_loss / max(1, total_count)
        val_acc = total_correct / max(1, total_count)
        log_info(f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")
        scheduler.step()

        # 保存最新与最佳
        last_ckpt = ckpt_dir / "last.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_acc": val_acc,
        }, last_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = ckpt_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_ckpt)
            log_info(f"Saved best checkpoint: acc={val_acc:.4f}")

        # 可选：保存完整 epoch checkpoint
        if args.save_all:
            full_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, full_ckpt)


def main():
    parser = argparse.ArgumentParser(description="Occupancy Network 训练脚本（足模点云条件）")
    parser.add_argument("--dataset-root", type=str, default=str(Path("data") / "dataset"))
    parser.add_argument("--out-dir", type=str, default=str(Path("outputs") / "occnet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--encoder", type=str, choices=["pointnet", "dgcnn"], default="dgcnn", help="编码器类型")
    parser.add_argument("--dgcnn-k", type=int, default=20, help="DGCNN KNN 的 k 值")
    parser.add_argument("--num-queries", type=int, default=16384, help="每样本训练时使用的查询点数")
    parser.add_argument("--pos-fraction", type=float, default=0.5, help="查询点中正样本目标占比")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="BCE 正样本权重 (>=1 强化正样本)")
    parser.add_argument("--lambda-smooth", type=float, default=0.0, help="平滑正则项权重 (0 关闭)")
    parser.add_argument("--step-size", type=int, default=20, help="LR StepLR 的 step_size")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR StepLR 的 gamma")
    parser.add_argument("--save-all", action="store_true", help="保存每个 epoch 的完整 checkpoint")
    parser.add_argument("--renorm-zero-mean-unit", action="store_true", help="在加载 .npz 时对 foot/sample 做零均值单位球归一化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    # 左右脚分别训练；支持一次性顺序训练两侧
    parser.add_argument("--side", type=str, choices=["L", "R", "both"], default="both", help="训练哪个侧别（both 表示先 L 后 R）")
    args = parser.parse_args()

    if args.side == "both":
        # 先左后右，分别输出到 out_dir/L 与 out_dir/R
        base_out = Path(args.out_dir)
        log_info("Training LEFT side ...")
        train(args, side_override="L", out_dir_override=base_out / "L")
        log_info("Training RIGHT side ...")
        train(args, side_override="R", out_dir_override=base_out / "R")
    else:
        ensure_dir(Path(args.out_dir))
        train(args)


if __name__ == "__main__":
    main()


