# 足模占据网络项目说明

本项目实现了一条从原始足模/鞋垫网格数据到占据网络训练与鞋垫重建的完整流水线，核心目标是基于足模点云条件预测鞋垫体素占据概率并重建三维网格。下文总结了工程中使用到的模型、算法、损失函数及配套脚本。

## 目录与主要脚本
- `scripts/normalize_align_stl.py`：原始 STL 清理、可选 ICP 对齐、全局归一化。
- `scripts/build_dataset.py`：占据采样、正负样本平衡、数据集划分。
- `scripts/train_occupancy.py`：占据网络（DGCNN / PointNet 编码 + MLP 解码）训练与验证。
- `scripts/generate_insole.py`：基于训练好的模型进行体素评估与 Marching Cubes 重建。
- `scripts/visualize_samples.py`：可视化 `.npz` 样本中的足模点云与占据标签。
- `scripts/crawler.py`：示例性数据抓取脚本（非训练核心流程）。

## 算法与模型
### 数据采样与处理
- **网格清理与归一化**：利用 Open3D 移除退化/重复/非流形构件，保留最大连通组件，并可选通过 ICP 对齐足模与鞋垫；所有样本共享 `global_shift` 和 `global_scale` 将几何体映射到近似 `[-1,1]` 的归一化空间。
- **占据采样策略**：针对每个鞋垫，综合近表面扰动采样、AABB 内部/外部均匀采样，并使用 `open3d.t.geometry.RaycastingScene.compute_occupancy`（回退到 signed distance）标注占据标签；对足模表面使用均匀或泊松采样提取点云特征。
- **平衡子采样**：训练数据加载时对查询点进行正负样本比例控制，确保网络在稀疏占据场景下的稳定学习。

### 占据网络模型
- **编码器 (Encoder)**：
  - *DGCNNEncoder*（默认）：基于 EdgeConv 的动态图卷积网络。通过 KNN 动态建图、计算局部边特征并叠加四层 EdgeConv → 全局 MaxPool → 全连接获取全局特征。
  - *SimplePointNetEncoder*：轻量级 PointNet 风格的逐点 MLP + 全局 MaxPool，作为 DGCNN 的替代选项。
- **解码器 (Decoder)**：三层 MLP 将查询点坐标与编码器输出的全局特征拼接后，输出占据概率（Logits）。
- **条件建模**：模型输入为足模点云 (B, P, 3) 与查询点 (B, Q, 3)，输出占据 logits (B, Q)，形成条件 Occupancy Network。

### 损失函数与训练策略
- **主损失**：`nn.BCEWithLogitsLoss`，可配置 `pos_weight` 强化正样本。
- **平滑正则**：可选 `lambda_smooth`，对 logits 相对于查询点的梯度范数进行惩罚，鼓励隐式场的空间平滑性。
- **优化器/调度器**：Adam（带权重衰减），搭配 StepLR（默认每 20 epoch ×0.5）；训练时记录逐步准确率与损失。
- **评估指标**：验证阶段计算平均 BCE 损失与二分类准确率；在训练与验证之间自动保存 `best.pt`、`last.pt` 以及按需的逐 epoch checkpoint。

## 数据构建流水线
1. **原始数据准备**：可通过 `crawler.py` 获取 STL 文件，放入 `data/raw/feet` 与 `data/raw/insoles`。
2. **归一化处理**：运行 `normalize_align_stl.py` 清理网格、可选脚本参数 `--align` 做 ICP 对齐，输出到 `data/normalize` 并生成全局归一化参数。
3. **占据样本生成**：执行 `build_dataset.py` 将归一化后的网格采样为 `.npz`（足模点云、查询点、占据标签），并依据配置划分 train/val/test。
4. **可视化验证**：使用 `visualize_samples.py` 检查采样质量、占据分布与归一化范围。

## 训练与验证流程
1. 准备 Python 环境安装 `requirements.txt` 列出的依赖（NumPy、PyTorch、Open3D、scikit-image 等）。
2. 运行 `python scripts/train_occupancy.py --dataset-root data/dataset --out-dir outputs/occnet`。
3. 可通过 `--encoder pointnet`、`--pos-weight`、`--lambda-smooth` 等参数调整模型结构与损失配置。
4. 日志与 checkpoint 默认写入 `outputs/occnet/logs`、`outputs/occnet/checkpoints`，每次训练记录最优指标。

## 鞋垫重建与导出
1. 选择最佳 checkpoint（默认 `outputs/occnet_both/L|R/checkpoints/best.pt`），运行 `generate_insole.py`。
2. 流程：对归一化足模采样 → 构建 3D 查询网格 → Occupancy Network 批量预测 → 以分位阈值 (`mc-quantile`) 生成体素 → `marching_cubes` 提取等值面 → 根据全局参数反归一化 → 输出 STL。
3. 支持批量模式（`--foot-dir`）与单样本模式（`--foot-file` + `--out-mesh`），并可针对坐标系差异进行轴向翻转。

## 日志与实验管理
- 脚本统一使用 `log/` 目录记录时间戳命名的运行日志，便于追踪数据构建、训练与生成过程。
- Checkpoint 默认包括 `best.pt`（最高验证精度）与 `last.pt`（最近一次），可选 `--save-all` 保存每个 epoch。

## 快速命令示例
```bash
# 1. 网格归一化
python scripts/normalize_align_stl.py --align

# 2. 构建占据数据集
python scripts/build_dataset.py --normalized-root data/normalize --aggregate

# 3. 训练占据网络
python scripts/train_occupancy.py --dataset-root data/dataset --out-dir outputs/occnet_both

# 4. 批量重建鞋垫
python scripts/generate_insole.py --foot-dir test/foot --out-dir test/output
```

## 常见问题与建议
- Occupancy 标签计算依赖 GPU 支持的 Open3D；若出现显存问题，可降低 `--num-occupancy` 或调整 `pool_multiplier`。
- 若输入足模未归一化，可在生成阶段使用 `--force-zero-mean-unit` 进行零均值单位球标准化，保证模型输入分布与训练一致。
- 针对正样本稀缺的场景，应提高 `--pos-weight` 并启用更高的 `--pos-fraction` 以改善召回。
- 推理网格分辨率会显著影响生成细节与耗时，可依据需求调整 `--grid-res` 并适当减小 `--chunk-size` 避免显存不足。

## 依赖与环境
- Python >= 3.10，需安装 `numpy`、`torch`、`open3d`、`scikit-image`、`tqdm` 等依赖，详见 `requirements.txt`。
- 训练与大规模推理推荐使用 GPU；所有脚本均提供 `--cpu` 开关以在无 GPU 时运行（性能较低）。

通过以上流程，项目实现了足模条件的 Occupancy Network 建模、训练与鞋垫重建，适用于个性化定制鞋垫或类似的隐式几何生成场景。
