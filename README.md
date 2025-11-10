# 项目总览

本仓库围绕“足模 → 鞋垫”生成流程搭建了一整套 Occupancy Network 管线：\
从原始 STL 预处理、体素/点云数据集构建、网络训练到基于概率场的鞋垫网格生成与可视化。本文档重点梳理项目中使用的模型、算法、损失函数及运行方式，便于快速理解与复现。

## 代码结构
- `scripts/normalize_align_stl.py`：原始 STL 清理、连通域筛选、可选 ICP 对齐与全局归一化。
- `scripts/build_dataset.py`：利用 Open3D 光线投射生成占据标签，按“近表面/内部/外部”分区采样构建 `.npz` 数据集。
- `scripts/train_occupancy.py`：足模条件 Occupancy Network 训练脚本，支持 PointNet 或 DGCNN 编码器。
- `scripts/generate_insole.py`：加载训练权重，对三维网格查询占据概率并用 Marching Cubes 抽取鞋垫等值面。
- `scripts/visualize_samples.py`：可视化 `.npz` 样本的足模点云与占据点分布。
- `scripts/crawler.py`：示例爬虫脚本，用于批量抓取 STL 数据（与建模流程解耦）。

## 数据处理与预处理算法
1. **网格清理**：移除退化/重复/非流形三角形并保留最大连通组件，确保 STL 几何一致性。
2. **可选 ICP 对齐**：当足模与鞋垫存在姿态差异时，使用 Open3D 的点到点 ICP 估计刚体变换，使两者在同一坐标系下工作。
3. **高度裁剪**：可将足模顶部裁剪至指定高度，抑制离群部分对归一化的影响。
4. **全局归一化**：扫描所有足模与鞋垫，计算联合 AABB；通过 `(x + shift) * scale` 将数据压缩到近似 `[-1, 1]^3`，并保存 `shift/scale` 以便后续反归一化。

## 数据集构建策略
- **分层采样**：鞋垫体积采样按“表面附近/内部/外部”比例抽样，确保训练时能见到足够的决策边界信息。
- **占据标注**：通过 `open3d.t.geometry.RaycastingScene` 的 `compute_occupancy` 或 SDF 回退方案，为每个查询点生成二值占据标签。
- **查询点均衡**：训练阶段进一步按设定的正例比例（默认 0.5）对子采样，缓解内部/外部样本数量不平衡。

## 模型架构
1. **编码器（Conditioning Network）**
   - `SimplePointNetEncoder`：MLP + 全局 max pooling 的轻量 PointNet 变体，输出固定长度的全局特征。
   - `DGCNNEncoder`：基于 EdgeConv 的动态图卷积网络，逐层构建 kNN 图并提取局部几何，通过 concat + 全局池化得到 256 维特征。
2. **解码器（Occupancy Decoder）**\
   - 将查询点坐标与全局特征拼接，使用 2 层 ReLU 的 MLP 预测占据对数几率。
3. **组合**：形成条件 Occupancy Network，输入 `(足模点云, 查询点)`，输出每个查询点的占据概率分布。

## 损失函数与正则化
- **主体损失**：`BCEWithLogitsLoss`，可选 `pos_weight` 提升正样本权重，以缓解数据不平衡。[默认取 1.0，可通过命令行调整]
- **梯度平滑正则（可选）**：当 `--lambda-smooth > 0` 时，对 `logits` 相对于查询点的梯度取 L2 范数均值，实现 Occupancy 场的空间平滑约束。

## 训练流程
1. **优化器**：Adam（学习率默认 `1e-3`，权重衰减 `1e-5`）。
2. **学习率调度**：StepLR（默认每 20 个 epoch 将 LR 乘以 0.5）。
3. **度量**：训练与验证阶段计算损失及基于 0.5 阈值的占据分类准确率。
4. **Checkpoint 策略**：始终保存 `last.pt`；当验证精度提升时覆盖 `best.pt`；可选 `--save-all` 逐 epoch 记录。
5. **左右脚顺序训练**：`--side both` 时自动分别训练左右脚模型，输出到 `outputs/occnet/L` 与 `outputs/occnet/R`。

## 推理与鞋垫生成
1. **输入处理**：读取足模 STL，应用与训练一致的归一化；可额外做零均值单位球规范。
2. **体积查询**：在归一化空间内构建规则网格（默认单轴 160 分辨率），分块送入 Occupancy Network 计算概率。
3. **阈值策略**：统计概率体素，使用可配置分位数（默认 0.3）作为 Marching Cubes 的等值面阈值，提升网格完整度。
4. **Marching Cubes**：`skimage.measure.marching_cubes` 抽取等值面，随后（可选）反归一化并导出 STL。
5. **侧别模型选择**：支持单 checkpoint 或按左右分别加载；自动从文件名推断 `*_foot_L/R` 侧别。

## 可视化与辅助脚本
- `.npz` 样本中的足模点云与正/负查询点可通过 `visualize_samples.py` 转换为 PLY 或直接窗口展示。
- `crawler.py` 展示了如何使用 DrissionPage 自动化采集 STL，但不参与核心训练/推理流程。

## 环境依赖
项目依赖见 `requirements.txt`，核心库包含：
- PyTorch ≥ 2.1（深度模型与训练）
- Open3D ≥ 0.18（网格处理、Occupancy 标注、点云操作）
- NumPy / SciPy、scikit-image（矩阵运算、Marching Cubes）
- tqdm（训练进度条显示）

## 典型使用流程
```bash
# 1. STL 归一化（可选对齐/裁剪参数按需调整）
python scripts/normalize_align_stl.py --align --clip-height 110

# 2. 构建 Occupancy 数据集
python scripts/build_dataset.py --normalized-root data/normalize --out-root data/dataset --aggregate

# 3. 训练 Occupancy Network
python scripts/train_occupancy.py --dataset-root data/dataset --out-dir outputs/occnet --encoder dgcnn --lambda-smooth 0.01

# 4. 基于训练模型生成鞋垫
python scripts/generate_insole.py --foot-dir test/foot --out-dir test/output --grid-res 160 --mc-quantile 0.3
```

## 日志与结果
- 所有脚本均在 `log/` 下生成带时间戳的日志文件，记录流程细节与异常信息。
- 训练权重按照 `outputs/occnet/<side>/checkpoints/` 组织，易于部署或继续训练。

## 扩展建议
- 可替换或扩展编码器（例如加入 Transformer-风格点云编码）以探索更高的几何建模能力。
- 强化数据增强（噪声扰动、镜像等）以提升泛化。
- 针对 Marching Cubes 输出引入后处理（平滑、孔洞填补）改善网格质量。

