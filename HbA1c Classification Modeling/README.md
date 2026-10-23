# HbA1c prediction models — CGM only

包含 **5 种模型代码、5 个随机种子 × 5 种模型共 25 个已训练 checkpoint、加载示例和配置说明**。

这批模型只使用 CGM 序列，输出 HbA1c 的三个类别概率，**不是连续 HbA1c 数值回归模型**。Metadata 和人工统计特征均未作为本批模型输入。


## 1. 文件结构

```text
hba1c_prediction_models/
├── README.md
├── requirements.txt
├── predict.py                       # 统一加载与事件/患者推理入口
├── verify_bundle.py                 # 校验并加载全部 25 个 checkpoint
├── model_manifest.json              # 每个模型的路径、种子、配置、原始来源与 SHA256
├── verified_environment.json         # 本次交接实际验证环境
├── verification_report.json          # 全模型加载/合成输入检查结果
├── SHA256SUMS                       # 交付文件完整性清单
├── configs/
│   ├── experiment.json              # 本批训练设置
│   └── segment_config.json          # 本批片段提取与质量控制设置
├── examples/
│   ├── example_events.json          # 纯合成 CGM 输入，非患者数据
│   └── load_model.py                # Python 调用示例
├── cgm_bag/
│   ├── models.py                    # CNN / TCN / BiGRU / Transformer
│   ├── knn.py                       # KNN 训练、保存、加载与推断
│   ├── training.py                  # 患者划分、标准化、训练与 checkpoint 加载
│   ├── metadata.py                  # 原加载器依赖；当前模型未启用此分支
│   ├── event_features.py            # 原加载器依赖；当前模型未启用此分支
│   ├── cohorts.py                   # 训练来源与划分审计
│   ├── ablation.py                  # 原审计模块的配套依赖
│   └── __init__.py
└── checkpoints/
    ├── knn/seed_42...seed_46/        # 每个目录：model.npz + train_config.json
    ├── 1d_cnn/seed_42...seed_46/     # 每个目录：model.pt + train_config.json
    ├── tcn/seed_42...seed_46/
    ├── bigru/seed_42...seed_46/
    └── transformer/seed_42...seed_46/
```

## 2. 快速加载和预测

建议使用 Python 3.12，在本目录创建独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 使用已训练的 Transformer seed 42，不会启动训练。
python predict.py --model transformer --seed 42 \
  --input examples/example_events.json --output-dir outputs/demo

# 验证全部 25 个模型。
python verify_bundle.py

# Python API 示例。
python examples/load_model.py
```

Windows 可用 `.venv\Scripts\activate` 激活环境。运行入口默认 CPU；模型目录根据脚本位置定位，移动整个文件夹后无需修改原机器路径。跨硬件或依赖版本的浮点结果可能有细微差异。

模型参数 `--model` 可取 `knn`、`1d_cnn`、`tcn`、`bigru`、`transformer`；`--seed` 可取 `42`、`43`、`44`、`45`、`46`。例如：

```bash
python predict.py --model knn --seed 46 \
  --input examples/example_events.json --output-dir outputs/knn_seed46
```

Python API：

```python
import json
from pathlib import Path
from predict import load_model, predict

events = json.loads(Path("examples/example_events.json").read_text())
loaded = load_model("transformer", seed=42)
tables = predict(loaded, events)
event_probabilities = tables["events"]
patient_probabilities = tables["patients"]
```

## 3. 输入：CGM 片段

输入 JSON 是事件列表。每个事件含 `subject_id`、`event_id`、`values`：

```text
[
  {
    "subject_id": "SYNTHETIC_A",
    "event_id": "event_1",
    "values": [64 个按时间顺序排列的 mmol/L 血糖值]
  }
]
```

上面仅展示结构；可运行的完整 JSON 见 `examples/example_events.json`。同一患者可有多个 event；同一患者内 `event_id` 必须唯一。推理不需要 HbA1c 标签、metadata 或人工特征。

**必须输入未标准化的绝对血糖值，单位 mmol/L。** 如果上游是 mg/dL，原项目换算方式是除以 `18.0182`。推理入口会应用对应 checkpoint 保存的标准化参数；不要提前做 z-score，也不要用私有数据重新拟合均值/标准差。

这 25 个模型使用同一个片段方案：**trough–peak–trough，长度 64**。先按真实时间选择完整片段并完成质量检查，再按时间线性插值：

```python
# t 为所选片段内有效观测的递增时间，单位秒；g 为对应 mmol/L 值。
target = np.linspace(t[0], t[-1], 64)
values = np.interp(target, t, g).astype(np.float32)
```

64 点覆盖每个片段自身的持续时间，不代表固定分钟采样间隔，也不是按原 CSV 行号插值。插值范围取首末有效观测，不向缺失边界外推。公开数据的中值平滑仅用于定位峰谷，模型输入从未平滑的 CGM 观测生成。

原提取设置：

| 项目 | 设置 |
|---|---|
| 片段类型 | trough–peak–trough |
| 重采样长度 | 64 |
| 有效原始采样点 | 至少 5 点 |
| 片段时长 | 20–360 分钟 |
| 相邻有效采样最大间隔 | 25 分钟 |
| 最大缺失比例 | 0.2 |
| CGM 范围 | 1–40 mmol/L |
| 最小振幅 | 0.4 mmol/L |
| 峰 prominence | 0.5 mmol/L |
| 峰谷定位平滑窗口 | 15 分钟 |
| 最小峰间距 | 30 分钟 |
| 左右谷搜索范围 | 180 分钟 |

私有 CGM 映射数据需采用 `predicted_cgm_mmol_L`；真实 CGM 的 `actual_cgm_mmol_L` 可作为单独的 oracle 对照输入。完整 TPT 片段的上升段、下降段各需至少 2 个有效点，峰需位于保留采样首末时间之间。只有 `trough1_to_peak` 的片段不能补成完整 TPT 后直接用于这些权重。

`predict.py` 校验输入格式、长度、唯一性和有限值；仅凭 64 点数组无法验证原始时间间隔或峰谷边界，因此原始片段筛选应由上游完成。

## 4. 模型结构与类别顺序

| 模型 | 本批 checkpoint 对应结构 |
|---|---|
| KNN | 标准化后的 64 点 CGM 向量 → Euclidean 距离 → 5 个近邻、距离加权投票 |
| 1D CNN | 3 层 Conv1d，kernel 5/5/3、通道 32，GroupNorm/GELU → 全局平均池化 → Linear(32,3) |
| TCN | 1×1 输入投影 → 4 个残差因果卷积块，dilation 1/2/4/8，每块 2 层 kernel 3 卷积 → 时间均值池化 → Linear(32,3) |
| BiGRU | 单层双向 GRU，每方向 hidden 16 → 所有时间步输出均值池化 → Dropout → Linear(32,3) |
| Transformer | 每个 CGM 时间点 Linear(1,32) 嵌入 + 正弦位置编码 → 2 层 Encoder、4 heads、FFN 64 → 时间均值池化 → Linear(32,3) |

BiGRU 使用时间均值池化，不是只取最后隐状态；Transformer 是逐点 token 嵌入，没有 patch 切片。KNN 是监督分类基线，不是聚类算法。

神经网络沿用 `PatientBagModel` 容器，但这些权重全部使用 `sample_unit="event"`、`K=1`、`pooling="mean"`，一条事件就是一个模型输入。底层张量为 `[batch, 1, 64]`，不要求每个患者凑齐多个片段。若直接调用底层网络，需自己使用保存的 scaler；推荐使用已经处理好的 `predict.py`。

输出类别顺序固定：

| 编号 | 名称 | 训练标签定义，HbA1c 单位 % |
|---|---|---|
| 0 | Healthy | < 5.7 |
| 1 | Prediabetes | ≥ 5.7 且 < 6.5 |
| 2 | T2D | ≥ 6.5 |

事件表和患者表包含 `p_Healthy`、`p_Prediabetes`、`p_T2D`、预测类别及置信度。事件表每个 event 一行；患者概率为该患者所有输入事件概率的算术均值，再取 argmax 得到类别。患者聚合不是多数票。

此前外部评估重点报告 Prediabetes 和 T2D；**这不改变模型仍为三分类的事实**。本包保留三类原始概率。若只显示 2×2 confusion matrix，仍需另外记录落入 Healthy 预测列的错误，不能删掉这些错误样本后重新计算准确率，也不应把剩余两类概率重新归一化并称作原模型结果。

## 5. 五次训练与权重来源

公开训练来源为 Hall2018、CGMacros、BIG IDEAS，共 116 名有标签患者、7,092 个合格片段。训练/验证按照患者划分，公开验证比例约 25%；同一患者的事件不会同时进入两组。

每个种子使用一次患者划分，同一种子下五种模型共享该划分。更换种子会改变患者划分，以及神经网络的初始化和训练样本顺序。因此这五轮属于重复随机划分实验，**不是五折交叉验证**；KNN 的差异来自训练划分，近邻推断本身没有随机初始化。

| 训练项 | 本批设置 |
|---|---|
| 随机种子 | 42、43、44、45、46 |
| 神经网络训练上限 | 200 epochs |
| 早停 | public validation loss，patience 10 |
| 优化器 | AdamW |
| 学习率 / weight decay | 0.0001 / 0.0001 |
| Batch size | 16 |
| Hidden dimension / dropout | 32 / 0.1 |
| 类别加权 | 启用，按公开训练事件数计算 |
| 标准化 | 仅公开训练患者拟合的 patient-balanced global z-score |
| Metadata / 手工特征 | 未使用 |
| 私有数据微调 | 本批 checkpoint 未进行 |

保存的是每次训练中公开验证损失最低时的网络参数，实际 `best_epoch`、训练轮数、训练/验证人数和事件数见 `model_manifest.json`；不要把 200 的训练上限当作所有模型实际执行的轮数。

神经网络 `.pt` 内保存权重、模型配置、标准化参数、公开训练/验证患者划分及训练历史。加载使用 `torch.load(..., weights_only=True)`，并严格匹配模型结构与患者划分审计。

KNN `.npz` 内保存公开训练 CGM 向量、类别标签、公开患者/event 标识、标准化参数及来源校验信息。**KNN 文件并非只有数值参数，推断本身需要这些训练实例。** 加载时会从已保存向量重建近邻分类器，不会选择新训练数据或重新开展五轮实验。`.npz` 使用 `allow_pickle=False` 读取。

本包不含私有测试数据。原 checkpoint 中的公开训练标识和 KNN 公开训练向量已按原格式保留，不能将本包描述成“完全不含数据的纯权重包”。

## 6. 验证、复用与范围

`verify_bundle.py` 检查 25 个 checkpoint 的数量、种子、配置、SHA256，加载每个模型，对合成 CGM 进行推断，检查三类概率有效、概率和为 1、患者聚合与事件概率一致。交接时的结果保存在 `verification_report.json`。

合成输入检查只用于验证代码和权重可运行，不是性能评估。本包没有重新计算 accuracy、F1 或 AUROC，也不据此宣称某种模型性能更好。

`requirements.txt` 与 `verified_environment.json` 记录此次交接验证使用的 Python 依赖版本；它们不是原实验运行时自动保存的环境锁文件。`model_manifest.json` 中的 `original_path` 仅记录相对来源，加载入口不依赖它。ZIP 解压后保留整个目录结构即可运行。

训练核心也已保留：神经网络可使用 `cgm_bag.training.train_model`，KNN 可使用 `cgm_bag.knn.train_knn`。如后续重新训练，应先完成原始数据提取，并加载本包对应的 `train_config.json`；不要直接使用旧训练类的默认 K/bag 设置。仅运行上述推理和验证命令不会训练新模型。
