# CGMacros 数据预处理

## 目标

将 CGMacros 原始目录整理成 CGM Studio 可直接读取的格式：

```text
CGMacros_studio_ready/
├── raw/
│   └── train/
│       ├── CGMacros-001.csv
│       ├── CGMacros-002.csv
│       └── ...
├── labels/
│   └── labels.csv
└── metadata/
    ├── processing_manifest.csv
    └── preprocessing_config.json
```

## 处理内容

- 递归查找每个 `CGMacros-XXX` 文件夹中的 CSV。
- `Libre GL` 和 `Dexcom GL` 均由 mg/dL 转为 mmol/L：`mmol/L = mg/dL / 18.0156`。
- 生成统一的 `CGM` 列，默认仅使用 `Dexcom GL`。
- 从 `bio.csv` 的 `A1c PDL (Lab)` 匹配 HbA1c。
- 标签定义：
  - Healthy：HbA1c < 5.7
  - Prediabetes：5.7 ≤ HbA1c ≤ 6.4
  - T2D：HbA1c > 6.4
- 删除无效时间戳和重复时间戳，但不进行插值、平滑和峰谷检测；这些步骤由界面完成。

## 使用方法

```bash
python prepare_cgmacros_for_studio.py \
  --input-root "/你的路径/CGMacros" \
  --bio-csv "/你的路径/bio.csv" \
  --output-root "/你的路径/CGMacros_studio_ready" \
  --cgm-source dexcom
```

`--cgm-source` 可选：

- `dexcom`：统一 `CGM` 列只使用 Dexcom，推荐用于当前实验。
- `libre`：统一 `CGM` 列只使用 Libre。
- `dexcom_then_libre`：Dexcom 缺失时使用 Libre 补充。该模式会混合传感器，需谨慎。

## 界面设置

将 `raw/train` 中的 CSV 上传为训练数据，将 `labels/labels.csv` 上传为标签表。界面中的血糖列设置为：

```text
CGM
```

因为已经转换为 mmol/L，峰突出度可以从约 `0.45 mmol/L` 开始尝试。
