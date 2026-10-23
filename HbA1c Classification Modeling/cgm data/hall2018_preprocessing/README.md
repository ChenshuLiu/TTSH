# Hall 2018 / Glucotypes 数据预处理

## 运行

```bash
python prepare_hall2018_for_studio.py \
  --cgm-file /path/to/pbio.2005143.s010 \
  --clinical-db /path/to/pbio.2005143.s014.db \
  --output-root ./Hall2018_studio_ready
```

## 界面设置

- 时间列：`Timestamp`
- CGM 列：`CGM`
- 患者编号：由文件名自动识别，或使用 `subject_id`
- 标签列：`label`
- 单位：mmol/L
- 进餐列：本数据没有公开进餐时间，不要启用餐后窗口模式

## 推荐训练模式

- 机器学习：全局峰谷片段、患者级聚合
- 深度学习：固定窗口、日级窗口、完整患者序列
- 不适用：餐后窗口模式

## 标签

使用官方 SQLite 数据库 `clinical.diagnosis`：
- non-diabetic → Healthy
- pre-diabetic → Prediabetes
- diabetic → T2D

A1C 作为可选字段保留，但不是分类训练所必需。
