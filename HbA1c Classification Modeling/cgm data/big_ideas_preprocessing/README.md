# BIG IDEAS 数据预处理

已将本次提供的 `BIG IDEAS 2.zip` 整理成与 `hall2018_preprocessing.zip` 一致的 Studio 数据结构。原始 ZIP 保持不变。此包是格式整理后的 CGM 数据，不包含模型训练或预先选出的峰谷片段。

## 本次数据

| 项目 | 数量 |
|---|---:|
| 患者 | 16 |
| CGM 测量记录（EGV） | 36,898 |
| Healthy（HbA1c <5.7） | 8 |
| Prediabetes（5.7≤HbA1c<6.5） | 8 |
| T2D（HbA1c≥6.5） | 0 |
| 有 HbA1c 的患者 | 16 |
| 无效时间 / 非数值 CGM / 重复时间戳 | 0 / 0 / 0 |
| 排除的非 CGM 测量记录 | 182 |
| 超过25分钟的时间缺口 | 20 |

文件中的 HbA1c 范围为5.3–6.4，沿用本项目的百分数约定。原始 `HbA1c` 列未另附单位或采血日期；未推断其与 CGM 同日采集。这些类别表示所选 HbA1c 规则下的分组，不是原数据提供的综合临床诊断。

**此数据单独没有 T2D，不能单独运行当前网页的完整三分类训练。** 可作为包含 T2D 的公开训练队列的补充；合并时仍按真实患者划分训练与验证。预处理未复制患者或补造 T2D 标签。

## 目录

```text
big_ideas_preprocessing/
├── README.md
├── prepare_big_ideas_for_studio.py
├── requirements.txt
└── BIG_IDEAS_studio_ready/
    ├── raw/train/
    │   ├── BIGIDEAS_001.csv
    │   ├── ...
    │   └── BIGIDEAS_016.csv
    ├── labels/
    │   └── BIG_IDEAS_labels_hba1c.csv
    └── metadata/
        ├── clinical_metadata.csv
        ├── processing_manifest.csv
        ├── preprocessing_config.json
        ├── data_dictionary.json
        ├── field_mapping.csv
        ├── source_file_manifest.csv
        ├── validation_report.json
        ├── excluded_event_counts.csv
        └── gap_audit.csv
```

`raw/train` 延续 Hall 包的目录名称，仅表示存放位置；16人尚未划为训练或验证集。没有另建一套冲突的诊断标签文件；逐点文件中的 `label` 与独立 HbA1c 标签表一致。

## 在当前网页版导入

### 与 Hall 一样分别上传

在“1 · 数据管理”的公开数据区域：

1. “上传 CGM CSV / ZIP”选择 `raw/train/` 的16份 CSV。
2. “标签 / HbA1c CSV”选择 `labels/BIG_IDEAS_labels_hba1c.csv`。
3. 公开血糖单位选择 **mmol/L**，公开 HbA1c 单位选择 **%**。
4. 自定义列名留空，不勾选“CGMacros 数字编号对应”。点击“导入公开数据”。

也可以直接在公开 CGM 上传框选择整个 `big_ideas_preprocessing.zip`，由当前 v1.2.0 读取其中的 CGM 与标签。该包不是私有 sensor 预测包，不要上传到私有数据区域。选择已处理的 `CGM` 列时不要再次执行 mg/dL 换算。

导入后应显示16人、8 Healthy、8 Prediabetes、0 T2D。可在片段页检查谷→峰和谷→峰→谷提取；独立训练三分类时会因缺少 T2D 而停止，这是正确的类别完整性检查。

### 与 Hall 合并时

- 同时上传两套患者波形。
- 标签框同时上传 `Hall2018_labels_hba1c.csv` 和 `BIG_IDEAS_labels_hba1c.csv`，明确使用同一 HbA1c 标签规则。不要改用 Hall 的旧 `labels_h.csv` 综合诊断标签。
- BIG IDEAS 编号为 `BIGIDEAS_001` 等，Hall 保留原完整编号；不要按文件顺序重新编号。
- 所有事件继续继承患者编号与患者标签，同患者事件不得跨训练/验证集。

## 临床 metadata

原始 `Demographics.csv` 只有 `ID,Gender,HbA1c`。处理后的临床表保留性别 `FEMALE/MALE` 和 HbA1c，另保留与 CGM 一致的 `subject_id`。HbA1c 是预测目标，不作为临床输入。

**没有可核实的年龄、BMI、身高或体重。** 对应列保持空值，未填0、未用其他患者数据替代。Dexcom 设备资料中15人的生日为1970-01-01，另1人为1990-01-01；这些生日未经核实，仅放在 `source_dob_text_unverified` 供核对，不据此计算年龄。

当前 v1.2.0 融合模型使用年龄、BMI等数值变量，尚不支持性别。因此，本包本身适用于 CGM-only；若要让这批患者参加有真实年龄/BMI的融合实验，需要补充临床资料。与 Hall 合并时，模型虽可按训练集规则填补缺值，但这16人的年龄/BMI仍然全部缺失，不能将填补值报告为已测量资料。

## 处理规则

1. 按 `Dexcom_001.csv` 的文件号 `001` 对应 `Demographics.ID=1`，并验证一一对应。设备 `PatientIdentifier` 只在4份文件出现，不作为连接键。输出加 `BIGIDEAS_` 前缀，避免与其他库的 `001` 等编号冲突。
2. 仅保留 `Event Type=EGV` 作为 CGM 观测。排除182条非测量记录：112条 Alert、各16条 FirstName/LastName/DateOfBirth/Device、4条 PatientIdentifier、2条 Exercise。报警阈值即使填有血糖数值，也不是 CGM 测量。
3. 使用原始时间字符串，输出 `Timestamp`，保留秒级抖动及实际缺口。原始文件没有时区说明，不进行时区转换。正常采样约5分钟，最长缺口为1,314.9833分钟，约21.916小时。
4. 原血糖列明确为 mg/dL。按 Hall 参考包相同系数转换：`CGM = glucose_original_mgdl / 18.0156`，`CGM_unit=mmol/L`。原始数值、原始字符串、来源文件与 CSV 行号保留在每条输出中。
5. 不平滑、不重采样、不插值、不补长缺口、不截取餐后窗口、不预先检测峰谷。片段选取交由网页的统一规则执行。没有公开的进餐时间用于餐后窗口。
6. 若出现无法解析的时间，程序排除并另存审计；非数值 CGM 置空并保留 `sensor_code`；完全相同的患者时间戳按 Hall 约定取数值平均并保留原始行审计。本批没有这些情况。
7. 数值40 mg/dL原样保留，不因接近设备范围而擅自改成缺失或推断为截断值。未对血糖值做裁剪。

## 字段与审计

每位患者 CSV 的主要列：

```text
subject_id,Timestamp,CGM,CGM_unit,label,hba1c,dataset,
glucose_original_mgdl,sensor_code,source_dataset,source_file,source_row,
event_type,source_device_id,source_timestamp,raw_glucose_value
```

`processing_manifest.csv` 按患者记录输入、输出、排除数量与时间覆盖；`excluded_event_counts.csv` 记录非 EGV 排除原因；`gap_audit.csv` 记录超过25分钟的观测缺口，未据此删除数据。其余缺口仍保存在原始时间轴上。

`source_file_manifest.csv` 保存所有17个源 CSV 的 SHA-256；`preprocessing_config.json` 保存源 ZIP 的 SHA-256、处理参数和总数，便于核对来源。`data_dictionary.json` 和 `field_mapping.csv` 说明列的含义及来源。

`validation_report.json` 记录逐行原始值核对、可重复生成和网页导入兼容性检查。当前网页默认质量规则下，可提取895个谷→峰片段、961个谷→峰→谷片段，均覆盖16名患者。这些仅是处理与兼容性检查，没有训练或选择模型。

## 重新生成

在此文件夹中安装依赖，再选择一个空输出目录：

```bash
python -m pip install -r requirements.txt
python prepare_big_ideas_for_studio.py \
  --input-zip "/path/to/BIG IDEAS 2.zip" \
  --output-root "./BIG_IDEAS_studio_ready_new"
```

程序只读取源包，不覆盖已有非空输出目录，也不执行源包中的内容。交付数据已经生成，无须重新运行脚本即可上传。
