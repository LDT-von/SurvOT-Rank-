# 50_otehv2_timelocal_competing (V50)

**Time-Localized Optimal Transport Event Decomposition** —
时间局部竞争事件危险率分解，用于多模态癌症生存分析。

这是一个**独立方法文件夹**：单独训练、单独出结果，不与 V45 (45_otehv2_rankevent)
混。方法代码自包含，共享基础设施（数据集加载 / loss / 底层网络层）复用兄弟目录
`SurvOT-Rank-` 基座。

## 与 V45 的区别（本方法的两个结构性创新）

V45 是 `OT plan -> event tokens -> gated hazard sum`。本方法把后半段升级为
`OT plan -> time-local risk/protective events -> hazard decomposition`：

- **创新 A：时间局部事件头**
  每个事件在每个离散生存时间 bin 上有一个可学习责任权重，每个 bin 上事件互相竞争
  （softmax over events），自发分化出"早期 / 中期 / 晚期风险事件"。配两个正则：
  - 时间特化：单事件的时间责任分布应尖锐（低熵）
  - 时间覆盖：所有事件的总责任应铺满时间轴（高熵）

- **创新 B：竞争性风险/保护门控**
  事件拆成"风险增强"和"风险保护"两条独立通路（各有 hazard 头 + 时间门控），
  `hazard = 风险贡献 − beta · 保护贡献`（beta 可学习、softplus 非负），
  能表达保护性因素，可解释性强；配小 L2 稳定项防保护通路发散。

预测主路仍不使用 SlotDecoder / CrossAttention / SelfAttention / D×3，保持 V45
"脱离 SlotSPE 主预测链"的性质。

## 文件说明

```
50_otehv2_timelocal_competing/
├── backbone.py     # OT 事件危险率骨架（复制自 V45，本方法基座）
├── model_v45.py    # V45 OTEHV2RankEvent 父类（复制自 V45，本方法父类）
├── model.py         # 本方法主模型 OTEHTimeLocalCompeting
├── paths.py          # 解析兄弟目录 SurvOT-Rank- 基座（models/utils/dataset/dataset_csv）
├── args.py            # 命令行参数（含 V50 专属超参数）
├── train.py            # 训练入口（5-fold）
├── smoke_test.py        # 合成数据冒烟测试，不需真实数据/GPU
└── README.md
```

## 依赖基座

本文件夹需要与 `SurvOT-Rank-` 放在同一父目录下（默认布局）：

```text
SurvOT-Rank/
├── 50_otehv2_timelocal_competing/   # 本方法
└── SurvOT-Rank-/                     # 基座（内含 models/utils/dataset/dataset_csv）
```

如放在别处，用环境变量指定：`set SURVOT_BASE=C:\path\to\SurvOT-Rank-`

## 快速开始

```bash
# 1. 冒烟测试（不需真实数据/GPU）
python smoke_test.py
# 预期: PASS timelocal_competing (standalone): logits=(2, 4) aux_loss=... eval_ok=True

# 2. 正式训练（5-fold BLCA）
python train.py ^
  --data_root_dir C:\path\to\CPathPatchFeature\blca\uni ^
  --data_path ..\SurvOT-Rank-\dataset_csv ^
  --results_dir .\results_v50 ^
  --study blca --rna_format Pathways --signature combine ^
  --label_col survival_months_dss --bag_loss nll_surv ^
  --n_classes 4 --num_patches 2048 --encoding_dim 1024 ^
  --max_epochs 30 --batch_size 4 --seed 3 --gpu 0
```

## V50 专属超参数（args.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lambda_timelocal_spec` | 0.01 | 时间特化正则（事件时间责任分布应尖锐） |
| `--lambda_timelocal_cover` | 0.01 | 时间覆盖正则（事件总责任应铺满时间轴） |
| `--lambda_compete_reg` | 0.001 | 竞争稳定正则（约束保护通路幅度） |
| `--compete_beta_init` | -2.0 | 保护通路竞争强度 beta 的初始 logit（softplus 前） |

骨架超参数（`otehv2_*` / `rankevent_*`）继承自 V45，含义不变。
