# SurvOT-Rank

Optimal-transport-induced event hazard decomposition for multimodal cancer
survival analysis, with pairwise ranking supervision and gated event
aggregation. C-index 0.7105 on TCGA-BLCA (5-fold), +0.91% over baseline.

独立、自包含的生存分析模型仓库。核心思路：用最优传输（OT）计划把 WSI 与
Omics 的 slot 表征直接聚合成一组"预后事件 token"，再对每个事件单独建模
hazard，最后门控聚合——不使用 SlotDecoder / CrossAttention / SelfAttention /
D×3 拼接这条基线主路。

在 TCGA-BLCA、5-fold、30 epoch 设置下，验证集 C-index = **0.7105 ± 0.0181**
（同数据设置下 baseline = 0.7014，v9 骨干 = 0.7078）。

## 本仓库包含什么

```
45_otehv2_rankevent/
├── backbone.py       # OT 事件危险率骨架（三代价矩阵 + log-Sinkhorn + 事件融合）
├── model.py          # V45 主模型：骨架 + per-event NLL + 排序损失 + 全局残差 + eps退火
├── paths.py          # 查找外部 SlotSPE/ 基座的路径解析
├── args.py           # 命令行参数（训练基础参数 + V45 自己的超参数）
├── train.py          # 训练入口（5-fold 循环 + 早停 + 安全落盘 + 每 epoch 监控输出）
├── run_v45_30ep.sh   # 30-epoch 训练启动脚本
├── ensemble_eval.py  # 多 seed 集成评估
├── collect_results.py# 汇总多次实验结果排名
├── analysis_plots.py # 结果可视化
├── dataset/
│   └── dataset_survival.py  # 自 SlotSPE 的数据加载（CC-BY-NC-4.0）
├── models/
│   ├── slot_attention.py    # 自 SlotSPE 的多头 slot attention
│   └── omics_encoder.py     # 自 SlotSPE 的 pathway 编码器
├── utils/
│   ├── core_utils.py        # 自 SlotSPE 的训练循环
│   ├── general_utils.py     # 自 SlotSPE 的实验工具
│   └── loss_func.py         # 自 SlotSPE 的生存损失（NLL/Cox/Sinkhorn）
├── V45_ANALYSIS.md  # 架构与实验分析笔记
├── requirements.txt
├── LICENSE          # CC-BY-NC-4.0
└── README.md
```

这一个文件夹本身不含其他实验方向的代码（不依赖 `common/` 或其他编号文件夹），
可以单独拷贝、单独建仓库。

## 唯一的外部依赖：SlotSPE 基座

本仓库复用 [SlotSPE](https://github.com/zylvemvet/SlotSPE)（ICLR 2026，
CC-BY-NC-4.0 许可）的数据集加载、loss 函数、底层网络层（slot attention /
omics encoder）。这些是所有实验共用的基础设施，不复制进本仓库，需要你单独
准备一份：

```bash
git clone https://github.com/zylvemvet/SlotSPE.git
```

克隆下来的 `SlotSPE/` 需要放在本仓库的**兄弟目录**下，比如：

```text
your_workspace/
├── 45_otehv2_rankevent/      # 本仓库
└── SlotSPE/                   # 官方基座仓库
```

如果不方便放兄弟目录，也可以用环境变量指定任意位置：

```bash
export SLOTSPE_DIR=/path/to/SlotSPE
```

`paths.py` 会按 `SLOTSPE_DIR` → 兄弟目录 → 本文件夹内部 的顺序自动查找，
找不到会报错并提示怎么设置。

> 引用许可提示：SlotSPE 采用 **CC-BY-NC-4.0**（非商业性使用），如果你要
> 公开发表基于本仓库+SlotSPE 基座的成果，请遵守该许可并按其 README 引用
> 原论文（Zhang et al., ICLR 2026）。

## 数据准备

真实训练需要三块数据，本仓库和 SlotSPE 基座都**不包含**任何实际数据：

### 1. 临床表 / 签名 / fold 划分（SlotSPE 官方仓库自带）
克隆 SlotSPE 后自动就有，位于 `SlotSPE/dataset_csv/{clinical,signatures,splits}/`。

### 2. RNA 表（需要单独下载）
官方托管在 Google Drive，下载后放进 `SlotSPE/dataset_csv/raw_rna_data_inter/`：
- https://drive.google.com/drive/folders/1RxCjSZYTWhJRnbYWAGySyvZk2RUKYb1t

预期文件例如 `blca_rna_inter.csv`、`brca_rna_inter.csv` 等。

### 3. WSI patch 特征（.pt 文件，需要自备）
这是私有的 TCGA 衍生特征，SlotSPE 官方仓库和本仓库都不提供，需要你：
- 从你现有的服务器路径（例如 `/data/CPathPatchFeature`）直接拷贝，或
- 参考 `SlotSPE/feature_extract/README.md` 用 CLAM 流程自己从原始 WSI 提特征

**这部分体积通常是几十 GB，不要塞进 git 仓库**，用 `--data_root_dir` 参数
指向本地/服务器上的实际路径即可。

数据准备完成后的目录大致是：

```text
SlotSPE/dataset_csv/
├── clinical/all/blca.csv
├── raw_rna_data_inter/blca_rna_inter.csv     # 从 Google Drive 下载
├── signatures/combine_signatures.csv
└── splits/5fold/blca/

/data/CPathPatchFeature/blca/uni/             # WSI .pt 特征，自备
├── TCGA-XX-XXXX-01Z-00-DX1.pt
└── ...
```

## 快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
# 按你的 CUDA 版本装 torch，例如：
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 训练前自检（确认路径解析、前向/反向链路正常）

```bash
python -c "from model import OTEHV2RankEvent; import torch; m=OTEHV2RankEvent(omic_dim=20,wsi_dim=1024); x=torch.randn(2,20); wsi=torch.randn(2,2048,1024); o=m(x,wsi); print('PASS logits=', o['hazards'].shape, 'loss=', o['loss'].item())"
```

预期输出：
```
PASS logits= torch.Size([2, 4]) loss= ...
```

### 3. 正式训练

```bash
export DATA_ROOT=/data/CPathPatchFeature/blca/uni
export DATA_PATH=../SlotSPE/dataset_csv
export RESULT_DIR=./results_v45_standalone
bash run_v45_30ep.sh 0        # 0 = GPU id
```

或直接调用 `train.py`：

```bash
python train.py \
  --data_root_dir /data/CPathPatchFeature/blca/uni \
  --data_path ../SlotSPE/dataset_csv \
  --results_dir ./results \
  --study blca --rna_format Pathways --signature combine \
  --label_col survival_months_dss --bag_loss nll_surv \
  --n_classes 4 --num_patches 2048 --encoding_dim 1024 \
  --max_epochs 30 --batch_size 4 --seed 3 --gpu 0
```

训练结束后 `results/.../summary.csv` 会给出 5-fold 的 `val_cindex` 均值/标准差。

## 修改 / 二次开发

- 改损失权重、超参：改 `args.py` 里的默认值，或训练时用命令行覆盖
- 改骨架结构（OT 融合方式、事件编码器层数等）：改 `backbone.py`
- 改 V45 自己的监督逻辑（排序损失、门控熵等）：改 `model.py`
- 想做单变量对比实验：参考 `experiment_template.py` 的写法新建一个继承类

修改后先跑上面"训练前自检"那一行确认前向/反向没坏，再上真实数据训练。

## 已知局限

- 本仓库只验证过 TCGA-BLCA；换 study 需要确认 SlotSPE 基座里有对应的
  clinical/RNA/split 数据
- 不含 `common/model_factory.py` 的多方法切换能力，这个仓库只跑 V45 一个模型

## 复现结果（最近一次实测）

设置：TCGA-BLCA, 5-fold, 30 epoch, seed=3, batch_size=4, lr=5e-4,
signatures=combine, rna_format=Pathways。
启动：`run_v45_5fold_30ep.bat`（Windows）或等价 `python train.py ...` 命令
（与 `run_v45_5fold_30ep.bat` 中完全相同的 flags）。

| Fold | best val_cindex | @ epoch |
|------|-----------------|---------|
| 0 | 0.6997 | 3 |
| 1 | 0.7107 | 3 |
| 2 | 0.7102 | 11 |
| 3 | 0.6574 | 20 |
| 4 | 0.6868 | 5 |
| **mean ± std** | **0.6929 ± 0.0198** | — |

附属指标均值：c-index_ipcw=0.6341, IBS=0.2519, iauc=0.6400。
完整训练日志与每 epoch 曲线见
`results_v45_5fold_30ep/blca/SlotSPE_otehv2_rankevent/.../summary.csv` 与
`epoch_curve_fold{0..4}.csv`。`process_monitor_fold{0..4}.csv` 是后台
`utils/monitor.py` 写入的 CPU/RAM/GPU/磁盘/网络采样，可用于复盘资源曲线。

> 训练时长：约 8h35min（在 RTX 3090, 12 vCPU, 24 GB RAM 上）。

## 监控 / 性能打点

`utils/monitor.py` 提供一个**零阻塞**的后台采样器，跑训练时只需在 `train.py`
里 4 行引用（构造 → start → 每 batch `set_meta()` → stop）即可在
`results_dir/process_monitor_fold{fold}.csv` 拿到一份 CPU/RAM/GPU/磁盘/网络/
线程的时间序列，便于复盘 OOM、显存利用率低、数据加载瓶颈等。可独立
`python utils/monitor.py` 跑 5 秒冒烟测试。
