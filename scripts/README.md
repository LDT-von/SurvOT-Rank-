# 运行脚本索引

> 状态快照：2026-08-17。这里的文件包含正式队列、筛选、监控、汇总和历史兼容入口，不应再统一称为“短包装脚本”。

## 首选入口

```bash
# 查看方法身份和别名
python -m survot_rank.cli methods

# 检查仓库、数据与方法目录
python -m survot_rank.cli doctor

# 直接用 YAML 训练
python -m survot_rank.cli train --config <config.yaml>
```

只有当实验需要多癌种队列、fold 锁、动态覆盖、断点续跑或专门诊断时，才使用本目录脚本。

## 脚本职责

| 前缀/入口 | 职责 | 例子 |
|---|---|---|
| `run_*_final_cross_cancer.py` | 冻结协议的正式跨癌种队列 | DCT v3.8.2、IST-Surv v4.0 |
| `run_priority_experiment_queue.py` | 历史优先队列与跨方法筛选 | 多方法顺序调度 |
| `run_dct_v3*.py` / `run_dct_v4*.py` | DCT 版本筛选、smoke、plan | v3.5–v4.1 |
| `run_v40_*.py` / `run_ist_v40_*.py` | IST-Surv v4.0 协议与修复 | plan、正式队列、repair gate |
| `monitor_*.py` / `watch_*.*` | 只读监控 | 训练进度、锁和结果状态 |
| `summarize_*.*` / `export_*.*` | 汇总和解释导出 | CSV 汇总、患者级解释 |
| `train_v45_*.*` / `run_ablation.*` | 旧基线兼容入口 | V45、早期消融 |

## 当前正式/工作入口

- DCT v3.10 DCT-Reg 正式队列：`run_dct_v310_final_cross_cancer.py`。
- DCT v3.10 配对消融与机制控制：`run_dct_v310_experiments.py`。
- DCT v3.8.2 fixed-full：`run_dct_v382_final_cross_cancer.py`（历史对照）。
- IST-Surv v4.0 冻结队列：`run_ist_v40_final_cross_cancer.py`。
- CA-PSA / CATET / ArcSurv 的统一协议：
  [THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md](../docs/roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md)。
- 当前未提交的 repair-gate 与 paper-ablation 脚本属于工作资产，在验证和提交前不改名、不移动。

## 为什么暂时不分子目录

部分脚本互相导入 `task_lock.py`、v3.8/v3.8.2 队列与公共调度逻辑，测试也直接
`import scripts.<module>`。直接移动会破坏已有训练命令、测试和服务器任务。因此本轮先建立索引；将来若物理分目录，必须保留原路径兼容包装并同步测试。

## 已知特殊项

- `generate_final_results.py` 引用的 `configs/v51_slimbridge.yaml` 不在本仓库；V51 来自外部 newSlotSPE 工作区。该脚本不能作为本仓库独立可运行入口。
- DCT v3.6/v3.8/v3.8.2 通过 launcher 在基础配置上动态覆盖方法和损失参数；复现时应保存最终命令与配置快照。
- 新脚本统一采用
  `run_<method>_<purpose>.py`、`monitor_<method>_<purpose>.py`、
  `summarize_<method>_<purpose>.py` 或 `export_<method>_<purpose>.py`。

方法角色见 [docs/METHODS.md](../docs/METHODS.md)，配置分工见
[configs/INDEX.md](../configs/INDEX.md)。
