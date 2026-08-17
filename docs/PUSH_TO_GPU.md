# ACT v5 补做实验 — GPU 端执行手册

> 这边已经写完 3 个改动，全部 `py_compile` 通过，可直接 push 到 GPU 跑。

## 改动清单

| 文件 | 改动 | 作用 |
|---|---|---|
| `scripts/verify_act_surv_v5_all.py` | `experiment_C` 加 N=2000/5000 | 把 §4.5 speed-up 从 73.9× 拉到 ≥100× |
| `scripts/verify_act_surv_v5_all.py` | 新增 `experiment_F_per_patch_retrieval` | §4.6 从 numerical 升级到 visualization |
| `scripts/visualize_act_surv_v5_archetypes.py` | **新建** | 真实 BLCA checkpoint 跑 F，输出 PNG + JSON |

---

## 那边要执行（顺序）

### Step 1: pull 最新代码
```bash
cd ~/SurvOT-Rank   # 你的工作目录
git pull origin main
```

### Step 2: 主任务 — v5.1 BLCA 5 折（后台）

```bash
# 在后台跑，记录 log
mkdir -p logs
nohup python scripts/run_act_surv_v5.py \
    --config configs/act_surv_v5_1_blca.yaml \
    --folds 0 1 2 3 4 \
    > logs/v5_1_blca_5fold.log 2>&1 &
echo "v5.1 PID: $!"

# 等 v5.1 跑完，再跑 v5.2
# （同 GPU 同时跑两版本会 OOM）
nohup python scripts/run_act_surv_v5.py \
    --config configs/act_surv_v5_2_blca.yaml \
    --folds 0 1 2 3 4 \
    > logs/v5_2_blca_5fold.log 2>&1 &
echo "v5.2 PID: $!"
```

**期望**: 每折约 4-6 小时，5 折 ≈ 1 GPU-day

**判定标准**（看 `docs/scores/act_surv_v5_1_blca_5fold.tsv`）：
- 5 折 mean ≥ 0.69 → §4.3 ablation 表写满
- 0.68-0.69 → 次优 baseline
- < 0.68 → v5 仍是 baseline，v5.1/v5.2 不进 paper

### Step 3: 补做实验 C（30 min，在 v5.1 跑的同时跑）

```bash
# C: N=5000 benchmark，不占 BLCA 训练太多（共用 GPU）
nohup python scripts/verify_act_surv_v5_all.py \
    --experiments C --device cuda \
    > logs/v5_proof_C_N5000.log 2>&1 &
```

**输出**: `results/act_surv_v5/proofs/act_surv_v5_proofs_<timestamp>.json` + `.md`

**判定**:
- N=5000 处 speed-up ≥ 50× → §4.5 写 "≥50× at N=5000"
- 还是 < 50× → §4.5 改写为 "20-50× at N=1000-5000"（不硬撑 100×）

### Step 4: 补做实验 F（1-2 hour，真实 BLCA checkpoint）

```bash
# F: per-patch retrieval（用真实 BLCA fold0 checkpoint）
nohup python scripts/visualize_act_surv_v5_archetypes.py \
    --cancer blca --fold 0 --top-k 16 --device cuda \
    > logs/v5_proof_F_blca_fold0.log 2>&1 &
```

**输出**:
- `results/act_surv_v5/proofs/act_surv_v5_per_patch_blca_fold0_<timestamp>.json`
- `results/act_surv_v5/proofs/figures_4_6_per_patch/per_archetype_top16_real_<timestamp>.png`

**判定**:
- ✅ top1_share < 0.5, L2 > 0.5 → §4.6 写"archetype patch retrievable"
- ❌ 不通过 → §4.6 诚实写"archetypes not visually distinct at patch level"

### Step 5: 跑全量 proof（包括 A/B/D/E）当 sanity check

```bash
# 完整 6 个实验（~2 小时）
nohup python scripts/verify_act_surv_v5_all.py \
    --experiments A,B,C,D,E,F --device cuda \
    > logs/v5_proof_all.log 2>&1 &
```

---

### Step 6（新）: 补做实验 A2 — ACT vs MLP head C-index on real v5.1 checkpoint

**为什么**：A 是 synthetic self-consistency test，ρ=-0.03/0.32 不是真问题——是 design 如此（ACT 用 archetype 结构，MLP 用自由参数）。真正的"ACT 几乎等于 MLP" claim 要在真实 v5.1 checkpoint + 真实 val set 上测 C-index。

```bash
# A2: 真实 v5.1 BLCA fold-0 checkpoint + 真实 val loader
nohup python scripts/verify_act_surv_v5_all.py \
    --experiments A2 --device cuda \
    --a2-ckpt-path results/act_surv_v5_1/blca/fold0/model_best_s0.pth \
    --a2-data-root /data1/TCGA-UNI2-h-features \
    --a2-cancer blca --a2-fold 0 \
    > logs/v5_proof_A2_blca_fold0.log 2>&1 &
```

**输出**: `results/act_surv_v5/proofs/act_surv_v5_proofs_<timestamp>.json` 里多 `experiments.A2` 段，含 `c_index_act`, `c_index_mlp`, `delta_c_index`, `verdict`, `passed`。

**判定**:
- |ΔC| < 0.02 → ✅ §4.3 写 "ACT-head ≈ MLP-head" (free swap claim 成立)
- |ΔC| ≥ 0.02 → ⚠️ 诚实写 "ACT 表达不同 ranking，idea 的 claim 偏渐进"

**auto-derive**：
- 不传 `--a2-ckpt-path` 会自动找：
  - `results/act_surv_v5_1/blca/fold0/models/best_model.pt`
  - `results/act_surv_v5_1/blca/fold0/model_best_s0.pth`
  - `results/act_surv_v5_1/blca/**/sp_act_surv_v5_v5_1_blca_fold0/model_best_s0.pth`
- 不传 `--a2-data-root` 会用 env `DATA_ROOT` 或 `/data1/TCGA-UNI2-h-features`

**A2 跑完 5 折**：把 fold 1..4 也跑一遍：

```bash
for fold in 1 2 3 4; do
    python scripts/verify_act_surv_v5_all.py \
        --experiments A2 --device cuda \
        --a2-ckpt-path results/act_surv_v5_1/blca/fold${fold}/model_best_s${fold}.pth \
        --a2-cancer blca --a2-fold ${fold} \
        --output-dir results/act_surv_v5/proofs_a2 \
        > logs/v5_proof_A2_blca_fold${fold}.log 2>&1
done
# 5 折合并：报告里写 |ΔC| mean ± std
```

---

### Step 7（新）: 1D Ablation — v5.3（仅去 ranking）→ v5.4（仅 KL×5）

**背景**：v5.1 同时改了两件事（去 IPCW ranking + KL×5），从 0.6727→0.6954。1D ablation 拆分贡献。

**实验设计**：

| yaml | 唯一改动 | 其余 |
|---|---|---|
| `act_surv_v5_3_blca.yaml` | `lambda_rank: 0.10→0.00` | KL balance=0.01（不变）|
| `act_surv_v5_4_blca.yaml` | `lambda_balance: 0.01→0.05` | ranking=0.10（不变）|

**判定矩阵**（跑完看数字）：

| v5.3 C | v5.4 C | 结论 |
|---|---|---|
| > v5 | > v5 | 两个改动都有效，最优配方 = v5.3 + v5.4 |
| > v5 | ≤ v5 | KL×5 是唯一有效改动 |
| ≤ v5 | > v5 | 去 ranking 是唯一有效改动 |
| ≤ v5 | ≤ v5 | v5 baseline 仍是 SOTA，v5.1 提分靠随机 |

**用 sequential wrapper 跑**：

```bash
# v5.3（~5 hour）+ v5.4（~5 hour）顺序执行，不并发
bash scripts/run_v5_1d_sequential.sh
# 日志在 logs/v5_3_blca_5fold.log 和 logs/v5_4_blca_5fold.log
```

**或者单独跑**（不用 wrapper）：

```bash
# v5.3: 仅去 IPCW ranking
nohup python scripts/run_act_surv_v5.py \
    --cancers blca --variant v5_3 --folds 0 1 2 3 4 \
    > logs/v5_3_blca_5fold.log 2>&1 &

# 等 v5.3 跑完，再跑 v5.4: 仅 KL×5
nohup python scripts/run_act_surv_v5.py \
    --cancers blca --variant v5_4 --folds 0 1 2 3 4 \
    > logs/v5_4_blca_5fold.log 2>&1 &
```

**看结果**：

```bash
# 跑完后，scores 写进：
# results/act_surv_v5_3/blca/5fold_results_final.tsv
# results/act_surv_v5_4/blca/5fold_results_final.tsv

# 快速提取 mean C-index
python -c "
import glob, re
for v in ['v5_3', 'v5_4']:
    files = glob.glob(f'results/act_surv_v5_3/blca/**/5fold_results_final.tsv')
    if not files:
        files = glob.glob(f'results/act_surv_v5_4/blca/**/5fold_results_final.tsv')
    for f in files:
        print(f'{v}: {open(f).read()[:200]}')
"
```

**预期时间**：v5.3 + v5.4 各 ~5 小时（顺序跑），共 ~10 小时。

---

## 跑完之后发我

请把以下文件 push 回来：

1. `docs/scores/act_surv_v5_1_blca_5fold.tsv`
2. `docs/scores/act_surv_v5_2_blca_5fold.tsv`
3. `results/act_surv_v5/proofs/act_surv_v5_proofs_*.json`（A2 结果）
4. `results/act_surv_v5_3/blca/**/5fold_results_final.tsv`（v5.3 1D ablation）
5. `results/act_surv_v5_4/blca/**/5fold_results_final.tsv`（v5.4 1D ablation）
6. `results/act_surv_v5/proofs/act_surv_v5_per_patch_blca_fold0_*.json`
7. `results/act_surv_v5/proofs/figures_4_6_per_patch/per_archetype_top16_real_*.png`

---

## 我现在帮你做的（不等 GPU）

- [ ] 把 v5.1/v5.2 的 5 折结果写进 paper Section 4.3（数字一回来就写）
- [ ] 把 C 的 N=5000 数字写进 §4.5（数字一回来就改）
- [ ] 把 F 的图嵌入 §4.6（PNG 一回来就嵌）
- [ ] 把台账 (`paper_drafts/ArcSurv/ArcSurv_主张与证据台账.md`) 的 §4 已核验段更新
- [ ] 把 A2 的 |ΔC| 写进 §4.3（替代/补充 A 的 synthetic verdict）
- [ ] 把 v5.3/v5.4 的 1D ablation 数字写进 §4.3（定位 ranking vs KL×5 各自贡献）