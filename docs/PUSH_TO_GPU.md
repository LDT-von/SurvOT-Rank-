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

## 跑完之后发我

请把以下文件 push 回来：

1. `docs/scores/act_surv_v5_1_blca_5fold.tsv`
2. `docs/scores/act_surv_v5_2_blca_5fold.tsv`
3. `results/act_surv_v5/proofs/act_surv_v5_proofs_<timestamp>.json`
4. `results/act_surv_v5/proofs/act_surv_v5_per_patch_blca_fold0_<timestamp>.json`
5. `results/act_surv_v5/proofs/figures_4_6_per_patch/per_archetype_top16_real_<timestamp>.png`

---

## 我现在帮你做的（不等 GPU）

- [ ] 把 v5.1/v5.2 的 5 折结果写进 paper Section 4.3（数字一回来就写）
- [ ] 把 C 的 N=5000 数字写进 §4.5（数字一回来就改）
- [ ] 把 F 的图嵌入 §4.6（PNG 一回来就嵌）
- [ ] 把台账 (`paper_drafts/ArcSurv/ArcSurv_主张与证据台账.md`) 的 §4 已核验段更新