#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
V45 otehv2_rankevent  涓€閿泦鎴愭帹鐞?(seed 3 + seed 5, 5-fold CV)
=========================================================================
鍔熻兘: 鍔犺浇棰勫厛璁粌濂界殑 10 涓?.pth (2 seeds 脳 5 folds), 璺?val 鎺ㄧ悊,
      鐒跺悗鎸?fold 瀵归綈 logits/risk, 闆嗘垚鍑?2-seed c-index.

鐩存帴杩愯:
    python important_outputs/v45_ensemble_bundle/quick_infer_ensemble.py

杈撳嚭:
    per-fold / per-seed c-index
    闆嗘垚 (logits 骞冲潎)  c-index
    闆嗘垚 (risk 骞冲潎)    c-index

鐩爣: 澶嶇幇涔嬪墠 2-seed 闆嗘垚 c-index 鈮?0.7237 (瓒呰繃鍗?seed 0.7105/0.7158)

涓嶉渶瑕佸啀璁粌, 鐩存帴璺戝嚭缁撴灉. 鎬昏€楁椂 ~5-10 鍒嗛挓 (CPU).
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
from types import SimpleNamespace
from collections import defaultdict
warnings.filterwarnings("ignore")

# 璺緞
BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BUNDLE_DIR))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "survot_rank", "training")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, PROJECT_ROOT)

from survot_rank.training.paths import SLOTSPE_DIR, ensure_slotspe_in_path
ensure_slotspe_in_path()

from dataset.dataset_survival import SurvivalDatasetFactory, _collate_pathways
from utils.core_utils import _process_data_and_forward, _calculate_risk, _update_arrays
from survot_rank.training.train_runner import get_split
from survot_rank.training.model_factory import get_model
from sksurv.metrics import concordance_index_censored


# ============================
# 璁粌鏃剁殑鍏ㄩ儴 args (浠?run_v45_multiseed_30ep.sh 澶嶅埢)
# ============================
def build_args(seed: int = 3):
    """鏋勯€?v45 璁粌鏃剁殑鍏ㄩ儴 args. seed 浠呯敤浜?DataLoader (鎺ㄧ悊鏃跺奖鍝嶄笉澶?."""
    args = SimpleNamespace(
        survot_method="otehv2_rankevent",
        # 鏁版嵁
        study="blca",
        data_path="./survot_rank/research/legacy/slotspe_runtime/dataset_csv",
        data_root_dir="./survot_rank/research/legacy/slotspe_runtime/dataset_csv",
        label_col="survival_months_dss",
        rna_format="Pathways",
        signature="combine",
        n_classes=4,
        num_genes=None,
        num_patches=2048,
        survival_bins=4,
        # 妯″瀷
        encoding_dim=1024,
        wsi_projection_dim=256,
        omics_dim=256,
        fusion_dim=256,
        dropout=0.25,
        size_arg="small",
        model_type="mlp_per_path",
        gated=True,
        sig_net_weight=8.0,
        sig_net_gate=8.0,
        # 璁粌 (鎺ㄧ悊鐢ㄤ笉涓? 浣?init 鏃跺彲鑳借)
        bag_loss="nll_surv",
        nll_weight=1.0,
        rank_weight=1.0,
        ple_weight=0.0,
        sinkhorn_weight=0.0,
        alpha_index="1.0",
        max_epochs=30,
        lr=0.0005,
        batch_size=4,
        reg_type="None",
        lambda_reg=0.0,
        weighted_sample=False,
        max_cindex=-1.0,
        max_cindex_epoch=0,
        # otehv2 涓撳睘
        otehv2_eps=0.05,
        otehv2_iter=50,
        otehv2_warmup=5,
        otehv2_num_events=24,
        otehv2_heads=4,
        otehv2_layers=4,
        otehv2_dropout=0.1,
        lambda_otehv2_ot=0.06,
        lambda_otehv2_div=0.01,
        lambda_otehv2_event_surv=0.25,
        lambda_otehv2_recon=0.2,
        # rankevent 涓撳睘
        lambda_rankevent_per_event=0.15,
        lambda_rankevent_rank=0.15,
        lambda_rankevent_global_cons=0.02,
        lambda_rankevent_gate_ent=0.005,
        rankevent_eps_start=0.10,
        rankevent_eps_end=0.05,
        rankevent_eps_anneal_epochs=12,
        rankevent_global_init=-2.0,
        # slot
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=5,
        topk_ratio=0.25,
        top_k_method="parallel_topk_st",
        # data loader
        num_workers=0,
        # inference 鏃跺～
        cur_epoch=29,
        omic_missing=False,
        seed=seed,
    )
    return args


# ============================
# 鍗?fold 鎺ㄧ悊
# ============================
def infer_one_fold(fold: int, seed: int):
    """鍔犺浇 seed{N}_fold{F}.pth, 璺?val, 杩斿洖 c_index, risk, logits, case_ids."""
    pth_path = os.path.join(BUNDLE_DIR, f"seed{seed}_fold{fold}.pth")
    assert os.path.exists(pth_path), f"缂哄皯鏉冮噸: {pth_path}"

    args = build_args(seed=seed)

    # 1) dataset_factory (鎸?fold 鍒掑垎)
    factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches,
    )
    args.omic_sizes = factory.omic_sizes
    args.omic_names = factory.omic_names
    args.pathway_names = getattr(factory, 'pathway_names', None)

    # 2) dataloader
    train_data, val_data, train_loader, val_loader = get_split(args, factory, fold)

    # 3) 妯″瀷 + 鏉冮噸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.survot_method, args).to(device)
    state = torch.load(pth_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [WARN] seed{seed} fold{fold}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # 4) 鎺ㄧ悊
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_logits, all_case_ids = [], []
    case_ids = val_loader.dataset.label_df["case id"]
    count = 0
    with torch.no_grad():
        for data in val_loader:
            out, y_disc, event_time, c = _process_data_and_forward(args, model, data, device, test=True)
            logits, _ = out
            risk, _ = _calculate_risk(logits)
            all_risk_scores, all_censorships, all_event_times = _update_arrays(
                all_risk_scores, all_censorships, all_event_times,
                event_time, c, risk, data,
            )
            all_logits.append(logits.detach().cpu().numpy())
            all_case_ids.append(case_ids.values[count])
            count += 1

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-8
    )[0]

    # 閲婃斁鏄惧瓨
    del model, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "case_ids": list(all_case_ids),
        "risk": all_risk_scores,
        "logits": all_logits,
        "censorships": all_censorships,
        "event_times": all_event_times,
        "c_index": float(c_index),
        "n": len(all_case_ids),
    }


# ============================
# 涓绘祦绋?# ============================
def main():
    print("=" * 70)
    print("V45 otehv2_rankevent  涓€閿泦鎴愭帹鐞?(seed3 + seed5, 5-fold)")
    print("=" * 70)
    print(f"pth bundle: {BUNDLE_DIR}")
    print(f"device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    SEEDS = [3, 5]
    FOLDS = [0, 1, 2, 3, 4]

    # 妫€鏌?pth
    missing_files = []
    for seed in SEEDS:
        for fold in FOLDS:
            p = os.path.join(BUNDLE_DIR, f"seed{seed}_fold{fold}.pth")
            if not os.path.exists(p):
                missing_files.append(p)
    if missing_files:
        print("[ERR] 缂哄皯 pth:")
        for m in missing_files:
            print(f"  {m}")
        sys.exit(1)

    # 1) 鎺ㄧ悊 10 涓?fold
    print("[1/3] 鎺ㄧ悊 10 涓?fold (绾?5-10 鍒嗛挓) ...")
    t0 = time.time()
    per_seed_fold = {3: {}, 5: {}}
    for seed in SEEDS:
        for fold in FOLDS:
            print(f"\n  -> seed={seed} fold={fold}")
            res = infer_one_fold(fold, seed)
            per_seed_fold[seed][fold] = res
            print(f"     c-index={res['c_index']:.4f}  (n={res['n']})")
    print(f"\n  鎺ㄧ悊鎬昏€楁椂: {time.time() - t0:.1f}s")

    # 2) 鍗?seed 闆嗘垚 (5-fold 鍙栧钩鍧?
    print("\n[2/3] 鍗?seed 闆嗘垚 (5-fold mean c-index) ...")
    seed_mean = {}
    for seed in SEEDS:
        cidx = [per_seed_fold[seed][f]["c_index"] for f in FOLDS]
        seed_mean[seed] = float(np.mean(cidx))
        print(f"  seed={seed}  fold c-index: {[f'{x:.4f}' for x in cidx]}")
        print(f"           mean={seed_mean[seed]:.4f}")

    # 3) 2-seed 闆嗘垚
    print("\n[3/3] 2-seed 闆嗘垚 (鎸?fold 瀵归綈 logits/risk, 鍙栧钩鍧? ...")
    ens_logits_fold, ens_risk_fold = [], []
    for fold in FOLDS:
        r3 = per_seed_fold[3][fold]
        r5 = per_seed_fold[5][fold]
        assert r3["case_ids"] == r5["case_ids"], f"fold{fold} case order mismatch"
        # Average logits and risk across seeds.
        ens_logits = (r3["logits"] + r5["logits"]) / 2.0
        ens_risk = (r3["risk"] + r5["risk"]) / 2.0
        censor = r3["censorships"]
        event_time = r3["event_times"]
        c_log = concordance_index_censored(
            (1 - censor).astype(bool), event_time, ens_logits.mean(axis=-1), tied_tol=1e-8
        )[0]
        c_risk = concordance_index_censored(
            (1 - censor).astype(bool), event_time, ens_risk, tied_tol=1e-8
        )[0]
        ens_logits_fold.append(c_log)
        ens_risk_fold.append(c_risk)
        print(f"  fold={fold}  ensemble_logits_c={c_log:.4f}  ensemble_risk_c={c_risk:.4f}")

    ens_logits_mean = float(np.mean(ens_logits_fold))
    ens_risk_mean = float(np.mean(ens_risk_fold))

    print("\n" + "=" * 70)
    print("Result summary")
    print("=" * 70)
    print(f"  seed 3 5-fold mean:              {seed_mean[3]:.4f}")
    print(f"  seed 5 5-fold mean:              {seed_mean[5]:.4f}")
    print("-" * 70)
    print(f"  2-seed ensemble, logits mean:    {ens_logits_mean:.4f}   target ~0.7237")
    print(f"  2-seed ensemble, risk mean:      {ens_risk_mean:.4f}")
    print("=" * 70)

    print()
    print("Why the ensemble can improve:")
    print("  - Different seeds usually learn partially different decision surfaces.")
    print("  - Averaging logits reduces seed-specific variance before risk conversion.")
    print("  - The gain should still be reported as an ensemble result, not a single-run result.")


if __name__ == "__main__":
    main()
