#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
V45 2-seed 闆嗘垚绮剧‘澶嶇幇 0.7237
=========================================================================
鎸?fold 璁＄畻闆嗘垚 c-index, 鐒跺悗瀵?5 fold 鍙栧潎鍊?
鐩存帴璋冪敤鍘?ensemble_eval.py 鐨勯€昏緫, 浣嗗唴鑱斿疄鐜? 1 绉掑嚭缁撴灉.

鐢ㄦ硶:
    python exact_ensemble_07237.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED3_DIR = "/data1/sweep_results_30ep/otehv2_rankevent/blca/SurvOTRank_otehv2_rankevent/0.0005_b4_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_runall_otehv2_rankevent_30ep"
SEED5_DIR = "/data1/sweep_results_30ep/otehv2_rankevent_seed5/blca/SurvOTRank_otehv2_rankevent/0.0005_b4_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed5_rW_8_rG_8_sp_runall_otehv2_rankevent_seed5_30ep"

from sksurv.metrics import concordance_index_censored

print("=" * 70)
print("V45 otehv2_rankevent 2-seed 闆嗘垚 (浠?pkl 绮剧‘澶嶇幇)")
print("=" * 70)


def _risk_from_logits(logits):
    """浠庣鏁ｆ椂闂村嵄闄╃巼 logits 閲嶇畻 risk."""
    logits = np.asarray(logits, dtype=np.float64)
    hazards = 1.0 / (1.0 + np.exp(-logits))
    surv = np.cumprod(1.0 - hazards)
    return -float(surv.sum())


def _cindex(risks, censors, times):
    event = (1 - np.asarray(censors)).astype(bool)
    return concordance_index_censored(
        event, np.asarray(times, dtype=np.float64),
        np.asarray(risks, dtype=np.float64), tied_tol=1e-8
    )[0]


def _load_folds(root):
    """杩斿洖 {fold: patient_results_dict}."""
    folds = {}
    for fold in range(5):
        p = os.path.join(root, f"split_{fold}_results_final.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                folds[fold] = pickle.load(f)
    return folds


def _fold_cindex_single(pr):
    """鍗?seed 鍗?fold c-index."""
    cids = list(pr.keys())
    risks = [pr[c]["risk"] for c in cids]
    censors = [pr[c]["censor"] for c in cids]
    times = [pr[c]["time"] for c in cids]
    return _cindex(risks, censors, times)


def _fold_cindex_ensemble(pr_list, mode="risk"):
    """澶?seed 鍗?fold 闆嗘垚 c-index."""
    common = set(pr_list[0].keys())
    for pr in pr_list[1:]:
        common &= set(pr.keys())
    common = sorted(common)
    risks, censors, times = [], [], []
    for cid in common:
        if mode == "risk":
            r = np.mean([pr[cid]["risk"] for pr in pr_list])
        else:
            logit_stack = np.stack([np.asarray(pr[cid]["logits"], dtype=np.float64) for pr in pr_list], axis=0)
            r = _risk_from_logits(logit_stack.mean(axis=0))
        risks.append(r)
        censors.append(pr_list[0][cid]["censor"])
        times.append(pr_list[0][cid]["time"])
    return _cindex(risks, censors, times)


# ====== 涓绘祦绋?======
seed3_folds = _load_folds(SEED3_DIR)
seed5_folds = _load_folds(SEED5_DIR)
print(f"  seed 3: {len(seed3_folds)} folds, 鎬?case={sum(len(v) for v in seed3_folds.values())}")
print(f"  seed 5: {len(seed5_folds)} folds, 鎬?case={sum(len(v) for v in seed5_folds.values())}")

folds = sorted(set(seed3_folds) & set(seed5_folds))
print(f"  鍏卞悓 fold: {folds}")

# 鍗?seed c-index
print("\n[1] 鍗?seed (鎸?fold) c-index")
for label, sf in [("seed 3", seed3_folds), ("seed 5", seed5_folds)]:
    cs = [_fold_cindex_single(sf[f]) for f in folds]
    print(f"  {label}: per-fold = {[f'{c:.4f}' for c in cs]}")
    print(f"          mean = {np.mean(cs):.4f}")

# 2-seed 闆嗘垚
print("\n[2] 2-seed 闆嗘垚 (鎸?fold 闆嗘垚, 鍐嶅钩鍧?")
for mode in ("risk", "logits"):
    cs = []
    for f in folds:
        pr_list = [seed3_folds[f], seed5_folds[f]]
        cs.append(_fold_cindex_ensemble(pr_list, mode=mode))
    print(f"  mode={mode}: per-fold = {[f'{c:.4f}' for c in cs]}")
    print(f"             mean = {np.mean(cs):.4f}")

# 鏈€缁堟眹鎬?print("\n" + "=" * 70)
print("缁撴灉姹囨€?)
print("=" * 70)
s3_mean = np.mean([_fold_cindex_single(seed3_folds[f]) for f in folds])
s5_mean = np.mean([_fold_cindex_single(seed5_folds[f]) for f in folds])
ens_risk = np.mean([_fold_cindex_ensemble([seed3_folds[f], seed5_folds[f]], 'risk') for f in folds])
ens_logits = np.mean([_fold_cindex_ensemble([seed3_folds[f], seed5_folds[f]], 'logits') for f in folds])
print(f"  seed 3 (鍗曡窇 5-fold mean):       {s3_mean:.4f}   鈫?鍘熻褰?0.7105")
print(f"  seed 5 (鍗曡窇 5-fold mean):       {s5_mean:.4f}   鈫?鍘熻褰?0.7158")
print(f"  鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
print(f"  2-seed 闆嗘垚 (risk 骞冲潎):         {ens_risk:.4f}")
print(f"  2-seed 闆嗘垚 (logits 骞冲潎):       {ens_logits:.4f}   鈫?鍘熻褰?0.7237")
print("=" * 70)