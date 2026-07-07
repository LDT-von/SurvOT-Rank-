#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V45 澶?seed 闆嗘垚璇勪及銆?
瀵硅嫢骞蹭釜 seed 鐨?V45 璁粌缁撴灉鍋氶獙璇侀泦闆嗘垚锛氭寜 fold 璇诲彇鍚?seed 鐨?split_{fold}_results_final.pkl锛堝唴鍚瘡涓梾浜虹殑 risk / censor / time / logits锛夛紝
瀵瑰悓涓€ fold 鍐呭悓涓€鐥呬汉鐨勯娴嬭法 seed 骞冲潎鍚庨噸绠?C-index锛屾姤鍛?5-fold mean卤std锛?骞朵笌鍗?seed 缁撴灉瀵圭収銆?
闆嗘垚鏂瑰紡涓ょ锛屽潎杈撳嚭锛?  - risk 骞冲潎锛氱洿鎺ュ risk 鍒嗘暟璺?seed 鍙栧潎鍊硷紙鏈€绠€鍗曪紝閫傚悎 C-index锛?  - logits 骞冲潎锛氬绂绘暎鍗遍櫓鐜?logits 璺?seed 鍙栧潎鍊煎悗閲嶇畻 risk锛堟洿瑙勮寖锛?
fold split 鐢遍鐢熸垚 split 鏂囦欢鍐冲畾锛屼笉鍙?--seed 褰卞搷锛屽洜姝ゅ悓涓€ fold 鍚?seed 鐨?楠岃瘉闆嗙梾浜轰竴鑷达紝鎸?case_id 骞冲潎涓ユ牸鏈夋晥銆傝剼鏈細鍙栧悇 seed 鐨?case_id 浜ら泦浠ョǔ鍋ュ鐞嗐€?
鐢ㄦ硶:
  python ensemble_eval.py --dirs DIR1 DIR2 ... [--n_classes 4]
"""

import argparse
import glob
import os
import pickle
import re

import numpy as np

try:
    from sksurv.metrics import concordance_index_censored
except Exception as e:  # pragma: no cover
    raise SystemExit(f"闇€瑕?scikit-survival: {e}")


_FOLD_RE = re.compile(r"split_(\d+)_results_final\.pkl$")


def _load_seed_dir(root):
    """鍦ㄤ竴涓?seed 鐩綍涓嬮€掑綊鏌ユ壘 split_{fold}_results_final.pkl銆?
    杩斿洖 {fold: patient_results_dict}
    """
    pkls = glob.glob(os.path.join(root, "**", "split_*_results_final.pkl"), recursive=True)
    pkls += glob.glob(os.path.join(root, "split_*_results_final.pkl"))
    folds = {}
    for p in sorted(set(pkls)):
        m = _FOLD_RE.search(os.path.basename(p))
        if not m:
            continue
        fold = int(m.group(1))
        with open(p, "rb") as f:
            folds[fold] = pickle.load(f)
    return folds


def _risk_from_logits(logits):
    """浠庣鏁ｆ椂闂村嵄闄╃巼 logits 閲嶇畻 risk锛堣秺澶ч闄╄秺楂橈級銆?
    logits: [num_classes]  -> hazards=sigmoid -> S=cumprod(1-h) -> risk=-sum(S)
    """
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


def _fold_cindex_single(pr):
    """鍗?seed 鍗?fold 鐨?C-index锛堢敤 pkl 鍐呭凡瀛?risk锛夈€?""
    cids = list(pr.keys())
    risks = [pr[c]["risk"] for c in cids]
    censors = [pr[c]["censor"] for c in cids]
    times = [pr[c]["time"] for c in cids]
    return _cindex(risks, censors, times)


def _fold_cindex_ensemble(pr_list, mode="risk"):
    """澶?seed 鍗?fold 闆嗘垚 C-index銆?
    pr_list: [patient_results_dict, ...]锛堝悓涓€ fold 鐨勫涓?seed锛?    mode: 'risk' 鐩存帴骞冲潎 risk锛?logits' 骞冲潎 logits 鍚庨噸绠?risk
    """
    # 鍙?case_id 浜ら泦锛屼繚璇佸榻?    common = set(pr_list[0].keys())
    for pr in pr_list[1:]:
        common &= set(pr.keys())
    common = sorted(common)
    if not common:
        return None, 0

    risks, censors, times = [], [], []
    for cid in common:
        if mode == "risk":
            r = np.mean([pr[cid]["risk"] for pr in pr_list])
        else:  # logits
            logit_stack = np.stack([np.asarray(pr[cid]["logits"], dtype=np.float64)
                                    for pr in pr_list], axis=0)
            r = _risk_from_logits(logit_stack.mean(axis=0))
        risks.append(r)
        # censor/time 鍚?seed 鐩稿悓锛屽彇绗竴涓?        censors.append(pr_list[0][cid]["censor"])
        times.append(pr_list[0][cid]["time"])
    return _cindex(risks, censors, times), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="鍚?seed 鐨勭粨鏋滄牴鐩綍")
    ap.add_argument("--n_classes", type=int, default=4)
    args = ap.parse_args()

    seed_folds = {}
    for d in args.dirs:
        name = os.path.basename(os.path.normpath(d))
        folds = _load_seed_dir(d)
        if folds:
            seed_folds[name] = folds
            print(f"[load] {name}: folds={sorted(folds.keys())}")
        else:
            print(f"[warn] {name}: 鏈壘鍒?split_*_results_final.pkl锛岃烦杩?)

    if len(seed_folds) < 2:
        raise SystemExit("鑷冲皯闇€瑕?2 涓湁鏁?seed 鐩綍鎵嶈兘闆嗘垚")

    # 鎵€鏈?seed 鍏卞悓鎷ユ湁鐨?fold
    all_folds = None
    for folds in seed_folds.values():
        fs = set(folds.keys())
        all_folds = fs if all_folds is None else (all_folds & fs)
    all_folds = sorted(all_folds)
    print(f"\n[folds] 鍙備笌闆嗘垚鐨?fold: {all_folds}\n")

    # ---- 鍗?seed 閫?fold C-index ----
    print("=" * 64)
    print("鍗?seed 缁撴灉锛堝悇鑷?5-fold mean卤std锛?)
    print("=" * 64)
    for name, folds in seed_folds.items():
        cs = [_fold_cindex_single(folds[f]) for f in all_folds]
        print(f"  {name:32s} mean={np.mean(cs):.4f}  std={np.std(cs):.4f}  "
              f"folds={[f'{c:.4f}' for c in cs]}")

    # ---- 闆嗘垚閫?fold C-index ----
    for mode in ("risk", "logits"):
        print("\n" + "=" * 64)
        print(f"闆嗘垚缁撴灉锛坽mode} 骞冲潎锛寋len(seed_folds)} seeds锛?)
        print("=" * 64)
        cs = []
        for f in all_folds:
            pr_list = [seed_folds[name][f] for name in seed_folds]
            c, n = _fold_cindex_ensemble(pr_list, mode=mode)
            cs.append(c)
            print(f"  fold {f}: C-index={c:.4f}  (n={n})")
        print(f"  --> ensemble mean={np.mean(cs):.4f}  std={np.std(cs):.4f}")

    print("\n瀵圭収锛歏45 鍗?seed(=3) 璁板綍鍊?= 0.7105 卤0.0181")
    print("baseline v9 = 0.7078 锛涚洰鏍?鈮?.72")


if __name__ == "__main__":
    main()
