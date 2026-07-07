#!/usr/bin/env python3
"""鑱氬悎 fold3 smoke 缁撴灉 鈥?浠?epoch_curve 鍙?val_cindex"""
import pandas as pd
from pathlib import Path
import json

SWEEP = Path("/data1/sweep_results_30ep/fold3_smoke_1ep")
runs = sorted([d for d in SWEEP.iterdir() if d.is_dir() and d.name.startswith("a")])

rows = []
for d in runs:
    name = d.name  # e.g. a0.0_b-nll_surv_ot-v45_default
    # 瑙ｆ瀽: a{alpha}_b-{bag_loss}_ot-{ot}
    # bag_loss 鍙兘鍚笅鍒掔嚎锛熶笉锛岄檺瀹?nll/cox/rank
    alpha_str = name.split("_", 1)[0]  # 'a0.0'
    alpha = float(alpha_str.lstrip("a"))
    rest = name.split("_", 1)[1]  # 鍘?'a0.0_'
    bag_loss = rest.split("_ot-")[0].replace("b-", "")
    ot = rest.split("_ot-")[1]

    # 鎵?epoch_curve
    ec = list(d.glob("blca/SurvOTRank_otehv2_rankevent/*/epoch_curve_fold3.csv"))
    if not ec:
        rows.append({"tag": name, "alpha": alpha, "bag_loss": bag_loss, "ot": ot,
                     "status": "NO_CURVE"})
        continue
    try:
        df = pd.read_csv(ec[0])
        last = df.iloc[-1]
        # val_cindex 鍒楀悕
        c = last.get("val_cindex", float("nan"))
        c_ipcw = last.get("val_cindex_ipcw", float("nan"))
        ibs = last.get("val_IBS", float("nan"))
        iauc = last.get("val_iauc", float("nan"))
        rows.append({"tag": name, "alpha": alpha, "bag_loss": bag_loss, "ot": ot,
                     "status": "OK", "val_cindex": c, "val_cindex_ipcw": c_ipcw,
                     "val_IBS": ibs, "val_iauc": iauc,
                     "epoch": int(last.get("epoch", -1))})
    except Exception as e:
        rows.append({"tag": name, "alpha": alpha, "bag_loss": bag_loss, "ot": ot,
                     "status": f"ERR: {e}"})

res = pd.DataFrame(rows)
res.to_csv(SWEEP / "summary_all.csv", index=False)

# Print as table (only nll rows are reliable)
print("\n=== FOLD 3 SMOKE 鈥?1 epoch results ===\n")
print(res.to_string(index=False))

print("\n\n=== NLL_SURV ONLY pivot ===\n")
nll = res[(res["bag_loss"] == "nll_surv") & (res["status"] == "OK")].copy()
if not nll.empty:
    pivot = nll.pivot_table(index="alpha", columns="ot", values="val_cindex", aggfunc="mean")
    print(pivot.to_string())
    print("\n--- same for val_cindex_ipcw ---")
    pivot2 = nll.pivot_table(index="alpha", columns="ot", values="val_cindex_ipcw", aggfunc="mean")
    print(pivot2.to_string())
else:
    print("(no nll_surv rows)")

print("\n=== COX_SURV / RANK_SURV all FAILED (PLE loss bug) ===")
fail = res[res["status"] != "OK"]
if not fail.empty:
    print(fail[["tag", "status"]].to_string(index=False))