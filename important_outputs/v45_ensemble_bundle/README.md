# V45 otehv2_rankevent 涓€閿帹鐞?+ 闆嗘垚鍖?
## 涓€閿鐜?0.7237

```bash
cd <SurvOT-Rank repo>
python important_outputs/v45_ensemble_bundle/exact_ensemble_07237.py
```

杈撳嚭锛?```
seed 3 (鍗曡窇 5-fold mean):       0.7105
seed 5 (鍗曡窇 5-fold mean):       0.7158
2-seed 闆嗘垚 (logits 骞冲潎):       0.7237   鈫?鐩爣
```

**鍙渶 1 绉?*锛岀洿鎺ヨ pkl 閲嶇畻 c-index銆?
---

## 鏂囦欢娓呭崟

| 鏂囦欢 | 澶у皬 | 浣滅敤 |
|---|---|---|
| `seed3_fold{0..4}.pth` | 118M 脳 5 | seed=3 璁粌濂界殑 5 涓?fold 妯″瀷鏉冮噸 |
| `seed5_fold{0..4}.pth` | 118M 脳 5 | seed=5 璁粌濂界殑 5 涓?fold 妯″瀷鏉冮噸 |
| `exact_ensemble_07237.py` | 5K | **鎺ㄨ崘**锛? 绉掔簿纭鐜?0.7237锛堣 pkl锛?|
| `quick_infer_ensemble.py` | 11K | 澶囬€夛細浠?pth 閲嶆柊鎺ㄧ悊 5-10 鍒嗛挓 |

---

## Seed 鏄粈涔堬紵

**Seed 鏄?闅忔満鏁扮瀛?锛屼笉鏄畻娉曠粨鏋?*銆?
璁粌绁炵粡缃戠粶闇€瑕佸緢澶氶殢鏈烘暟锛?- 鍒濆鍖?30M 涓潈閲嶏紙姣忎釜浠?N(0,1) 閲囨牱锛?- 璁粌鏃舵暟鎹墦涔遍『搴?- 鍚勭 noise / dropout

```
seed = 闅忔満鏁板簭鍒楃殑"璧风偣鍧愭爣"
     = 鏁翠釜璁粌杩囩▼鐨?绉嶅瓙"

涓嶅悓鐨?seed 鈫?涓嶅悓鐨勫垵濮嬫潈閲?鈫?瀛﹀埌涓嶅悓鐨勬ā鍨?鈫?涓嶅悓鐨?c-index
鐩稿悓鐨?seed (鍦?fix 鎵€鏈夐潪纭畾鎬ф簮鍚? 鈫?瀹屽叏涓€鑷寸殑璁粌缁撴灉
```

### 涓轰粈涔?seed3 + seed5 鑳芥彁鍗囷紵

```
seed 3 瀛﹀埌涓€缁勬潈閲?W3: 閿欏鏍锋湰 A 姝ｇ‘, 閿欏鏍锋湰 B 閿欒
seed 5 瀛﹀埌涓€缁勬潈閲?W5: 閿欏鏍锋湰 A 閿欒, 閿欏鏍锋湰 B 姝ｇ‘

闆嗘垚 = (W3 + W5) / 2: 閿欏鏍锋湰 A銆丅 閮芥帴杩戞纭?鉁?
鏁板鏈川: 鐙珛璇樊鍧囧€肩殑鏂瑰樊 = sigma^2 / n
- 1 seed:   var = sigma^2
- 2 seeds:  var = sigma^2/2  鈫?c-index 鎻愬崌 ~0.01
- 4 seeds:  var = sigma^2/4
- 8 seeds:  var = sigma^2/8  鈫?缁х画鎻愬崌
```

### 涓轰綍 seed=3锛?
浠ｇ爜閲?`--seed 3` 鏄换鎰忛€夌殑锛堥粯璁わ級锛屽搴旓細
- `torch.manual_seed(3)` 鈫?妯″瀷鍒濆鍖?- `np.random.seed(3)` 鈫?鏁版嵁澧炲己
- `random.seed(3)` 鈫?shuffle 椤哄簭

鍙互閫変换浣曟暟銆? 鏄釜灏忔暣鏁帮紝甯哥敤浣滃熀绾裤€?
### 涓轰綍 seed=5 鏄柊鍔犵殑锛?
涔嬪墠鐨?0.7105 鏄?seed=3 鍗曠嫭璺戝嚭鐨?杩愭皵濂?鐨勭粨鏋溿€?浣嗗洜涓?`train_runner.py` 涔嬪墠娌℃湁姝ｇ‘浼犻€?seed锛堝皯浜?`set_global_seed` 璋冪敤锛夛紝
**姣忔瀹為檯璺戦兘浼氫笉鍚?*锛?.7105 鏄笉绋冲畾鐨勩€?
鍔犱笂 seed=5 鍚庯細
- seed 3 = 0.7105锛堝凡瀛橈級
- seed 5 = 0.7158锛堥噸璺戯紝鍥哄畾 seed锛?- **闆嗘垚 = 0.7237**锛堢ǔ瀹氭彁鍗?~0.01锛?
---

## 宸ヤ綔鍘熺悊

### 1. exact_ensemble_07237.py锛堟帹鑽愶紝1 绉掞級

璁粌鏃跺凡缁忔妸姣忎釜 case 鐨?`risk + logits + censor + time` 淇濆瓨鍒?`split_*_results_final.pkl`銆?鑴氭湰鐩存帴璇昏繖浜?pkl锛屾寜 fold 闆嗘垚 c-index銆?
```python
for fold in 0..4:
    for each case_id in (seed3 鈭?seed5):
        logit_avg = (logit_seed3 + logit_seed5) / 2
        risk_avg  = sigmoid 鈫?cumprod(1-h) 鈫?-sum
    cindex(fold) = concordance_index_censor + time + risk_avg
final = mean(cindex[0..4])  # 0.7237
```

### 2. quick_infer_ensemble.py锛堝閫夛紝5-10 鍒嗛挓锛?
濡傛灉浣犲彧鏈?pth 娌℃湁 pkl锛岃剼鏈細閲嶆柊鍔犺浇妯″瀷璺?10 娆℃帹鐞嗐€?娉ㄦ剰锛?*浼氫笌绮剧‘鍊兼湁 卤0.01 宸紓**锛堜竴浜?svs 鐗瑰緛鏂囦欢缂哄け锛岃缁冩椂璺宠繃锛夈€?
---

## 璁粌鍙傛暟锛堝凡鍥哄寲锛?
| 鍙傛暟 | 鍊?|
|---|---|
| model | otehv2_rankevent |
| 鏁版嵁闆?| blca (TCGA) |
| 鏍囩 | survival_months_dss (Disease-Specific Survival) |
| RNA 鏍煎紡 | Pathways (329 涓€氳矾) |
| 浠诲姟鍒嗙被鏁?| 4 (DSS bins) |
| 瀛︿範鐜?| 5e-4 |
| batch_size | 4 |
| epochs | 30 |
| 浼樺寲鍣?| Adam |
| OT eps / iter / warmup | 0.05 / 50 / 5 |
| otehv2 events / heads / layers | 24 / 4 / 4 |
| lambda_otehv2_ot / div / event_surv / recon | 0.06 / 0.01 / 0.25 / 0.2 |
| lambda_rankevent (per_event / rank / global_cons / gate_ent) | 0.15 / 0.15 / 0.02 / 0.005 |
| rankevent_eps (start / end / anneal_epochs) | 0.10 / 0.05 / 12 |
| rankevent_global_init | -2.0 |
| slot (wsi / omics / iters) | 8 / 8 / 5 |
| num_patches | 2048 |
| encoding_dim | 1024 |
| wsi_projection_dim | 256 |

---

## 鏂囦欢浣嶇疆

```
important_outputs/v45_ensemble_bundle/
鈹溾攢鈹€ exact_ensemble_07237.py         (鎺ㄨ崘)
鈹溾攢鈹€ quick_infer_ensemble.py         (澶囬€?
鈹溾攢鈹€ seed3_fold0.pth ... seed3_fold4.pth    (5 脳 118M)
鈹溾攢鈹€ seed5_fold0.pth ... seed5_fold4.pth    (5 脳 118M)
鈹斺攢鈹€ README.md
```

---

## 甯歌闂

**Q: 鎴戞病鏈?pth 鏂囦欢锛岃兘鐩存帴璁粌鍑?0.7237 鍚楋紵**
A: 鍙互锛屼絾闇€瑕侊細
1. 淇 `train_runner.py` 鐨?seed 鍒濆鍖栵紙宸蹭慨锛孭R 鍦?train_runner.py L51-67锛?2. 璺?`bash run_v45_final.sh` 涓ゆ锛坰eed=3, seed=5锛夛紝姣忔 ~30 鍒嗛挓
3. 璺?`python survot_rank/research/methods/prognostic_event_transport/ensemble_eval.py --dirs seed3_dir seed5_dir`

**Q: 鍗?seed 闆嗘垚 (logits) vs (risk) 鍝釜濂斤紵**
A: logits 鏇村ソ锛?.7237 vs 0.7208锛夈€傚洜涓猴細
- risk 鏄?logits 缁?sigmoid+cumprod 鐨勯潪绾挎€ц緭鍑?- 鐩存帴骞冲潎 risk 浼?鎶樹腑"闈炵嚎鎬у尯闂?- 骞冲潎 logits 鍚庡啀绠?risk 淇濈暀浜嗕笉纭畾鎬т俊鎭?
**Q: 鑳界敤 seed 7/11/鏇村 seed 闆嗘垚鍚楋紵**
A: 鍙互锛佹洿澶?seed 閫氬父鏇寸ǔ銆傜洿鎺ュ姞 seed=7, seed=11 閲嶈窇锛坮un_v45_final.sh 鏀?SEEDS 鏁扮粍锛夛紝鐒跺悗 ensemble_eval.py 鍔?4 涓?dir銆?
**Q: 0.7237 鏄?5-fold CV 鐨?mean 杩樻槸 test锛?*
A: **5-fold CV mean**銆傛瘡涓?fold 鍐?
- 304 涓?train sample (璁粌)
- 76 涓?val sample (璇勪及, 鐢?model_best_s{fold}.pth)
- 闆嗘垚鏃舵寜 case_id 瀵归綈, 姹?fold 鍐?c-index, 鍐?mean 5 fold
- 鎬昏瘎浼?case = 380 涓?BLCA 鐥呬汉

**Q: pth 鏂囦欢鑳借法鏈哄櫒鐢ㄥ悧锛?*
A: 鑳斤紝浣嗛渶瑕侊細
- 鍚屼竴浠?SlotSPE 鐩綍缁撴瀯
- 鍚屼竴浠?dataset_csv (blca.csv, omics_csv, splits)
- 鍚屼竴浠界壒寰?(WSI features, 1024-dim UNI)

**Q: 娌℃湁 GPU 鑳借窇鎺ㄧ悊鍚楋紵**
A: 鑳斤紝鎱竴浜涖€侰PU 璺?5-10 鍒嗛挓銆侴PU < 1 鍒嗛挓銆
