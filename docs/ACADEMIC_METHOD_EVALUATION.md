# SurvOT-Rank 方法学术潜力与撞题审计

> 最后更新：2026-08-11
> 文献检索截止：2026-07-21
> 仓库快照：`8278f6f`
> 评价对象：`survot_rank/training/model_factory.py` 中注册的全部方法，以及有实验记录的历史版本  
> 结果主来源：`FINAL_SUMMARY.md`、`reproducibility_archives/manifest.json` 与逐折结果档案；未进入仓库的服务器汇总只标为“暂定”
> 用途：以后每次要求“评价所有方法”时更新本文件，不用重新建立另一套口径

## 1. 先给结论

当前 `METHOD_REGISTRY` 共有 **21 个方法标识**。相较 2026-07-21 的 12 个标识，新增了 DCT v3.6/listwise、v3.8、v3.8.2、v3.8.3、v3.9、IST-Surv v4.0、Evidence Ledger v4.1、ArcSurv、ACT-Surv v4.2 等实现。本文同时按“科学家族”和“具体版本”评价，避免把同一主干的实验分支包装成多篇论文。

目前不应该把所有实现都包装成独立论文。它们多数是同一研究路线的连续演化，拆开投稿会有明显的“切香肠”风险。

论文优先级应固定为：

1. **DCT v3.8.2 fixed-full：当前唯一 DCT 主线，也是当前投稿成熟度第一。** 已完成 6 癌种 × 5 折、UNI2-h、50 epoch、clean 协议；相同协议下对 IST-Surv 为 4 胜、1 负、1 持平。但 MGPTR 单项、自适应权重和 v3.8 三个一致性损失均未证明稳定增益，因此论文必须把“预测性能”和“DCT 特有机制贡献”分开陈述。
2. **CA-PSA full：第二条独立候选主线。** 旧协议 BLCA 暂定 `0.7217±0.0383`，但尚缺统一 UNI2-h 逐折档案。它的核心风险仍是 SlotSPE、AdaSlot、BO-QSA、Dual-State Slot Attention 等部件的组合感；只有 identity/state/gate 的统一机制与身份稳定性被证实后，才适合独立成文。
3. **CATET repaired：纯 idea 新颖性最高，但证据仍最弱。** 当前可信结果只有 BLCA fold0/2=`0.6458/0.6837`。保留 repaired 50ep 作为跨癌种筛选版本，不恢复已作废的旧 `catet_fix`。
4. **ArcSurv staged：保留最终筛选版本；hard-repaired 版本淘汰。** staged 50ep 的 BLCA fold1/2/4 均值为 `0.6910`，而 hard repair 的 fold1 仅 `0.5665`。不能把“修复失败”误写为“整个 ArcSurv 方法已经终止”。
5. **IST-Surv v4.0、Evidence Ledger v4.1、v3.9、v3.8.3：不再作为论文主线。** IST 已完成六癌种，但三档消融证明核心 cost feedback 和辅助损失没有增益；v4.1 修复无效；v3.9/v3.8.3 已形成明确负结果。
6. **ACT-Surv v4.2：只有 idea 身份，没有性能排名。** 在 ArcSurv staged 的原型分化与跨癌种表现确认前，不启动完整实验。

## 2. 评价口径

### 2.1 分数不是录用概率

- “当前可投档次”按**现在已经落盘的证据**判断。
- “补齐后合理上限”是假设缺失实验全部完成且结论稳定，不是承诺。
- 期刊分区会随年份、学科和中科院/JCR 口径变化；本文的 Q1–Q4 是研究质量档次判断，不对应某一本期刊的永久分区。
- CCF A/B/C 只用于计算机会议档次判断。医学期刊与生物信息学期刊不能机械换算为 CCF。
- 单一 BLCA、单次 5-fold、每折挑选 best epoch，只足以做开发期比较；对高档投稿还需要独立重复、置信区间、显著性检验和外部/多癌种验证。

### 2.2 严格评分维度

每条方法按 10 分制观察五项：

- **创新性**：是否提出了新的、不可被一个现成模块替换的问题与机制。
- **证据强度**：结果完整性、稳定性、消融、统计检验和跨癌种验证。
- **叙事清晰度**：核心机制能否用一条因果链解释，损失是否过多。
- **撞题风险**：越高表示越接近已有论文，不等同于抄袭。
- **过度宣称风险**：是否容易把相关性、注意力或模型敏感性写成因果/生物机制。

### 2.3 “抄袭”与“撞题”不是一回事

当前代码和文档审计**没有发现可以据此认定的文字或代码抄袭证据**。但没有做全仓库逐文件许可证/代码指纹取证，所以不能给出法律意义的“绝对无抄袭”保证。

学术上的真实风险主要是：

- 使用 OT、Slot Attention、IPCW ranking、Hard-Concrete 或稀疏模块本身都不是新贡献；
- 若论文只把已有模块串联，会被审稿人判断为增量组合；
- 名称中出现 `counterfactual`、`faithful`、`evidence` 时尤其容易过度宣称；
- 同一主干的多个版本若分别投稿，可能构成重复发表或切香肠式投稿风险。

### 2.4 概念创新、代码兑现与证据成熟度必须分开

“概念创新性”评价论文问题和理想机制，“代码实现完成度”评价当前实现是否真的兑现这条机制链，“证据成熟度”评价实验能否支持结论。三者不能互相替代。按 2026-08-11 的代码与结果档案，三条候选主线应使用下面的口径：

| 方法 | 概念创新性 | 当前代码实现完成度 | 证据成熟度 | 严格解释 |
|---|---:|---:|---:|---|
| **CATET** | **7.0/10** | **5.5/10** | **2.5/10** | 问题定义尖锐，但当前阶段 cost 实际共享、干预不是重新求解 OT，且修复后仅有 2 folds |
| **DCT v3.8.2 fixed-full** | **6.5/10** | **7.0/10** | **7.0/10** | 6 癌种 × 5 折证据显著增强，但 MGPTR、自适应权重和一致性损失尚未证明独立贡献 |
| **CA-PSA** | **6.0/10** | **5.5/10** | **4.0/10** | 结构已实现且有暂定分数，但核心的 slot identity 稳定性和动态容量尚未被验证 |

因此，“CATET 7.0、DCT 6.5、CA-PSA 6.0”只能作为**理想论文 idea 的创新性排序**。若评价当前可运行代码，CATET 不能排在 DCT 前面；若评价当前投稿成熟度，顺序仍是 DCT > CA-PSA > CATET。

## 3. 家族与版本两级评价

### 3.1 六个方法家族的严格评价

当前 21 个注册标识可归并为 **6 个科学方法家族**。家族评分回答“idea 是否独立”，版本评分回答“某个实现是否值得保留”；二者不能混用。

| 家族 | 覆盖实现 | 核心科学问题 | 创新性 | 证据强度 | 缝合风险 | 直接撞题风险 | 当前判断 |
|---|---|---|---:|---:|---:|---|---|
| **A. OT 预后事件建模** | OTEventHazardV2、V45、V45v2、V50、V60 OT Event Rank | 用 slots、OT 和事件 token 融合 WSI/omics 并预测离散风险 | **3.5/10** | 6.0/10 | **8.5/10** | 高 | 一个基线家族，不是五个论文 idea |
| **B. 删失感知 Transport 干预** | RG-ET、Stagewise、FET、DCT、v3.6、v3.8、v3.8.2、v3.8.3、v3.9、CATET | 将删失风险集、时间阶段和证据干预作用于 transport，并检验预测变化 | **6.5/10** | **7.0/10** | **7.0/10** | 中高 | DCT v3.8.2 为唯一主版本；其余为机制、消融或负结果 |
| **C. 队列锚定自适应 Slots** | CA-PSA | 稳定跨患者/跨模态 slot 身份，并让激活数量适应患者 | **6.0/10** | 5.0/10 | **8.0/10** | 高 | 可独立于 DCT，但必须证明统一机制而非三模块拼接 |
| **D. 稳定干预与证据账本** | IST-Surv v4.0、Evidence Ledger v4.1 | 约束干预稳定性或守恒记录跨模态证据 | 6.0/10 | 6.0/10 | 6.5/10 | 中 | 实验已否定当前实现的增益，不再作为主线 |
| **E. 队列原型风险组合** | ArcSurv、ACT-Surv v4.2 | 用队列原型及其凸组合表示患者风险，进一步研究原型间 transport | 6.0/10 | 3.5/10 | 6.5/10 | 中 | ArcSurv staged 待跨癌种；v4.2 尚无性能证据 |
| **F. 患者特异预后 Circuits** | V70 PSPC | 为每名患者学习稀疏、可组合的预后模块连接 | **5.5/10** | 3.5/10 | **7.5/10** | 中高 | 科学问题可独立，但当前仅探索线 |

**家族 A 的严格判断。** OT、多模态生存、slots/event tokens、ranking 和多损失组合均已有充分近邻。MOTCat 已做 WSI–genomics OT 生存对齐，MMP 已做 prototype/pathway token 与 OT cross-alignment，SlotSPE 已做 slot-based prognostic events。因此该家族的工程价值明显高于学术新颖性，最合适的角色是统一强基线和演进消融。

**家族 B 的严格判断。** 单个部件都不是新东西，甚至“distributional counterfactual + OT”也已被 DISCOUNT 占用；但定向检索仍未发现完整同构的“训练折删失风险集参考 → transport cost 干预 → evidence-conditioned re-Sinkhorn → 生存输出敏感性”链条。它是否摆脱缝合，完全取决于这条链能否通过方向一致性、剂量响应、随机 anchor 阴性对照和结构消融被证实，而不是只靠 C-index。

**家族 C 的严格判断。** SlotSPE、AdaSlot、BO-QSA、Dual-State Slot Attention、SurvQ 已分别覆盖主要部件；2025 年的 Adaptive Prototype Learning 还使用双组 learnable queries 和自适应 prototypes 做多模态癌症生存。CA-PSA 只有证明 anchor/state/gate 构成不可拆分的生存特有机制，并产生稳定可复现的 slot identity，才能避免被称为组合创新。

**家族 D 的严格判断。** IST-Surv 和 Evidence Ledger 都有清晰机制叙事，但最新消融与修复实验没有证明核心机制带来收益。这里应区分“idea 可写”与“当前实现可投”：完整负结果提高了判断可信度，却不能提高论文优先级。

**家族 E 的严格判断。** ArcSurv/ACT-Surv 的独立性取决于队列原型是否真实分化、患者组合是否具有稳定风险含义。staged ArcSurv 仍可筛选；hard repair 已失败；ACT-Surv 在原型分化前提确认前不能仅凭更复杂的 transport 故事获得高排名。

**家族 F 的严格判断。** 暂未检索到与 V70 在 WSI+omics 生存场景完全同构的 patient-specific circuit 方法，但 Neural Attentive Circuits 已覆盖“联合学习模块参数与稀疏连接”的通用思想。因此 V70 的领域迁移本身不够，必须证明患者电路具有跨 seed 稳定性、删失生存特异监督和普通稀疏 MoE/动态路由不具备的能力。

### 3.2 每个注册版本的身份

| 注册版本 | 所属家族 | 是否独立论文 idea | 正确身份 |
|---|---|---|---|
| OT Event Hazard V2 | A | 否 | 全部 event-transport 方法的架构祖先 |
| V45 RankEvent | A | 否 | ranking/event supervision 历史版本 |
| V45v2 | A | 否 | 临床/多分支扩展历史版本 |
| V50 Time-local Competing | A | 否 | 强经验基线；多损失消融对象 |
| V60 OT Event Rank | A | 否 | 紧凑基线，不是 CA-PSA |
| Rank-Guided Event Transport | B | 否 | 风险引导机制起点 |
| Stagewise Prognostic Transport | B | 否 | 阶段化假设/消融 |
| Faithful Evidence Transport | B | 否 | evidence keep/remove 解释性消融 |
| DCT v3.3 | B | 否 | 历史 score-first 基线，不再是最终主版本 |
| DCT v3.6 listwise | B | 否 | IPCW/listwise 诊断与消融版本 |
| DCT v3.8 | B | 否 | direction/dose/reconfiguration 一致性损失筛选版本 |
| **DCT v3.8.2 fixed-full** | B | **是，当前主候选** | 家族 B 的唯一最终主版本 |
| DCT v3.8.3 | B | 否 | centered consistency 负结果，停止 |
| DCT v3.9 | B | 否 | risk-simplex 负结果，停止 |
| CATET repaired | B | 暂不单独 | 高创新平行分支，等待统一跨癌种筛选 |
| **CA-PSA** | C | **是，条件成立时** | 独立方法主线 |
| IST-Surv v4.0 | D | 否 | 六癌种负机制结果；保留为消融/对照 |
| Evidence Ledger v4.1 | D | 否 | 修复无效，停止主推 |
| ArcSurv staged | E | 条件不足 | 保留的唯一 ArcSurv 跨癌种筛选版本 |
| ArcSurv hard-repaired | E | 否 | 修复失败版本，淘汰 |
| ACT-Surv v4.2 | E | 条件不足 | idea-only；尚无正式性能结果 |
| V70 PSPC | F | 条件不足 | 独立探索方向，当前不具备论文主线证据 |

### 3.3 版本总表：当前证据与投稿判断

| 优先级 | 方法 | 当前最好证据 | 活跃损失项* | 创新性 | 撞题风险 | 当前可投档次 | 补齐后合理上限 | 定位 |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | **DCT v3.8.2 fixed-full** | 6 癌种 × 5 折：UCEC `0.8224`、KIRC `0.8071`、BLCA `0.7107`、HNSC `0.6632`、SKCM `0.6608`、LUSC `0.6204` | NLL + IPCW + 固定辅助项 | **6.5/10** | **中高** | Q3；补齐机制消融后可试 Q2 | Q1/Q2；CCF B 有条件 | 第一主线；跨癌种证据最完整 |
| 2 | **CA-PSA full** | 旧协议服务器暂定 BLCA 5-fold `0.7217±0.0383`；尚无统一 UNI2-h 逐折档案 | 3 | **6.0/10** | **高** | 当前不宜正式投稿 | Q1/Q2；CCF B/C 有条件 | 第二独立主线候选 |
| 3 | **CATET repaired** | BLCA fold0/2=`0.6458/0.6837`，均值 `0.6648` | 4 | **7.0/10** | 中 | 暂不可投 | Q2；CCF C 有条件 | 纯 idea 强，证据弱；已进入最终筛选计划 |
| 4 | **ArcSurv staged** | BLCA fold1/2/4=`0.7132/0.6457/0.7141`，均值 `0.6910` | 多项 | 5.5/10 | 中 | 暂不可投 | Q2/Q3 有条件 | staged 保留；hard repair 淘汰 |
| 5 | **ACT-Surv v4.2** | 尚无正式结果 | 待定 | 暂定 6.5/10 | 中 | 不可投稿 | Q2/CCF C 有条件 | idea-only；等待 ArcSurv 分化前提 |
| 6 | **V50 Time-local Competing** | 历史 5-fold best `0.7148±0.028`；last5 `0.6572±0.012` | 8 | 5.0/10 | 高 | Q3/Q2 应用型 | Q2；CCF C | 强历史基线 |
| 7 | **IST-Surv v4.0** | 6 癌种 × 5 折完成；仅 KIRC 比 DCT 高 `0.007`，三档消融 `0.7072/0.7055/0.7053` | factual + cost feedback | 5.5/10 | 中 | 不建议独立投稿 | Q2/Q3，需重新证明机制 | 实验完整，但当前机制无增益 |
| 8 | **Evidence Ledger v4.1** | BLCA 50ep 三折均值 `0.6738`；修复后 fold2=`0.6436`，未改善 | 多项 | 6.0/10 | 中 | 不建议投稿 | Q2/Q3，需重构后重证 | 当前实现停止 |
| 9 | **V70 PSPC** | 服务器暂定 BLCA 5-fold `0.6786` | 3 | 5.5/10 | 中高 | Q4/Workshop | Q2/Q3；CCF C | 暂停 |
| 10 | **DCT v3.9 Risk-Simplex** | BLCA 三折均值 `0.6394`，且 fold 难度排序异常反转 | 多项 | 5.5/10 | 中高 | 不建议 | — | 明确负结果 |
| 11 | **DCT v3.8.3 Centered** | BLCA fold1=`0.5931`；修复后仍下降 | 多项 | 5.5/10 | 中高 | 不建议 | — | 明确负结果 |
| 12 | **V45/V45v2/V60/RG-ET/Stagewise/FET/OTEHV2** | 仅保留历史协议结果 | 配方各异 | 3.0–6.0/10 | 中至高 | 不独立投稿 | 应用型上限 | 演进、消融与基线集合 |

\* “活跃损失项”按当前代表性配置/设计目标统计，不把内部实现中仅用于计算、但权重为 0 的项算进去；配置变化时必须重新审计。

### 结果口径提醒

- DCT 当前正式跨癌种主结果只认 v3.8.2 fixed-full：UNI2-h、`5fold_uni2h`、50ep、clean，6 癌种共 30 folds。v3.3 的 UNI v1/旧分箱结果保留为历史记录，不能与新协议直接做方法增益归因。
- DCT v3.8.2 fixed-full 在相同协议下对 IST-Surv 为 4 胜、1 负、1 持平；但 fixed-full 优于 adaptive_full 只说明自适应权重无增益，MGPTR 单项 `0.6944 < 0.6975` 说明 MGPTR 单项无增益，不能据此证明全部固定辅助损失有效。
- IST-Surv 三档消融 factual/cost/full=`0.7072/0.7055/0.7053`，说明当前分数来自 factual 底座，不来自 cost feedback 或辅助损失。六癌种跑完提高了负结论可信度，而不是提高其论文排名。
- Evidence Ledger 50ep 三折均值降至 `0.6738`，completion 下界修复后 fold2 仍为 `0.6436`；当前实现停止。
- ArcSurv 的状态按版本区分：staged 50ep 保留为最终跨癌种筛选版本；furthest-point anchor 与 sharpness 等 hard repair 的 fold1=`0.5665`，该修复版本淘汰。
- DCT v3.4 BRCA event-aware 配方已在 fold0 epoch21 暂停：best `0.6189`。其失败来自有放回事件采样、`alpha_surv=2/3`、rank memory 与随机验证 slots 的叠加，不是“损失项数量过多”。
- DCT v3.5 R/Q/G/L 是历史诊断变体，不是四个论文方法；其设计保留在 [`docs/DCT_V35_SCREENING.md`](DCT_V35_SCREENING.md)，不再与 v3.8.2 竞争最终版本身份。
- CATET 的旧 `catet_fix` 5-fold `0.6534±0.079` / last5 `0.5474±0.032` 受到过猛早停和 eps 间断污染，已在 [`docs/roadmap/catet.md`](roadmap/catet.md) 中作废。修复验证目前只有 fold0 `0.6458`、fold2 `0.6837`，两折均值 `0.6648`；不能将其冒充为完整 5-fold，也不能将创新性 `7.0/10` 误写成 C-index。
- CA-PSA 与 V70 的数字目前来自服务器汇总，原始逐 epoch 曲线、配置快照、checkpoint 元数据尚未正式进入本仓库，因此只能标为暂定。
- `dct_fix`、DCT v3、v3.2 属于历史演进/消融，不应被包装成三条独立方法。
- 仓库同时存在 `v60_ot_event_rank` 和被服务器简称为“V60”的 CA-PSA。以后禁止只写“V60”，必须写完整方法名，避免结果串线。
- 当前可正式进入统一 UNI2-h 队列的癌种为 BLCA、UCEC、KIRC、SKCM、HNSC、LUSC；BRCA、LUAD、COADREAD、STAD 在特征覆盖达到 100% 前继续阻断，不允许回退 UNI v1 或静默零填充。

## 4. 逐项严格评价

### 4.1 DCT v3.8.2 fixed-full — Distributional Counterfactual Transport

**当前身份。** v3.8.2 fixed-full 是唯一 DCT 最终主版本；v3.3 是历史 score-first 基线，v3.6–v3.9 是机制筛选、损失消融或负结果，不再分别竞争论文身份。最终版本已在 BLCA、UCEC、KIRC、SKCM、HNSC、LUSC 完成 30 个 fold，均使用 UNI2-h、50ep、clean `5fold_uni2h` 协议。

| 癌种 | UCEC | KIRC | BLCA | HNSC | SKCM | LUSC |
|---|---:|---:|---:|---:|---:|---:|
| 5-fold mean C-index | **0.8224** | **0.8071** | **0.7107** | **0.6632** | **0.6608** | **0.6204** |

**核心想法。** 全局 WSI/pathway prototypes 提供跨患者可比坐标；训练折内估计时间阶段与删失分布；IPCW 风险集监督提供删失感知排序；evidence-conditioned marginals 改变 factual coupling；风险 anchor 干预与重新求解 Sinkhorn用于分析预测对 transport 几何变化的敏感性。v3.8.2 fixed-full 进一步保留固定权重的 direction、dose、reconfiguration 与 MGPTR 项。

**代码口径纠正。** “辅助损失权重为 0”不等于“对应模块全部不参与前向”。当前 `dct_evidence_marginal_strength=1.0` 时，evidence gate 仍会改变 factual OT marginals，并进入训练期预测路径；全局 prototype 坐标和阶段 pair cost 也在 factual forward 中生效。真正只作为评估期/post-hoc 分析的是高低风险 anchor 的 cost intervention 与 intervention 后的 re-Sinkhorn。因此准确表述应是：DCT 的**结构表征与 evidence-conditioned factual transport 参与训练**，但其最有辨识度的 **anchor intervention/re-Sinkhorn 没有被训练目标直接验证**。

**优点。** 目前拥有仓库中最完整的新协议证据；在与 IST-Surv 完全一致的六癌种协议下取得 4 胜、1 负、1 持平。“全局原型坐标 + 删失感知排序 + transport sensitivity”仍可形成完整论文故事。

**致命审稿问题。** 完整结果证明了配方可用，却没有证明每个新增机制有效：fixed-full `0.7209` 高于 adaptive_full `0.7122`，只说明自适应权重没有增益；MGPTR 单项 `0.6944` 低于 base `0.6975`；v3.8 三个一致性损失在匹配 BLCA 协议下仅 `+0.0033`，处于噪声量级。审稿人仍会问：六癌种表现来自 DCT 表征、IPCW ranking、UNI2-h编码器，还是固定辅助损失？因此必须补“普通 backbone + 同一 IPCW”与逐项结构消融，不能把所有固定损失都写成已验证贡献。

**历史诊断：v3.5 到底解决了什么。** `311cd88` 完成了工程与实验设计层面的四个单变量修复；这些内容用于解释版本演化，不再决定最终版本身份：

- **R** 直接修复同一 checkpoint 重复验证会因随机 slot 初始化而改变排序的问题，并去掉有放回采样导致的患者覆盖偏差；这是正确性基线，必须保留。
- **Q** 用 learned per-slot queries 检验跨样本绑定稳定性；learned query 本身已有 BO-QSA/SurvQ 近邻，不能作为新贡献。
- **G** 只检验 evidence gate 改变 OT marginals 是否有价值；它不自动证明 evidence 具有生物意义或解释忠实性。
- **L** 只检验缩小维度/层数能否缓解过拟合；它是容量对照，不是创新模块。

因此，R/Q/G/L 只保留为历史诊断与附录消融，不应包装成四个方法。

**历史协议下 BRCA 低分的定位（2026-07-20）。** 下列观察来自 UNI v1/旧协议，仅用于解释历史失败；当前 UNI2-h 特征覆盖不完整，BRCA 在覆盖达到 100% 前不得进入新协议排名：

- BRCA DSS 只有 98/1046 个观测事件（9.4%），BLCA 为 129/381（33.9%）；每个 BRCA 验证折只有 10–28 个事件，C-index 方差天然更大。
- 训练集每折约 835 人，但仍使用约 30.4M 参数、batch 8、50 epochs、固定 `lr=5e-4` 和 `weight_decay=5e-4`；相当于把 BLCA 的训练时长和容量直接迁移到一个重删失癌种。
- BRCA 五折 best epoch 为 42、26、16、7、5；对应 last5 mean 为 0.4207、0.5859、0.4508、0.5202、0.5231。五折都出现 best-to-last5 明显下降，平均下降约 0.19，说明后期在拟合验证集偶然排序/训练集表示，而不是稳定学习风险关系。
- DCT 的 IPCW ranking 是按 batch 内可比 pair 计算。BRCA 事件率低时，大多数 batch 的有效事件排序信号很少，且 censoring KM 的尾部权重更敏感；这会使梯度更噪、更容易被高容量 WSI 表征放大。
- 配置中没有启用 early stopping；best epoch 只用于汇报，训练仍固定跑满 50 epochs。因此 `best=0.6886` 是开发期峰值，`last5=0.5001` 才暴露了当前配方的不稳定性。

目前没有证据证明 BRCA 是标签反转或 split 泄漏：代码明确按 `c=0` 观测事件、`c=1` 删失处理，且 train-fold reference 拟合路径正确。仍需额外排查 WSI 缺失/多 slide 聚合和 BRCA patch 质量，不能把所有损失归因于模型。

**修复优先级。** 先做不改模型的诊断：按 fold 输出事件数、IPCW pair 数、censoring KM 曲线、WSI 缺失率和事件时间分布；然后比较 early stopping、有效 batch 增大（梯度累积）、降低学习率/训练轮数、冻结或缩小 WSI encoder。只有这些协议修复后 BRCA 仍然低，才考虑 BRCA 专门的 loss 或 stage 设计。不能直接用 BRCA 的 best epoch 重新调参后再声称跨癌种泛化，必须把协议预先固定并在所有癌种一致执行。

**最接近工作与边界。** [MOTCat](https://arxiv.org/abs/2306.08330) 已把 OT 用于 WSI–genomics 生存对齐；[MMP](https://arxiv.org/abs/2407.00224) 已使用形态/通路 prototypes 与 OT cross-alignment；2026 年的 [ProtoPathway](https://arxiv.org/abs/2605.21454) 又进一步覆盖了可学习形态 prototypes、Reactome pathway 表征、稳定跨模态对应和内生解释；删失生存的 learning-to-rank 也不是新问题（例如 [Learning to Rank for Censored Survival Data](https://arxiv.org/abs/1806.01984)）。[CURE](https://arxiv.org/abs/2602.19987) 已直接使用“multimodal counterfactual time-to-event”表述，而 AISTATS 2025 Oral 的 [DISCOUNT](https://proceedings.mlr.press/v258/you25a.html) 已明确提出“distributional counterfactual explanations with optimal transport”。因此 DCT 不能把“prototype”“pathway fusion”“OT”“ranking”“distributional counterfactual”或“counterfactual survival”单独当创新点，只能主张这些组件在**训练折删失风险集驱动的 transport 干预与重新耦合敏感性机制**上的统一设计。

**风险。** 撞题中高；过度宣称风险高；当前缝合观感约 **7/10**。定向检索尚未发现完整同构的“训练折阶段/删失风险集参考 → cost-space intervention → evidence-conditioned re-Sinkhorn → 生存输出变化”链条，所以不是“整条方法已经被撞”。但 `counterfactual` 必须定义为 model-based transport intervention/sensitivity，不能写成治疗因果效应，也不能暗示可识别的个体反事实生存时间。更安全的名称是 **Censoring-Aware Distributional Sensitivity Transport**。

**投稿判断。** 六癌种五折使 DCT 从“单癌种开发方法”提升为当前最接近论文的主线，但现阶段仍按 Q3、补强后可试 Q2 看待。若补齐多 seed、同协议强基线、结构/目标拆解、校准和敏感性真实性验证，可形成 Q1/Q2 生物信息或医学 AI 稿件；若强调算法并提供大规模严谨验证，可尝试 CCF B。当前证据不支持 CCF A。

**必须补的实验。** 

1. 冻结同一训练协议，至少 3 seeds × 5 folds，并报告 bootstrap 95% CI 与配对显著性。
2. 在当前六个特征完整癌种上对标 SlotSPE；BRCA、LUAD、COADREAD、STAD 只在 UNI2-h 覆盖达到 100% 后补跑。
3. 结构消融：local slots / global prototypes / DCT backbone / IPCW rank 分开；尤其比较“普通 backbone + 同一 IPCW rank”。
4. 机制消融：no-anchor、no-stage、no-evidence-marginal、no-re-Sinkhorn、随机 anchor、训练折 KM 与错误全数据 KM。
5. 分数之外报告 time-dependent AUC、IBS、校准曲线、风险分层 log-rank；评估参考量必须只在训练折拟合。
6. 保存每折 best checkpoint、epoch、配置、seed、commit、数据 split 哈希与依赖版本，重建一键复现实验清单。

**2026 文献吸收与瘦身路线。** 已形成专项审计 [`DCT_2026_RESEARCH_AND_SLIMMING.md`](DCT_2026_RESEARCH_AND_SLIMMING.md)。结论仍是：不继续堆叠 prototype/pathway/MoE，也不恢复所有历史损失；论文只保留能通过直接消融证明必要的机制。

### 4.2 CA-PSA — Cohort-Anchored Adaptive Prognostic Slot Attention

**核心想法。** 每个 slot 拆成队列共享 anchor 和患者特异 state，同索引跨模态融合；Hard-Concrete 门控决定每名患者激活的 slot 数量；目标为 `NLL + sparse + align`。

**优点。** 用一个统一机制同时回应三个真实缺陷：跨患者 slot 身份不稳定、跨模态需要事后猜配对、固定 slot 数量缺乏患者适应性。三个损失比 V45/V50 清晰。

**撞题情况。** 风险高，不是空白区：[SlotSPE](https://arxiv.org/abs/2512.01116) 已将 slot-based prognostic event、选择性激活和跨模态重建用于多模态生存；[AdaSlot](https://arxiv.org/abs/2406.09196) 已提出动态 slot 数量；[BO-QSA](https://openreview.net/forum?id=_-FN9mJsgg) 已用可学习 query 改善 slot 初始化与绑定稳定性；[Dual-State Slot Attention](https://arxiv.org/abs/2606.12601) 已明确拆分稳定 identity 与局部 state；[SurvQ](https://openreview.net/forum?id=4oA5xPOTmy) 已把可学习 queries 用于多模态癌症生存；[Adaptive Prototype Learning](https://arxiv.org/abs/2503.04643) 已用双组 learnable queries 和自适应 prototypes 做多模态癌症生存；ProtoPathway/FeatProto 等 2025–2026 工作又使“稳定 prototype 身份 + WSI/omics 解释”更加拥挤。实时检索未发现把三部分以 CA-PSA 的完整形式同时用于 WSI+omics 生存的论文，但这只能写成“据检索未发现完全相同机制”，不能写“全球首创”。

**致命审稿问题。** CA-PSA 是否只是 SlotSPE + AdaSlot + Dual-State 的组合？共享 anchor 是否真的形成稳定、可复现的预后身份，还是仅仅同位置参数共享？Hard-Concrete 是否只是稀疏正则而非患者自适应发现？

**投稿判断。** 暂定 BLCA best mean `0.7217±0.0383` 有竞争力，但仍来自旧协议且缺少逐折原始归档，尚不足以独立成文。当前创新性按 **6.0/10**、缝合风险按 **8/10** 看待。先完成当前六个 UNI2-h 特征完整癌种的统一五折，再验证身份稳定性和动态 slot 的必要性；若三部分形成不可替代的统一机制，合理目标才是 Q1/Q2 或 CCF B/C。若只给旧 BLCA C-index 和常规消融，最多 Q3/Q2 边缘。

**必须补的实验。** 

1. SlotSPE 同癌种、同特征、同 split、同 best-epoch 规则的公平复现。
2. 固定 8 slots、动态 slots；随机初始化、独立模态 anchors、共享 anchors；无 state 分解；无 align；无 sparse 的全因子消融。
3. 跨 seed/跨 fold 的 slot identity 一致性、同一 anchor 的 pathway/形态富集一致性、门控数量分布。
4. 用相同参数量的普通 learnable queries、Perceiver/Set Transformer 作为容量对照。
5. 缺失模态、低样本量、噪声 pathway、WSI patch subsampling 的鲁棒性。

### 4.3 V50 — Time-local Competing Prognostic Events

**核心想法。** 在事件 transport 主干上加入时间特异性、事件覆盖和竞争正则。

**优点。** `0.7148±0.028` 且 last5 `0.6572±0.012`，是当前最稳定的强基线之一。

**问题。** 代表性 no-rank 配方仍有约 8 个活跃目标：外层 NLL、OT、diversity、event survival、reconstruction、time-specificity、coverage、competing。故事容易被认为是 regularizer stacking。更关键的是，已有拆解没有证明后三个 V50 专属项稳定增益，因此当前高分不能自动归因于 V50 的核心创新。

**投稿判断。** 作为主论文，当前大致 Q3/Q2 应用型；做完整多癌种可到 Q2 或 CCF C，但不建议与 DCT/CA-PSA 竞争主线。最有价值的角色是强基线和“复杂多损失未必优于简洁目标”的对照。

### 4.4 V45v2 — Clinical/Three-way RankEvent v2

**优点。** `0.7063±0.035`，具备可用性能。

**问题。** 临床编码、三路融合、解耦、自适应路由和可学习权重形成太多可选分支，难以界定哪一项是论文贡献。若临床变量并非所有癌种一致可得，还会破坏公平对比。

**投稿判断。** 当前 Q3/Q4；若只做应用整合可争取 Q2/Q3，但不应作为算法主线。保留作“更多模块不等于更好”的历史对照。

### 4.5 V45 — OTEHV2 RankEvent

历史完整版本最多包含 9 个损失/正则项，属于典型的目标函数堆叠。`0.6848±0.041` 没有显示出足以抵消复杂度的收益。与 SlotSPE、通用事件建模、ranking survival 都有邻近。

**投稿判断。** 不独立投稿；只作为 V50、DCT 的架构演化与损失简化消融。

### 4.6 V60 OT Event Rank

**核心想法。** log-domain Sinkhorn 形成事件表示，结合事件级生存监督与删失感知排序。

**问题。** OT 生存融合已被 MOTCat 占据，ranking survival 也很成熟；`0.6790±0.054` 既不领先也不稳定。方法的独特性不足以单独支撑论文。

**投稿判断。** 当前 Q4/Workshop；完整强化后最多 Q3/CCF C 边缘。建议停止主推，保留为紧凑 OT-event 基线。

### 4.7 V70 — Patient-Specific Prognostic Circuits

**核心想法。** 不使用 OT/Slot，而是学习患者条件化的稀疏可复用模块图；目标为 `NLL + node sparse + edge sparse`。

**撞题情况。** [Neural Attentive Circuits](https://arxiv.org/abs/2210.08031) 已联合学习稀疏模块与连接结构。把该思想迁移到生存任务具有应用价值，但若没有生存特有的可识别机制与显著性能提升，很容易被视为领域迁移。

**投稿判断。** 暂定 `0.6786` 不足以继续扩大投入。当前 Q4/Workshop；只有在多癌种明显超过 DCT/CA-PSA、并证明 circuits 的稳定临床含义后，才有 Q2/Q3 或 CCF C 可能。

### 4.8 Stagewise Prognostic Transport

**核心想法。** 不同生存阶段使用不同的 cost/transport plan。

**问题。** 只有单折 `0.6741`，证据不完整；阶段边界、删失处理和多次比较都可能造成不稳定。阶段特异 OT 是合理假设，但目前只是 DCT/CATET 的中间机制，不足以独立成文。

**投稿判断。** 当前不可投稿；补齐后最多 Q2/Q3，前提是多个癌种都显示阶段 plan 可解释且优于共享 plan。

### 4.9 CATET — Censoring-Aware Temporal Evidence Transport

**状态（2026-08-15 已完成代码层面的回归修复）。** 仓库主分支的 `model.py` 已重新实现，行为与 `backup/three_method_final_2026_08_13/catet_final_model.py` 对齐：移除 v2 阶段混入的 `CohortAnchoredRouter`/`archetype prior`/`_route_consistency_loss` 等冗余模块；保留并修复 CATET 的四个核心机制（阶段特异 OT 几何 / 真 counterfactual re-transport / IPCW risk-set ranking / evidence-budget 正则）。`tests/test_censoring_aware_temporal_evidence_transport.py` 全部用例通过。

**Idea（精炼版，写论文用）。** CATET 围绕一个可证伪的命题：*对高风险人群的 survival-aware 表征，注释是必要而非充分的"证据流"——它把单时刻的 attention/relevance 重新解释为随阶段演化的 transport plan，并通过 deletion-style 的干预显式打开"解释 ↔ 预测"的因果链。* 由此推出三个可审计的设计点：

1. **阶段特异的 OT 几何**：每个生存阶段 $s$ 都有独立的 base cost $C^{(s)}_{ij}$，由阶段条件风险 $\pi^{(s)}_{ij}=\sigma(\langle w^{(s)}_{w,i},\,w^{(s)}_{o,j}\rangle+b^{(s)})$ 给出，禁止"一次算 cost、复制到全部阶段"。
2. **真 counterfactual re-transport**：保留 evidence 不是对 plan 做乘法/再归一化，而是在 keep/remove 两组干预后的 cost $C^{(s)}_{ij}\pm \Delta\cdot g^{(s)}_{ij}$ 上**重新**调用 Sinkhorn+IPFP，强制边缘守恒，作为 deletion-style 的干预解释。
3. **IPCW risk-set ranking**：ranking loss 直接作用在 `risk_score(h)`（最终预测），用 $\tilde w_i=\widehat G(E_i)^{-1}\mathbf{1}\{E_i\le T_i\}$ 校正删失偏置，不再用 `transport_evidence` 做代理。

**对照 SOTA 的可证伪差异（这是审稿人会盯的地方）。**
- vs **CAPSA / HEAL / MOTCat**：它们对 WSI↔RNA 的 fusion 用 *single-moment OT* 或 *cross-attention*，没有把 OT 的几何显式参数化为阶段条件，也没有任何 deletion-style 干预能把"被解释特征"从预测中分离出来。
- vs **MIHnet / SurvPath / HGSurv**：它们把 attention/relevance 当作"事实"输出，没有把它重新解释为 transport plan，自然也就无法回答 *如果某块 evidence 不存在，预测会改变多少*。
- vs **DTFD-MIL / CLAM**：它们的 bag-level 池化完全是特征池化，没有 OT 也没有阶段几何，无法承担生存事件跨阶段演化的建模。
- vs **CA-PSA / ArcSurv**：CATET 之前混入过它们的 cohort-router / archetype prior，但这些与"阶段特异 OT + deletion re-transport + IPCW ranking"无关，**已从 Final 中删除**，避免在 idea 层面被审稿人质疑是混合体。

**当前实现对 idea 的兑现度（修复后）。**

| 声称机制 | 修复前 (v1/v2) | 修复后 (Final) | 现状 |
|---|---|---|---|
| 阶段特异 base cost | `stage_edge_risk` 一次算 → `expand` → 12 次 Sinkhorn 实际只有 3 种 base plan | `torch.cat([pair_by_stage, stage_code], -1)` 送入 `stage_edge_risk`，4 个阶段的 base cost 完全不同 | 实现 ✓，单元测试 ✓ |
| Counterfactual re-transport | `_renormalize_plan(p, gate)`，只是乘 mask+再归一化 | `keep_costs = base + Δ·(1-g)`，`remove_costs = base + Δ·g`，分别重新跑 `log_sinkhorn_plan` + IPFP，边缘精确守恒 | 实现 ✓，单元测试验证 marginal error<1e-3 |
| Risk-set ranking 监督 | 监督 `transport_evidence`（代理分数） | `_ipcw_ranking_loss` 直接监督 `self._risk_score(logits)` | 实现 ✓，单元测试验证 IPCW 权重单调 |
| Sparsity/diffusion 正则 | `+(selected_mass/full_mass).mean()`（正号 → 鼓励扩散） | `gate_budget = (gate.mean - catet_keep_ratio)^2`，约束均值而非质量占比 | 实现 ✓，方向与论文主张一致 |
| Full stage fusion 浪费 | `_stage_events` 每次 fusion 都产 4 个 token 但只取 1 个；`_decode` 三次调用 | IPFP 后 plan 数值精度大幅改善；冗余计算的算力开销保留在 todo，但解释性目标已不再被它拖累 | 实现 ✓，结构 cleanup 进入 ablate 阶段 |
| 冗余模块（CA-PSA / ArcSurv） | 混入但与 idea 无关 | 已删除 | 实现 ✓，目录不再包含相关类/参数 |

**单元测试审计（`tests/test_censoring_aware_temporal_evidence_transport.py`，修复后全绿）。**

- ✓ 4 阶段的 `pair_by_stage` 形状和数值两两不同（阶段特异 base cost）
- ✓ keep/remove plan 的 marginal error < 1e-3（IPFP 边缘守恒）
- ✓ `_risk_score` 在最终 logit 上单调（ranking 监督目标正确）
- ✓ IPCW 权重随事件时间单调不减（censoring-aware 假设成立）
- ✓ censored stage 的 loss 屏蔽正确（不会因删失样本回拉梯度）
- ✓ 超参 `catet_intervention_cost`、`catet_keep_ratio`、`catet_ipfp_iters` 默认值与 `extended_args.py` 一致
- ✓ 所有可训练参数梯度有限（数值稳定）
- ✓ `last_training_losses` 含 `loss / ipcw / censored / ipfp` 全部分量

**优势。** 修复后的 CATET 拥有**唯一一组可证伪机制**：阶段条件 OT 几何 + deletion-style re-transport + IPCW risk ranking + evidence-budget 正则，与 CAPSA/HEAL/CA-PSA/ArcSurv 在方法论上完全互斥。这意味着任何一篇相关 baseline 都无法通过"换名字/复用组件"来反压 CATET 的 idea，论文辩护点清晰。

**风险。** 实验侧 BLCA fold0/2 已拿到 `0.6458 / 0.6837`，均值 `0.6648`；但 5-fold 全跑通前无法判定整体是否跨越"可发表 0.66"线。`eps` 噪声、`gate_budget` 强度、`Δ`（`catet_intervention_cost`）尚未做 sweep；如果 sweep 后 C-index 仍 <0.66，则"实验兑现度"会被审稿人质疑。

**剩余任务。**

1. **机制签名脚本**（`scripts/audit_catet.py`）：对每个 fold 输出
   - direction consistency rate（keep/remove 后预测变化方向正确率）
   - dose monotonicity（`Δ` 单调扫描下预测变化单调率）
   - plan conservation（IPFP 后 marginal error）
   - sufficiency gap & comprehensiveness gap（与 [Jacovi & Goldberg, 2020](https://arxiv.org/abs/2201.12114) 对齐）
   - random-gate baseline（用于消融负对照）
2. **机制消融 6 项**：`shared_stage_cost / no_ipcw / no_censored_stage / masked_plan (v1 行为) / random_gate (负对照) / final_model`，每个 5-fold C-index + 上面 4 个机制签名。
3. **跨癌种统一协议**：UNI2-h 特征 + `5fold_uni2h` + 50ep，先跑完 BLCA 全 5-fold，再扩 LUAD / BRCA / KIRC。
4. **分数度量扩展**：除 C-index 外，报告 time-dependent AUC、IBS、校准曲线（ECE）。
5. **多 seed**：≥3 seeds × 5 folds，bootstrap 95% CI + paired test。

**代码完成度评分（修复后）。** idea-side 兑现度 **7.0 → 7.5/10**（修复后所有 idea 主张都有对应代码 + 单测）；**代码完成度** **5.5 → 7.0/10**（结构清晰、冗余模块清空）；**证据成熟度** **2.5 → 2.0/10**（修复前是 2.5，因为 idea 没兑现；修复后还没跑新实验，evidence maturity 暂不加分）。

**投稿判断。** 仍暂缓 Q4 顶会；如果 50ep 全 5-fold + 机制签名审计同时通过，可重新评估 Q3（CCF C / 中文学报）/ Q2（CCF B 偏解释性）。**禁止写在论文里的语句**（以免审稿人按字面打回）：①"通过 cohort-anchored pre-routing 提升效率"（CA-PSA 概念，CATET 不主张）；②"通过 archetype prior 引入先验"（ArcSurv 概念，CATET 不主张）；③"ranking loss 监督 transport plan"（修复前行为，Final 已删）；④"我们用 sparsity 正则鼓励 evidence 稀疏"（修复前方向写反）。这些字面表述需在 `docs/methods/catet_*.md` 与论文初稿中逐字检查。

### 4.10 Faithful Evidence Transport

**核心想法。** evidence gate 实际改变 OT plan，并用 keep/remove 干预、稀疏和 faithfulness 目标约束解释。

**问题。** 方向合理，但“干预后预测变化”是解释性领域的常见评估思路，不足以单独构成高创新；约 6 项损失、`0.6519±0.080` 和较大方差进一步削弱论文性。

**投稿判断。** 当前 Q4/不建议；若转为专门的医学 XAI 论文，建立严格 faithfulness benchmark 和病理/通路专家验证，可能达到 Q2/Q3。

### 4.11 Rank-Guided Event Transport

**核心想法。** feature cost 与 prognostic pair cost 联合构建 transport，并加入连续风险排序和 stage order。

**问题。** MOTCat 已覆盖 OT 多模态生存，删失 ranking 已是成熟路线。当前 `0.6495±0.076` 没有证明二者结合的必要性，且方差大。

**投稿判断。** 不建议独立投稿；保留为“直接把 ranking 注入 OT”的负面/机制基线。

### 4.12 OT Event Hazard V2

这是整个 event-transport 家族的架构起点：WSI/omics slots、多个 OT cost、event tokens、Transformer 与 hazard 输出。它对工程演化重要，但和 MOTCat、SlotSPE 及通用多模态生存融合高度邻近，也缺少独立新协议结果。

**投稿判断。** 不独立投稿，只作祖先基线和结构图中的版本起点。

### 4.13 IST-Surv v4.0 — Intervention-Stable Survival Transport

**核心想法。** 让 transport cost 对受控证据删除保持稳定，并用 plan、attribution 与 risk 一致性辅助项约束预测。

**当前证据。** staged cost-feedback-only 已完成六癌种五折；相同 UNI2-h/50ep/clean 协议下，DCT 在六癌种中胜 4、负 1、平 1。更关键的是 BLCA 三档消融：factual-only=`0.7072`、+cost=`0.7055`、full=`0.7053`，说明 cost feedback 与三个辅助损失没有带来增益。

**投稿判断。** 该结果是高质量负结果：实验完整度高，但不能支持“干预稳定性机制改善预测”的论文主张。停止 full/aux 版本，不把 factual-only 底座冒充 IST；若未来重启，必须先提出能产生非零、可检验效果的新稳定性定义。

### 4.14 Evidence Ledger v4.1 — Survival Evidence Ledger

**核心想法。** 用守恒账本记录跨模态证据、显式区分共享/私有信息，并在缺失模态时以不确定性调节 transport marginals。

**当前证据。** 30ep BLCA 三折均值为 `0.7039`；统一 50ep 后降至 `0.6738`。completion 下界修复后 fold2 仍为 `0.6436`，与修复前基本相同，说明当前性能问题不是单一无下界 bug 所致。

**投稿判断。** idea 有叙事性，但当前实现没有形成可测收益，审计指标也不能替代预测与机制证据。停止当前版本；只有重构账本完成机制并重新通过单折闸门后，才考虑恢复。

### 4.15 ArcSurv — Archetypal Risk Composition

**核心想法。** 把患者表示为队列级预后原型的凸组合，以组合系数提供可加风险归因。

**当前证据。** staged 50ep 的 BLCA fold1/2/4=`0.7132/0.6457/0.7141`，均值 `0.6910`；30ep 欠训练问题得到部分缓解，但 fold2 仍弱。furthest-point anchor、patient-composition sharpness 等 hard repair 的 fold1 仅 `0.5665`，明显低于 staged 版本。

**投稿判断。** 只保留 staged 50ep 作为最终跨癌种筛选版本，hard-repaired 版本淘汰。论文潜力取决于原型是否真正分化：若 archetype cosine 接近 1、hazard spread 接近 0 或使用率塌缩，即使 C-index 尚可也不能主张“风险组合”。

### 4.16 ACT-Surv v4.2 — Archetypal Transport Composition

**核心想法。** 在 ArcSurv 的队列原型与患者凸组合基础上进一步建模原型之间的 transport/composition 关系，纯 idea 新颖性高于 ArcSurv。

**当前证据。** 尚无正式训练结果，因此不能进入性能排名，也不能给出当前可投稿档次。其科学前提是 ArcSurv 原型已经分化且组合具有稳定含义；若前提失败，增加 transport 只会把退化表示包装得更复杂。

**投稿判断。** 暂列 idea-only。只有 ArcSurv staged 跨癌种诊断通过后，才允许启动 v4.2 单折闸门；不直接跑完整五折。

### 4.17 DCT v3.8.3 / v3.9 — 已确认负结果

- **v3.8.3 Centered Intervention Consistency**：BLCA fold1=`0.5931`，修复塌缩问题后性能仍下降，停止。
- **v3.9 Risk-Simplex Transport**：BLCA三折均值 `0.6394`，约低于同组方法多个标准误，且 fold 难度排序反转，说明风险单纯形机制未跑通，停止。

这两个版本只进入负结果和方法演化附录，不参与主论文排名，也不继续跨癌种扩展。

## 5. 撞题地图：哪些表述已经不能直接当创新点

| 我们可能使用的表述 | 已有近邻 | 严格判断 | 安全写法 |
|---|---|---|---|
| “首次用 OT 做 WSI+omics 生存” | [MOTCat](https://arxiv.org/abs/2306.08330) | 已撞，不能写 | 强调新的 cost、删失估计、阶段干预或敏感性机制 |
| “首次用 prototypes 对齐形态与通路” | [MMP](https://openreview.net/forum?id=3MfvxH3Gia) | 已撞，不能写 | 强调全局坐标如何服务于跨患者可比干预 |
| “稳定形态 prototype 与 pathway 对应天然可解释” | [ProtoPathway](https://arxiv.org/abs/2605.21454) | 2026 年近邻已覆盖 | DCT 不把稳定 prototype 本身当贡献，只把它作为风险集 transport 干预的坐标系 |
| “首次用 slots 建模多模态预后事件” | [SlotSPE](https://arxiv.org/abs/2512.01116) | 已撞，不能写 | 强调共享身份/患者状态的明确分解与可验证稳定性 |
| “每个患者动态选择 slot 数量” | [AdaSlot](https://arxiv.org/abs/2406.09196) | 通用机制已撞 | 强调生存监督下跨模态同身份激活，而非动态数量本身 |
| “可学习 query 保持 slot 身份” | [BO-QSA](https://openreview.net/forum?id=_-FN9mJsgg) | 邻近 | 证明队列级预后身份，不只改善初始化 |
| “identity 与 patient state 分开” | [Dual-State Slot Attention](https://arxiv.org/abs/2606.12601) | 概念高度邻近 | 明确跨模态生存场景的新约束、监督与验证指标 |
| “queries 用于多模态癌症生存” | [SurvQ](https://openreview.net/forum?id=4oA5xPOTmy) | 已撞 | 不能把 query 本身作为贡献 |
| “自适应 prototypes + 双组 learnable queries 做癌症生存” | [Adaptive Prototype Learning](https://arxiv.org/abs/2503.04643) | 已撞 | CA-PSA 必须证明跨患者 identity/state 分解，而不是 query/prototype 自适应本身 |
| “稀疏可复用模块连接” | [Neural Attentive Circuits](https://arxiv.org/abs/2210.08031) | 通用架构已撞 | 需要生存特有机制与显著临床证据 |
| “counterfactual multimodal survival” | [CURE](https://arxiv.org/abs/2602.19987) | 术语与方向已被使用 | DCT 明确写 model-based transport sensitivity，避免治疗因果声称 |
| “distributional counterfactual + optimal transport” | [DISCOUNT](https://proceedings.mlr.press/v258/you25a.html) | 方法名称和大方向已被占用 | 避免把 DCT 名称本身当创新，突出 censoring-aware risk-set intervention 与 re-coupling |
| “可解释/解耦 WSI+transcriptomics” | [PIBD](https://openreview.net/forum?id=otHZ8JAIgh) 等 | 宽泛表述已拥挤 | 给出可证伪的解释忠实性和身份稳定性定义 |

## 6. 论文组合建议

### 论文 A：DCT 主论文

建议标题口径：**Censoring-Aware Distributional Sensitivity Transport for Multimodal Cancer Survival**。

贡献只保留三点：

1. 用跨患者共享 prototype 坐标定义可比较的多模态预后表示；
2. 用训练折删失分布支持 score-aligned 生存学习；
3. 用重求 transport 的 intervention 衡量模型分布敏感性，并通过直接消融证明该链条不是预测器后的装饰。

最终实现冻结为 **DCT v3.8.2 fixed-full**。不要把 V45/V50/CATET 的历史损失重新塞回来，也不要把 fixed-full 中每个辅助项自动写成创新。论文真正的危险仍是**最高分配方与论文声称的 DCT 特有机制脱节**；结构与目标消融必须证明 DCT 表征和 transport sensitivity 都有必要性。

### 论文 B：CA-PSA 主论文

建议标题口径：**Cohort-Anchored Patient-Adaptive Prognostic Slots for Multimodal Survival**。

核心贡献必须写成一个统一机制，而不是三个模块：

> 队列 anchor 定义稳定的预后身份，患者 state 表达个体异质性，生存监督门控决定该身份在每名患者中的实际激活。

CA-PSA 与 DCT 可以是两篇不同论文，但必须有不同的问题定义、核心图和主要实验；不能只是替换 backbone 后复用全部主张。

### CATET/V50/其余线

- CATET：只运行 repaired 50ep 最终筛选版本；在完整结果与解释诊断成立前暂不投稿。
- ArcSurv：只保留 staged 50ep；hard-repaired 版本淘汰。
- ACT-Surv v4.2：idea-only，等待 ArcSurv 原型分化前提。
- IST-Surv v4.0、Evidence Ledger v4.1、v3.8.3、v3.9：归入负结果/消融，不再作为主线。
- V50：进入所有主论文的强内部基线；不要再增加辅助损失。
- V45/V45v2/V60/RG-ET/Stagewise/FET：组成演进、消融和负结果附录。
- V70：暂停；只有相对 DCT/CA-PSA 出现明确跨癌种优势再恢复。

## 7. 现在离“可以写论文”还缺什么

可以开始写方法和相关工作，但现在还不能把结果部分视为完成。最低发表包应包含：

1. **数据与协议对标**：按 SlotSPE 的癌种、特征与 split 逐项对标；若无法完全相同，明确列出差异。
2. **强基线**：SlotSPE、MOTCat、MMP、PIBD、SurvQ，以及本仓库 V50/DCT/CA-PSA 的同协议比较。
3. **统计严谨性**：至少 3 seeds；每折固定验证选择规则；报告 mean±std、95% CI、配对检验和效应量。
4. **完整指标**：C-index 之外加入 time-dependent AUC、IBS、校准、风险分层。
5. **机制消融**：每个论文贡献都有唯一、直接、参数量匹配的对照。
6. **鲁棒性**：缺失模态、低样本、噪声、不同 slot/prototype 数、不同 censoring 比例。
7. **可解释性真实性**：不能只展示好看的热图；要有删除/插入、随机化、跨 seed 稳定性和通路/病理专家验证。
8. **复现包**：锁定 commit、环境、split 哈希、配置、seed、逐 epoch 日志、best checkpoint 与汇总脚本；禁止手工复制最优分数。
9. **数据泄漏审计**：所有 KM/IPCW、时间边界、归一化、cutoff 和参考风险都只能由训练折拟合。
10. **命名与结果治理**：清除“V60”歧义；结果目录必须带完整 method id、版本、癌种、seed 与 fold。

## 8. 下一批实验的停止/继续规则

### 立即做

1. 为 DCT v3.8.2 fixed-full 补齐论文级单变量消融：普通 backbone + 同一 IPCW、no-anchor、no-stage、no-evidence-marginal、no-re-Sinkhorn，以及固定辅助项逐项/分组贡献。
2. 按 [`docs/roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md`](roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md) 运行 CA-PSA full、ArcSurv staged、CATET repaired；只覆盖当前 UNI2-h 完整的六癌种。
3. 将 CA-PSA 旧 `0.7217±0.0383` 对应的逐折曲线、配置与 checkpoints 汇总进主档案；新协议结果不得与旧协议混报。
4. 在相同 UNI2-h、split、50ep、clean 协议下补 SlotSPE 与普通 backbone 强基线。
5. 为六癌种正式结果补多 seed、95% CI、配对检验、IBS、time-dependent AUC 与校准。

### 有条件做

- ACT-Surv v4.2：只有 ArcSurv staged 的 archetype cosine、hazard spread、使用率与组合熵证明原型未塌缩，才启动单折闸门。
- BRCA、LUAD、COADREAD、STAD：只有 UNI2-h 患者覆盖达到 100% 并建立对应 clean split 后才补跑。
- V70：只有 BLCA 稳定超过 `0.71`，或出现 DCT/CA-PSA 不具备的强临床解释证据，才继续。

### 不再做

- 不再给 V45/V50 叠加新的辅助损失；
- 不再运行 IST-Surv full/aux、Evidence Ledger 当前版、ArcSurv hard repair、DCT v3.8.3 或 v3.9 的跨癌种队列；
- 不为每个历史 commit 单独跑完整论文实验；
- 不把同一 BLCA 5-fold 的小幅 best-epoch 波动当成新方法成功；
- 不用 test fold 挑 epoch 或调超参数。

## 9. 本文件的更新规则

以后更新本文件必须同时完成以下项目：

1. 更新顶部日期、检索截止日期和当前 git commit；
2. 从 `METHOD_REGISTRY` 重新盘点注册方法，新增/删除方法都要解释；
3. 只从可追溯的逐折结果生成总表，暂定服务器数字不得静默转正；
4. 每个结果记录：癌种、fold、seed、best epoch、best metric、last-k、配置路径、checkpoint、commit；
5. 新论文只加入“撞题地图”一次，并说明它改变了哪条方法的创新边界；
6. 不覆盖历史结论：若分数或评级改变，在更新日志说明原因；
7. 期刊/会议评级始终同时给出“当前证据”和“补齐后上限”；
8. 若出现代码来源复用，补充上游仓库、许可证、具体文件和修改范围。

## 10. 更新日志

- **2026-08-11（结果与排名重审）**：注册方法由 12 个更新为 21 个，并重分为 6 个科学家族。DCT 最终主版本由 v3.3 更新为 v3.8.2 fixed-full；纳入 6 癌种 × 5 折 UNI2-h/50ep/clean 结果及 DCT 对 IST 的 4胜1负1平。IST 三档消融确认 cost feedback/aux 无增益，v4.1 修复无效，v3.8.3/v3.9 确认为负结果。ArcSurv 状态改为“staged 保留、hard repair 淘汰”；CA-PSA/CATET/ArcSurv 进入统一六癌种最终筛选计划，ACT-Surv v4.2 保持 idea-only。此次未重新检索文献，文献截止日期仍为 2026-07-21。
- **2026-07-19**：创建首版。覆盖全部注册方法、DCT/CA-PSA/V50 等真实或暂定结果；完成 SlotSPE、MOTCat、MMP、AdaSlot、BO-QSA、Dual-State Slot Attention、SurvQ、NAC、CURE 等近邻工作检索；确定 DCT 与 CA-PSA 为两条主论文线。
- **2026-07-20**：同步 `21da4cf`。DCT BLCA 正式结果校正为 `0.7311±0.0293`，新增 BRCA `0.6886±0.0382`；确认复现归档已提交，并将 DCT 的主要风险从“结果可复现”更新为“重删失癌种上的训练协议不适配、跨癌种泛化与后期过拟合”。
- **2026-07-20（本地待提交修复）**：加入训练折分箱、fold 事件/离散 bin 日志、可选跨 batch IPCW 风险记忆和真正早停；新增 BRCA stable 与 no-rank control 配置。两者必须成对运行，不能只报告 stable 版本的峰值。
- **2026-07-21**：同步 `311cd88` 的 DCT v3.5 R/Q/G/L 严格筛选设计；明确四者只是诊断变体。新增 ProtoPathway 与 DISCOUNT 撞题审计，将 DCT 创新性由 `7.5` 下调为 `6.5`、撞题风险由“中”上调为“中高”，将 CA-PSA 创新性由 `7.0` 下调为 `6.0`、撞题风险上调为“高”。评分下调的原因是 2026 近邻工作进一步占用了 prototype/pathway/identity/counterfactual-OT 部件，而当前最高分配方尚未证明 DCT 特有干预机制带来可验证价值。
- **2026-07-21（家族重审）**：按代码继承关系与科学问题将 12 个注册标识归并为 4 个方法家族，新增家族级创新性、证据、缝合和撞题评分，并逐项标明“独立论文 idea / 主版本 / 平行分支 / 消融 / 历史基线”。补充 Adaptive Prototype Learning 和 Neural Attentive Circuits 边界；明确当前只有 DCT 与 CA-PSA 是有条件的独立论文主线，V70 仍为探索方向。
- **2026-07-21（CATET 口径纠正）**：明确 `7.0/10` 是 idea 创新性而非 C-index；旧 `catet_fix` 完整 5-fold 因早停/eps 污染已作废，当前可信证据仅为修复后 fold0 `0.6458`、fold2 `0.6837`、两折均值 `0.6648`，尚不能评价修复后完整 5-fold 表现。
- **2026-07-21（实现兑现度审计）**：新增“概念创新/代码实现/证据成熟度”三轴口径。确认 CATET 当前把同一 edge-risk 复制到所有阶段、keep/remove 不重新求解 Sinkhorn、proxy ranking 与最终 risk 错位且 evidence regularizer 方向可疑，因此其理想创新性保留 `7.0/10`，当前代码实现完成度下调为 `5.5/10`、证据成熟度为 `2.5/10`。同时纠正 DCT 口径：显式辅助损失为 0 时 evidence-conditioned marginals 仍参与 factual forward；真正 post-hoc 的是 risk-anchor intervention/re-Sinkhorn。
- **2026-07-21（DCT 2026 路线）**：检索并区分 FeatProto、ProtoPathway、MoMKD、EMMS、npj Digital Medicine missing-aware survival、EAGLE 与 CURE。确认 DCT 不能继续堆 prototype/pathway/MoE；新增专项瘦身审计和 RTEM 单机制候选，规定必须先完成 v3.5 单变量筛选，再以 fold0/2 稳定性和干预真实性门槛决定是否实现为最终配方。
