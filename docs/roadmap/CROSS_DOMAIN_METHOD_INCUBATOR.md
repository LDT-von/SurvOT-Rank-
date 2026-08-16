# 跨领域新方法孵化路线图

> 状态：候选研究计划，尚未实现  
> 建立日期：2026-08-04  
> 研究范围：WSI + 组学多模态癌症生存预测

## 1. 目标与边界

本计划寻找已经在计算机视觉、信息论、生成建模、强化学习、图学习或统计推断中证明价值，但尚未在“WSI + 组学多模态癌症生存预测”中得到充分开发的机制。

目标不是把六种机制同时塞入 DCT，也不是简单替换 SlotSPE 的槽模块。每条路线必须作为独立候选方法开发，通过相同数据、split、训练和评价协议进行筛选，最终只保留一至两条能够形成完整论文问题的主线。

“尚未应用”在本文中表示截至 2026-08-04 的针对性检索未发现直接工作，不等于对全部文献作绝对否定。正式立项前仍需进行系统文献复核。

## 2. 候选路线与优先级

### 2.1 第一批候选

评分均为 5 分制；实现难度越高表示风险越大。

| 优先级 | 候选方法 | 新颖性 | 任务适配性 | 实现难度 | 提分潜力 | 论文叙事 |
|---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 生存感知多模态信息分解 | 4.5 | 5.0 | 3.0 | 4.5 | 5.0 |
| 2 | GFlowNet 多样化风险证据集合 | 5.0 | 4.5 | 4.5 | 4.0 | 5.0 |
| 3 | Flow Matching 概率跨模态生成 | 4.5 → **2.5** | 4.5 | 4.0 | 4.0 | 4.5 → **3.0** |
| 4 | Neural Sheaf 异质证据传播 | 5.0 | 4.0 | 5.0 | 3.5 | 4.5 |
| 5 | Deep Equilibrium 平衡态融合 | 4.0 | 3.5 | 4.0 | 3.0 | 3.5 |
| 6 | Conformal Survival 校准预测 | 3.0 | 4.0 | 2.5 | 1.5 | 4.0 |

路线三的评分下调依据见 2.2 与 6.0。

### 2.2 新颖性审计（检索日 2026-08-04）

第 1 节要求"正式立项前进行系统文献复核"。本节是该复核的第一轮结果，用途是**避免把已有工作误判为创新**。

以下机制在本方向已被占用，不应作为新颖性主张：

| 机制 | 已有工作 | 结论 |
|---|---|---|
| 最优传输 / Sinkhorn | MOTCat、OTSurv | 占用（本项目自身基线即在此赛道） |
| 状态空间模型 Mamba | SurvMamba；MDCS-MoAME, CVPR 2026 | 占用 |
| 混合专家 MoE | MoME, MICCAI 2024；Region-Graph OT Routing MoE | 占用 |
| 双曲几何 / 层级嵌入 | Multimodal Survival Prediction with Pathology Reports in Hyperbolic Space | 占用，且其卖点为层级蕴含 + 一对多关系 |
| 扩散 / VAE 补缺失模态 | CLDVAE (arXiv 2503.09496)；Nat Commun 2025 跨模态生成基因表达 | 占用 |
| **Flow Matching 做 histology → 组学** | **STFlow (arXiv 2506.05361)；HoloTea (arXiv 2511.14613)** | **占用，直接影响路线三** |
| 证据深度学习不确定性 | Dual-Prototype Evidential Fusion (arXiv 2510.00053) | 占用 |
| 因果后门 / 前门去混淆 MIL | IBMIL, CVPR 2023；front-door MIL (arXiv 2607.12376) | WSI 分类已占用，生存场景仅剩残余空间 |
| 稀疏自编码器可解释性 | Pathology FM + SAE (arXiv 2407.10785) | 半占用 |
| 概念瓶颈 | ConcepPath；Concept-Guided Multimodal MoE | 占用 |

对本文件既有六条路线的审计结论：

| 路线 | 审计结论 |
|---|---|
| 一 信息分解 | **通过**。本方向的"交互建模"工作（MoME、Holistic Multimodal Interactions, arXiv 2507.04891）全部是注意力启发式，没有一篇做信息论分解，也没有一篇把分解量当训练信号。差异化点必须落在"生存特异 + 删失感知"，见 4.6 |
| 二 GFlowNet | **通过**，且是全场新颖性最高项。病理方向检索无命中 |
| 三 Flow Matching | **不通过（重大下调）**。见 6.0 |
| 四 Neural Sheaf | **通过**。病理生存、WSI + 组学方向零使用 |
| 五 DEQ | **通过但动机弱**。占用情况干净，问题是 8.4 已指出的"只是更深的融合层" |
| 六 Conformal | **部分通过**。病理侧只做过分类的保形不确定性（Nat Commun 2022），生存 + WSI + 组学是空白；但它本质是后处理包装，9.4 的定位判断正确 |

本节结论不等于对全部文献的绝对否定。每条路线在正式立项前仍需针对该路线单独复检一次。

## 3. 阶段 0：统一实验协议

任何候选方法开始真实训练前，必须先完成以下工作。

### 3.1 数据协议

- 只纳入 WSI 特征、组学和生存标签均有效的患者。
- 明确 UNI 与 UNI2-h 特征的选择规则，禁止训练时静默使用零特征代替缺失 WSI。
- 所有方法使用完全相同的患者集合和患者级五折 split。
- 每折记录训练/验证患者数、事件数、删失率和特征覆盖率。
- 在特征未补齐的癌种上禁止开展正式比较。

### 3.2 统一基线

选定一个结构简单、已通过数值检查的多模态生存基线，统一保留：

- WSI encoder；
- pathway-level omics encoder；
- 基础多模态融合；
- 离散时间生存 NLL；
- 经验证后才启用的 IPCW 排序项。

每个候选方法只替换或增加其核心机制，不同时引入其他候选机制。

### 3.3 统一评价

主要指标：

- 五折 Harrell C-index 均值和标准差；
- IPCW C-index；
- Integrated Brier Score；
- time-dependent AUC；
- 生存概率校准误差。

补充指标：

- 多随机种子稳定性；
- 最佳 epoch 分布；
- 缺失模态性能；
- 参数量、显存和训练时间；
- 患者级解释稳定性。

## 4. 路线一：生存感知多模态信息分解

### 4.1 科学问题

现有方法集中研究“如何融合或对齐 WSI 与组学”，但没有明确区分患者风险来自：

- 两个模态共有的冗余信息；
- WSI 独有信息；
- 组学独有信息；
- 只有组合两个模态后才出现的协同信息。

### 4.2 初步模型

将病理表示和组学表示分解为：

\[
z_{\mathrm{red}},\quad z_{\mathrm{wsi}},\quad
z_{\mathrm{omic}},\quad z_{\mathrm{syn}}.
\]

四类成分分别产生时间相关风险，再由患者级、时间相关门控形成最终风险：

\[
h(t\mid x)=\sum_k \alpha_k(x,t)h_k(t).
\]

必须在通用信息分解基础上增加生存特异设计：删失感知的分解目标、时间相关协同风险、IPCW 监督和患者级风险来源解释。不能直接复制通用 DMIL 架构。

### 4.3 必要消融

- 仅冗余信息；
- 冗余 + 两个独有成分；
- 去掉协同成分；
- 固定权重与患者自适应权重；
- 静态权重与时间相关权重；
- 普通信息分解与删失感知信息分解。

### 4.4 Go/No-Go 条件

只有满足以下条件才进入完整五折：

- 至少一个筛选癌种稳定超过统一基线；
- 四个成分没有塌缩到同一表示；
- 协同成分在消融中提供额外预后价值；
- 改进不依赖某个癌种的人工专属参数。

### 4.5 起点文献与代码

- DMIL, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.pdf>（arXiv: <https://arxiv.org/abs/2606.11614>）
- ~~代码：`github.com/GeWu-Lab/DMIL`~~ — **该仓库不存在**。论文声明了这个地址，2026-08-04 实测 Repository not found，代码尚未发布。
- **实际可用起点**：同组已发布的 `GeWu-Lab/LSMI_Estimator`，对应 Efficient Quantification of Multimodal Interaction at Sample Level, ICML 2025 <https://proceedings.mlr.press/v267/yang25aj.html>。它做的是**样本级**冗余/独有/协同估计，正好对上 4.6 指出的估计器方差瓶颈，且样本级输出可直接做患者级门控信号。注意该仓库**无许可文件**，只读算法、自己重写。
- 前身工作：Towards Holistic Multimodal Interaction <https://openreview.net/forum?id=BZWssJoYEv>

理论源头（必须引用，否则会被指为只跟一篇 CVPR）：

- Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework, NeurIPS 2023: <https://openreview.net/forum?id=J1gBijopla>
  将 \(I(X_1,X_2;Y)\) 分解为冗余 R、唯一 U_1/U_2、协同 S，并给出可扩展到连续高维的估计器。
- PID 的公理化基础（Williams-Beer / Bertschinger 系）作为背景引用。

### 4.6 生存特异差异化（本路线成立的前提）

审计结论指出，本路线的新颖性不在"用信息分解"，而在"把信息分解接到删失生存监督上"。三个必须自己解决、通用 DMIL 不提供的技术点：

**离散标签的天然契合。** PID 估计器要求 \(Y\) 离散，而离散时间生存标签本身就是区间索引 \(j\in\{1,\dots,M\}\)。这是本任务相对通用多模态分类任务的结构性便利，应在论文中明确指出，不要当作巧合掩盖。

**删失样本不能直接进估计器。** 删失患者的 \(j\) 是区间截断的。两条可行路线：

- IPCW 加权：用 Kaplan-Meier 估计删失分布 \(\hat G\)，以 \(1/\hat G(t_i)\) 加权，只让完整观测进入 PID 估计。**第一版走这条**，因为它是标准做法且与既有 IPCW 排序项的口径一致。
- 集合值标签：把删失样本视为 \(Y\in\{j,\dots,M\}\) 的集合值标签，用集合值互信息的下界。作为备选，不在第一版实现。

**估计位置。** PID 估计只在低维池化表征上做，禁止在 patch 级或全 pathway 级做。高维连续下 PID 估计方差极大，必要时先聚类离散化。此项应写入 4.3 的数值诊断。

**可写进正文的产物。** 泛癌 R/U/S 谱本身就是一张主图。它同时为"某些癌种加组学无收益"提供可测量的解释——目前该现象在本方向无人给出定量答案。

## 5. 路线二：GFlowNet 多样化风险证据集合

### 5.1 科学问题

同一患者可能存在多组不同但同样合理的“病理区域 + 分子通路”致险路径。普通注意力、Top-K 或单一槽解释通常只能返回一个结果，并且可能随随机种子变化。

### 5.2 初步模型

把患者解释定义为证据集合：

\[
E=\{\text{patch cluster},\text{pathway},\text{genomic event}\}.
\]

GFlowNet 通过逐步添加证据，学习采样多个高价值集合。训练奖励只允许由训练数据构建：

\[
R(E)=-\mathcal L_{\mathrm{surv}}(E)
-\lambda_1|E|+\lambda_2R_{\mathrm{stability}}
+\lambda_3R_{\mathrm{diversity}}.
\]

禁止使用验证集 C-index 作为训练奖励。最终风险应对多个证据集合进行概率边缘化，而不是挑选单个最高奖励集合。

### 5.3 第一版范围

- 先聚类 WSI patch，避免直接搜索数千个实例；
- 组学只使用 pathway token；
- 限制每个集合的最大规模；
- 第一版仅处理离散证据选择；
- 先检查策略塌缩和奖励可辨识性，再跑真实数据。

### 5.4 必要消融

- GFlowNet 与普通 Top-K；
- 单一集合与多集合边缘化；
- 去掉多样性奖励；
- 去掉稀疏奖励；
- 不同集合规模；
- 不同随机种子的证据重合率。

### 5.5 Go/No-Go 条件

- 能稳定产生多个不同且高价值的证据集合；
- 不出现策略完全塌缩；
- 相比 Top-K 至少改善性能或解释稳定性之一；
- 计算成本能够支持两癌种两折筛选。

### 5.6 起点文献与代码

- GFlowNet: <https://arxiv.org/abs/2106.04399>
- GFlowNet Foundations: <https://arxiv.org/abs/2111.09266>
- torchgfn: <https://github.com/GFNOrg/torchgfn>

## 6. 路线三：Flow Matching 概率跨模态生成

### 6.0 撞车警告（2026-08-04 补充，优先级下调）

**"用 Flow Matching 从组织形态生成分子表达"已被占用，且不止一篇：**

- STFlow: Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching, arXiv 2506.05361 — 全片级 flow matching 生成空间转录组，并显式建模细胞间相互作用。
- HoloTea: 3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology, arXiv 2511.14613。
- 相邻占用：Nat Commun 2025 用跨模态生成基因表达来改进多模态预测（<https://www.nature.com/articles/s41467-025-66961-9>）；CLDVAE (arXiv 2503.09496) 已占用"生成式补缺失模态做生存预测"。

6.1 的科学问题陈述（形态到分子是一对多、确定性 MSE 不成立）本身正确，但**这个论点已经被上述工作提出并实现过**。按当前 6.2 的设计直接推进，几乎肯定会被要求与 STFlow 类工作同口径比较，且创新点只剩"应用到生存预测"。

可能的剩余空间（均未验证，需单独复检）：

- 生成目标从 spot 级表达换成 **pathway 级 token 分布**，即不生成表达谱而直接生成下游任务所需的表征分布；上述工作都生成表达值。
- 把生成不确定性与**删失**耦合：删失患者的风险边缘化中，生成方差与删失时间共同决定预测区间。这是生存特有的，STFlow 不涉及。
- 与路线六联合：用生成采样的分散度作为保形预测的非一致性分数。

**处置决定：** 除非能把创新点收缩到上述某一条并通过单独复检，本路线从"第 3 优先级"降为储备项，不进入 10.1 的最小可行性验证前三位。方案 C 相应降级。

### 6.1 科学问题

确定性 MSE 或余弦重建默认一张 WSI 对应唯一组学表示，但形态到分子的映射通常具有一对多性和不确定性。

### 6.2 初步模型

学习条件分布：

\[
p(z_{\mathrm{omic}}\mid z_{\mathrm{WSI}}),
\]

并在缺失组学时生成多个可能表示，将风险边缘化：

\[
p(h\mid z_h)=\int p(h\mid z_h,z_g)
p(z_g\mid z_h)\,dz_g.
\]

第一阶段只实现 WSI 到组学的单向条件 Flow Matching；只有单向版本有效后才考虑双向生成。

### 6.3 必要消融

- MSE、余弦、VAE 与 Flow Matching 重建；
- 单次生成与多次采样；
- 是否加入生存条件；
- 单向与双向生成；
- 完整模态与缺失模态测试。

### 6.4 特有评价

- pathway-level correlation；
- 缺失组学下的 C-index 和 IBS；
- 多次采样的风险方差；
- 生成不确定性与预测误差的相关性。

### 6.5 起点文献与代码

- Flow Matching, ICLR 2023: <https://openreview.net/pdf?id=PqvMRDCJT9t>
- Meta Flow Matching: <https://github.com/facebookresearch/flow_matching>

## 7. 路线四：Neural Sheaf 异质证据传播

### 7.1 科学问题

病理与组学证据不必相似，有时可能相互矛盾。普通图传播或强制对齐容易把异质信息过度平滑。

### 7.2 初步模型

建立病理—通路联合图：

- 病理簇节点；
- pathway 节点；
- WSI 空间边；
- pathway 生物关系边；
- 病理—组学跨模态边。

每类边学习关系特异的 restriction map，使信息经过转换后传播，而不是要求相邻节点直接相似。

### 7.3 必要消融

- 普通 GCN；
- Hypergraph；
- Neural Sheaf；
- 仅模态内边与加入跨模态边；
- 固定映射与可学习映射；
- oversmoothing 和表示可分性检查。

### 7.4 Go/No-Go 条件

只有在小型合成图上验证传播、梯度和异质关系行为后，才允许进入真实 TCGA 训练。若无法证明其优于普通图模型的必要性，不作为论文主线。

### 7.5 起点文献与代码

- Neural Sheaf Diffusion, NeurIPS 2022: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html>
- 代码：<https://github.com/twitter-research/neural-sheaf-diffusion>
- OpenReview（含 diagonal / orthogonal / general restriction map 三档消融，小样本下应直接沿用前两档）：<https://openreview.net/forum?id=vbPsD-BhOZ>

超图扩展（与 7.3 消融中的 Hypergraph 对照项直接相关，不应遗漏）：

- Sheaf Hypergraph Networks, NeurIPS 2023: <https://arxiv.org/abs/2309.17116>；代码：<https://github.com/IuliaDuta/sheaf_HNN>
- Hypergraph Neural Sheaf Diffusion（对称单纯集构造，解决超图的定向歧义与邻接稀疏）: <https://arxiv.org/abs/2505.05702>

有了 sheaf hypergraph 之后，7.3 的"Hypergraph vs Neural Sheaf"不再是两个割裂的对照，而是同一族方法的两档，消融逻辑更干净。

### 7.6 概念定位建议

本路线最有价值的一句话不是"用层论做图神经网络"，而是：**跨模态失谐（disagreement）在图上就是异嗜性，异嗜性的正确工具是层论。** 形态学看着惰性、分子分型却是高危的患者，在普通 GCN 里只会被过平滑抹平，而 sheaf Laplacian 的 restriction map 允许非平凡符号，把矛盾保留为有判别力的信号。这类患者在预后上恰好是临床价值最高的一批。

由此可额外产出一个可解释量：逐边一致性能量 \(\lVert F_{u\triangleleft e}x_u - F_{v\triangleleft e}x_v\rVert^2\)，直接读作"这条证据与其他证据有多矛盾"，可作为辅助损失或患者分层依据。该量应纳入 7.3 消融与可解释性验证。

## 8. 路线五：Deep Equilibrium 平衡态融合

### 8.1 科学问题

现有融合模块人为指定交叉注意力迭代次数，无法判断两个模态是否已经完成信息交换。

### 8.2 初步模型

令融合表示满足：

\[
z^*=F_\theta(z^*,z_{\mathrm{WSI}},z_{\mathrm{omic}}),
\]

使用固定点求解器得到平衡表示，并记录收敛残差、求解次数和患者级稳定性。

### 8.3 必要消融

- 1、2、3、5 层普通迭代融合；
- 参数共享的有限迭代；
- DEQ 平衡融合；
- 有无 Jacobian 稳定正则；
- 固定与自适应停止阈值。

### 8.4 定位

若只能证明它是更深的融合层，而不能将平衡残差与患者风险、模态冲突或解释联系起来，则只作为技术模块，不独立成文。

### 8.5 起点文献与代码

- Deep Equilibrium Models, NeurIPS 2019: <https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html>
- 代码：<https://github.com/locuslab/deq>

## 9. 路线六：Conformal Survival 校准预测

### 9.1 科学问题

C-index 只评价风险排序，不能给出具有覆盖率保证的患者生存时间范围。

### 9.2 初步模型

在最终选定模型外部增加 conformal calibration，输出生存时间下界或双侧区间：

\[
\hat T_{\mathrm{lower}}(x),\qquad
[\hat T_{\mathrm{lower}}(x),\hat T_{\mathrm{upper}}(x)].
\]

训练、模型选择、校准和最终测试必须严格分离，禁止重复使用验证集完成校准和报告。

### 9.3 特有评价

- 经验覆盖率；
- 区间宽度；
- 不同癌种、风险组和缺失模态状态下的条件覆盖率；
- 删失患者与事件患者的校准差异。

### 9.4 定位

该路线通常不会直接提高 C-index。它作为最佳模型的临床可信度增强模块，不作为纯提分主线。

### 9.5 起点文献与代码

- General Right-Censored Conformal Survival, ICLR 2025: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/f49d76cf84df83a611883c621c96d2d9-Abstract-Conference.html>；OpenReview: <https://openreview.net/forum?id=JQtuCumAFD>
- Adaptive Conformal Survival 代码：<https://github.com/zhimeir/adaptive_conformal_survival_paper>

源头工作（Type-I 右删失下的下界保证，必须引用）：

- Conformalized Survival Analysis, JRSS-B 2023: <https://arxiv.org/abs/2103.09763>
- 病理侧已有的保形工作只覆盖**分类**诊断不确定性，不涉及生存：<https://www.nature.com/articles/s41467-022-34945-8>。这是本路线的空白确认依据，也是相关工作一节的必引对照。

### 9.5 一个被低估的用法：加权保形处理队列漂移

标准保形要求校准集与测试集可交换。跨癌种、跨队列时该条件不成立，而**加权保形（weighted conformal）正是为协变量漂移设计的**。这给本路线一个比"外挂校准模块"更强的定位：不只是给出区间，而是给出**在队列漂移下仍然有效**的区间，并把 9.3 中的条件覆盖率从描述性指标升级为方法主张。

若采用这一定位，本路线的提分潜力评分仍然低（1.5），但论文叙事评分可从 4.0 上调，且更有资格作为方案 A 的辅助贡献而非附录内容。

## 10. 分阶段执行顺序

### 10.1 最小可行性验证

按以下顺序进行：

1. 生存感知多模态信息分解；
2. GFlowNet 多样化证据集合；
3. Flow Matching 概率生成（**依 6.0 降级为储备项，实际不排在第 3 位**）；
4. Neural Sheaf；
5. Deep Equilibrium；
6. Conformal Survival。

第二批候选（路线七至九）的插入位置见 14.5。

每条路线首先完成：

- 张量形状测试；
- 梯度和参数有限性测试；
- 极小数据过拟合测试；
- 单癌种单折筛选；
- 核心机制特有的数值诊断。

此阶段不运行完整五折。

### 10.2 双癌种筛选

通过最小测试后，在两个数据完整癌种上统一运行 folds 0、2，并保持相同 epoch、早停、checkpoint 选择和随机种子。

进入完整五折的最低条件：

- 平均表现超过统一基线；
- 改进不依赖单个异常 fold；
- 核心消融方向正确；
- 没有数值不稳定或数据泄漏；
- 至少两个随机种子结论一致。

### 10.3 完整五折

最多允许三个候选进入：

- 两至三个癌种完整五折；
- 多随机种子；
- 缺失模态实验；
- 完整消融；
- 与 SlotSPE、MCAT、MOTCAT、CMTA、OTSurv 和统一 DCT 基线进行同口径比较。

## 11. 主线选择标准

最终方法应同时满足：

1. 五折平均 C-index 有实质提升，目标至少约 +0.01；
2. 多数 fold 改善，而不是单个 fold 拉高均值；
3. 标准差没有明显恶化；
4. 核心模块能通过独立消融验证；
5. 能回答一个明确的新科学问题；
6. 不依赖癌种专属人工调参；
7. 与 SlotSPE、DCT 及直接竞争方法的创新边界清楚。

## 12. 预期论文组合

### 方案 A：最稳妥

主方法：生存感知多模态信息分解。  
辅助贡献：Conformal Survival 校准。  
主题：分解共享、独有与协同预后信息，并提供可靠的生存预测。

### 方案 B：原创性最强

主方法：GFlowNet 风险证据集合。  
辅助贡献：证据稳定性和多样性分析。  
主题：从单一注意力解释转向多样化病理—分子风险证据分布。

### 方案 C：缺失模态方向（已依 6.0 降级为储备）

主方法：Flow Matching。  
辅助贡献：生成不确定性与 Conformal Survival。  
主题：通过概率跨模态生成实现缺失组学条件下的可靠生存预测。

降级原因：主方法与 STFlow、HoloTea 及 Nat Commun 2025 的跨模态生成工作撞车。若要恢复，需先把创新点收缩到 6.0 列出的剩余空间之一并通过单独复检。

### 方案 D：高风险高回报

主方法：Neural Sheaf。  
辅助贡献：跨模态矛盾证据解释。  
主题：在病理—组学联合图中建模异质和矛盾的预后信号。

## 13. 执行纪律

- 六条路线独立开发，禁止一开始全部组合。
- 每次实验只引入一个可归因的核心机制。
- 失败结果如实保留，不继续叠加损失掩盖失败。
- 先验证机制成立，再扩大训练规模。
- 所有实验记录患者集合、split、特征版本、seed、epoch 和 checkpoint 选择方式。
- 最终只保留一个主要模型，其余路线作为独立储备工作。

---

## 14. 第二批候选（2026-08-04 补充）

本批三条路线来自同一轮跨领域检索，均通过 2.2 的占用审计，且与既有六条路线不重叠。它们服从完全相同的阶段 0 协议与 Go/No-Go 纪律，不因为"新"而获得任何豁免。

| 优先级 | 候选方法 | 新颖性 | 任务适配性 | 实现难度 | 提分潜力 | 论文叙事 |
|---:|---|:---:|:---:|:---:|:---:|:---:|
| 7 | 主动证据获取（Active Feature Acquisition） | 5.0 | 5.0 | 3.0 | 3.0 | 5.0 |
| 8 | Neural CDE 连续时间风险轨迹 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| 9 | 反事实生存 / 异质治疗效应 | 4.0 | 3.0 | 4.0 | 5.0 | 5.0 |
| — | 不变学习 / anchor regression（观察项） | 2.0 | 4.0 | 2.0 | 1.0 | 3.0 |

### 14.1 路线七：主动证据获取

#### 科学问题

现有多模态生存工作全部假设模态是给定的，研究的是"给了怎么融"。真实临床问题是反过来的：**这个患者还值不值得再做一次组学检测**。这是从预测模型到决策支持系统的定位差别，本方向无人做过。

#### 初步模型

- 状态 \(s_t\)：已获取模态集合与当前风险估计。
- 动作：获取下一项证据（组学 panel、追加 WSI、特定 pathway 检测）或停止。
- 奖励：

\[
r_t = -\bigl[\mathcal L_{\mathrm{surv}}(t{+}1)-\mathcal L_{\mathrm{surv}}(t)\bigr] - \lambda\,\mathrm{cost}(a_t).
\]

**删失处理**：奖励中的 \(\mathcal L_{\mathrm{surv}}\) 用既有离散时间右删失 NLL，天然处理删失。若改用 C-index 类奖励则必须 IPCW 校正，否则奖励直接被删失分布污染——这一点是本路线最容易出错的地方，应写入数值诊断。

#### 第一版范围（关键风险控制）

**禁止上在线强化学习。** 当前患者规模下 PPO 一类算法会崩。第一版走摊销贪心（amortized greedy）：训练一个小网络直接预测每个候选动作的期望信息增益，训练信号来自离线枚举的模态子集。这把强化学习问题降级为监督回归，可在现有数据规模下跑通。只有摊销贪心版本先证明有效，才考虑序贯策略版本。

#### 与既有路线的关系

- 与 `stagewise_pt.md` 的分阶段思路共享"阶段信息增量"语义，但两者的对象不同：SPT 是给定阶段学阶段特异运输，本路线是决定**是否进入下一阶段**。若两者都成立，本路线可作为 SPT 的决策层，但第一版必须独立实现、独立消融。
- 与路线一互补：路线一测量"哪些信息是协同的、哪些是冗余的"，本路线用该测量决定"该去买哪份信息"。两者共用同一套信息增量语言，是本文件内最自然的组合。

#### 必要消融

- 固定全模态 vs 摊销贪心获取；
- 随机获取顺序 vs 学习获取顺序；
- 不同 \(\lambda\) 下的代价—性能帕累托前沿（**必须报前沿，不能只报单点**，否则代价权重会被质疑是调出来的）；
- 停止准则：固定预算 vs 自适应阈值；
- 奖励用 NLL vs 用 IPCW 校正的排序指标。

#### Go/No-Go 条件

- 在相同平均获取代价下优于随机获取顺序；
- 代价—性能前沿单调，不出现"多花钱反而更差"的大面积区域；
- 获取策略不退化为"总是获取全部"或"总是不获取"；
- 代价定义能给出可辩护的来源（周转天数、检测费用或侵入性），并对其做敏感性分析。

#### 起点文献与代码

- Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams, ICLR 2019: <https://arxiv.org/abs/1901.00243>
- EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE, ICML 2019（信息增益估计的标准做法）
- Active Feature Acquisition with Generative Surrogate Models, ICML 2021: <https://arxiv.org/pdf/1709.05964>
- 医学侧仅有的相邻工作（分类任务，非生存）：<https://repositori.upf.edu/bitstreams/e6e2bc02-0933-4bae-98ff-301758c0efdb/download>

### 14.2 路线八：Neural CDE 连续时间风险轨迹

#### 科学问题

分阶段建模隐含"阶段离散且等距"的假设，但真实诊疗时间间隔从数天到数月不等，而这个间隔本身携带预后信息（延迟治疗、需要新辅助）。同时随访本身是不规则采样的。

#### 初步模型

令风险状态由观测流 \(X(t)\) 驱动：

\[
\mathrm{d}\eta = f_\theta(\eta)\,\mathrm{d}X(t),\qquad
\lambda(t)=\sigma\bigl(g_\theta(\eta(t))\bigr).
\]

控制路径 \(X(t)\) 由各阶段可用证据的插值构成，证据在其获取时刻作为跳变注入。**分阶段离散模型是本模型在观测时刻的采样特例**，因此消融关系天然干净：这是严格推广而非替换，可直接与既有阶段化方法对照。

删失处理比离散近似更干净：删失患者的似然积分到 \(C_i\) 为止即可。

#### 必要消融

- 离散等距阶段 vs Neural CDE；
- 是否把时间间隔作为控制路径的一个通道（这是本路线的核心动机，必须单独消融）；
- 插值方案（线性 vs Hermite cubic）；
- 求解器与容差敏感性；
- 与 Neural ODE（无控制项）对照，证明"受控"这一步的必要性。

#### 主要风险

小样本 + ODE 求解器容易训练不稳，需要 adjoint 与刚性控制。既有的 X-CAL 类校准项与温度参数需要重新推到连续时间，这部分工程成本是本文件所有路线中第二高的（仅次于 GFlowNet）。**必须先在合成不规则时间序列上验证，再碰真实数据。**

#### 起点文献与代码

- Neural Controlled Differential Equations for Irregular Time Series, NeurIPS 2020 Spotlight: <https://arxiv.org/abs/2005.08926>；代码：<https://github.com/patrick-kidger/NeuralCDE>（配合 `torchcde`）
- Neural Jump ODE（观测时刻跳变的一致性理论）: <https://github.com/HerreraKrachTeichmann/NJODE>
- Stable Neural SDEs, ICLR 2024 Spotlight（稳定性处理）: <https://github.com/yongkyung-oh/Stable-Neural-SDEs>
- 生存侧仅有的相邻工作（纯表格，不涉及 WSI/组学）：Neural ODEs for Multi-State Survival Analysis

### 14.3 路线九：反事实生存 / 异质治疗效应

#### 科学问题

本方向所有工作做的都是**预后**（会活多久），临床真正要的是**疗效预测**（做这个治疗能多活多久）。这是本文件所有候选中天花板最高的问题定义。

#### 初步模型

在最终风险头之前分叉为处理特异 hazard：

\[
\eta^{A=1},\quad \eta^{A=0},
\]

并加表征平衡损失（MMD 或 Wasserstein）对齐 treated / control 的表征分布。SurvITE 的核心贡献是指出此处的协变量漂移**比"混淆偏差 + 删失偏差"的简单叠加更复杂**，并给出相应的泛化界，这一点必须在方法动机中体现，否则会被看作"CATE + 生存的拼接"。

#### 立项前的硬性前置检查

**在写任何代码之前，先确认数据里治疗变量的可用性。** 需要确认三件事：

1. 治疗变量（辅助化疗、放疗、靶向）是否被记录，缺失率多少；
2. 该变量是否有足够变异，还是绝大多数患者同一取值；
3. 是否存在足以支撑 positivity 假设的重叠区域。

若三项中任一不成立，本路线不成立，**不要投入实现**。这项检查的成本不到半天，收益是避免数周的无效工作。

#### 主要风险

观测性数据上的因果声明需要 positivity 与 no-unmeasured-confounding，审稿人会紧盯。基本无法用 ground-truth 反事实验证，只能依赖敏感性分析与安慰剂检验。这是"叙事分 5.0、但被拒风险同样最高"的典型路线。

#### 必要消融

- 单一风险头 vs 处理特异双头；
- 有无表征平衡损失；
- 平衡损失强度敏感性；
- 与朴素分层（按治疗分别建模）对照；
- 未观测混淆的敏感性分析（如 Rosenbaum 界）。

#### 起点文献与代码

- SurvITE: Learning Heterogeneous Treatment Effects from Time-to-Event Data, NeurIPS 2021: <https://openreview.net/forum?id=f0_tkoEJV88>；代码：<https://github.com/chl8856/survITE>
- BITES: Balanced Individual Treatment Effect for Survival Data, Bioinformatics 2022

### 14.4 观察项：不变学习 / anchor regression

多癌种、多队列是天然的 environment 划分，叙事顺畅。但已有系统性实证显示，**调优良好的 ERM 常常反超 IRM 与 V-REx**：多站点乳腺影像的对比见 <https://arxiv.org/html/2503.06759>；多中心 ICU 上 anchor regression 的大规模研究见 <https://ar5iv.labs.arxiv.org/html/2507.21783>。

处置：**不作为独立路线立项**。可作为最终主线的一节跨队列鲁棒性分析，并且必须同时报告调优良好的 ERM 基线，否则结论不可信。

参考：IRM <https://www.researchgate.net/publication/334288906_Invariant_Risk_Minimization>；V-REx <https://openreview.net/forum?id=foNTMJHXHXC>。

### 14.5 第二批的执行位置

第二批不改变 10.1 的既有顺序，而是按以下方式插入：

- **路线七**与路线一并列进入最小可行性验证。两者共用信息增量语义，可共享部分诊断代码，但必须分别消融、分别归因。
- **路线八**排在 Neural Sheaf 之后。它的合成数据验证门槛必须先过。
- **路线九**在完成 14.3 的前置数据检查之后才排期。检查不通过则直接归档。
- 观察项不排期。

10.3 的"最多三个候选进入完整五折"约束对第一批与第二批合计生效，不因新增路线而放宽。

### 14.6 推荐组合（补充方案）

**方案 E：测量 + 决策（推荐）**

主方法：路线一（生存感知信息分解）+ 路线七（主动证据获取）。
主题：先定量刻画协同、冗余与独有预后信息，再据此决定应当为患者获取哪份证据。

选择理由：两条路线共用同一套信息增量语言，逻辑闭环；都不需要新数据；都不依赖高风险数值方法；且分别提供一张可上正文的主图（泛癌 R/U/S 谱、代价—性能帕累托前沿）。相对方案 A，它把 Conformal 这个提分潜力仅 1.5 的辅助项，换成一个本身就有独立问题定义的第二贡献。

**方案 D 的修订：** 若采用 Neural Sheaf 作为主线，建议按 7.6 的概念定位改写卖点为"跨模态失谐即异嗜性"，并把逐边一致性能量作为可解释性主张，而不是停留在"用了层论"。

**方案 C 的处置：** 依 6.0 降级为储备方案。

## 15. 修订记录

| 日期 | 变更 |
|---|---|
| 2026-08-04 | 建立文件，六条候选路线 |
| 2026-08-04 | 补 2.2 新颖性审计；路线三因撞车（STFlow / HoloTea）下调并降级；路线一补 PID 理论源头与生存特异设计（4.6）；路线四补 sheaf hypergraph 与概念定位（7.6）；路线六补加权保形定位（9.5）；新增第二批候选路线七至九与观察项（第 14 节） |
| 2026-08-04 | 建立实现工作区 `E:\第三篇工作`：九条路线各一目录，15 个上游仓库已拉取并记录 commit/许可/可跑性。**修正 4.5：DMIL 声明的代码仓库不存在**，改以 `LSMI_Estimator` 为起点。发现三项许可限制（四个仓库无许可、`flow_matching` 为 CC BY-NC、`adaptive_conformal_survival_paper` 为 GPL），详见 `第三篇工作/docs/REPO_INVENTORY.md` |

> 本文件的占用审计基于 2026-08-04 的针对性检索，不构成对全部文献的绝对否定。每条路线立项前须单独复检一次，并把复检日期与结论追加到第 15 节。
