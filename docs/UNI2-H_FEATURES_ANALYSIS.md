# UNI2-h WSI 特征数据分析报告

> 数据路径: `/data1/TCGA-UNI2-h-features/`
> 分析时间: 2026-07-23
> 状态: 初步分析，远端需确认并完成集成

---

## 1. 数据概览

| 属性 | 值 |
|:-----|:---|
| 总大小 | **265 GB** |
| 癌种数量 | **10** |
| 文件格式 | `.h5` (HDF5) |
| 存储方式 | 每癌种一个 `tar.gz` 压缩包 |
| 编码器版本 | **UNI2-h** (UNI v2, H-optimized) |

---

## 2. 目录结构

```
/data1/TCGA-UNI2-h-features/
├── blca/uni2-h/pt_files/TCGA-BLCA.tar.gz        (30 GB)
├── brca/uni2-h/pt_files/TCGA-BRCA_IDC.tar.gz    (46 GB)
├── coadread/uni2-h/pt_files/TCGA-COAD.tar.gz    (526 MB)
│             └── pt_files/TCGA-READ.tar.gz       (501 MB)
├── hnsc/uni2-h/pt_files/TCGA-HNSC.tar.gz         (16 GB)
├── kirc/uni2-h/pt_files/TCGA-KIRC.tar.gz         (29 GB)
├── luad/uni2-h/pt_files/TCGA-LUAD.tar.gz         (33 GB)
├── lusc/uni2-h/pt_files/TCGA-LUSC.tar.gz         (34 GB)
├── skcm/uni2-h/pt_files/TCGA-SKCM.tar.gz         (23 GB)
├── stad/uni2-h/pt_files/TCGA-STAD.tar.gz         (18 GB)
├── ucec/uni2-h/pt_files/TCGA-UCEC.tar.gz         (39 GB)
```

### 单个文件命名示例

```
TCGA-BT-A3PH-01Z-00-DX1.18FB196C-9F66-4676-81DC-F58A0A5577D8.h5
```

格式: `TCGA-{slide_id}.{UUID}.h5`

---

## 3. 与现有 SurvOT-Rank 的差异

### 3.1 格式差异

| 项目 | 当前 SurvOT-Rank | UNI2-h 新数据 |
|:-----|:-----------------|:--------------|
| 文件格式 | `.pt` (PyTorch tensor) | `.h5` (HDF5) |
| 存储方式 | 扁平目录，每 WSI 一个 `.pt` 文件 | 每癌种一个 `tar.gz`，内部含所有 `.h5` |
| 路径前缀 | `data/TCGA-PatchFeature/` 或 `data/TCGA_uni_features/pt_files/` | `/data1/TCGA-UNI2-h-features/{cancer}/uni2-h/pt_files/` |
| 编码器 | UNI v1 | **UNI v2 (H)** |
| encoding_dim | **768** | **待确认** (可能是 1024/1536/2048) |
| num_patches | 待确认 | **待确认**（可能不同） |

### 3.2 加载代码对比

**当前 SurvOT-Rank 加载方式** (`dataset_survival.py:312-326`):
```python
def load_wsi(self, slides):
    slide_ids = slides.split(", ")
    wsi = []
    for slide_id in slide_ids:
        wsi_path = os.path.join(self.wsi_path, '{}.pt'.format(slide_id.rstrip('.svs')))
        if os.path.exists(wsi_path):
            wsi.append(torch.load(wsi_path))
        else:
            wsi.append(torch.zeros((self.dataset_factory.num_patches, self.encoding_dim)))
    wsi = torch.cat(wsi, dim=0).type(torch.float32)
    return wsi
```

**UNI2-h 需要的新加载方式**:
- 需要预解压 `tar.gz` 或运行时按需读取
- 需要使用 `h5py` 替代 `torch.load`
- 文件名匹配逻辑不变 (slide_id → UUID mapping)

### 3.3 癌种覆盖

| 癌种 | 当前是否有 WSI | UNI2-h |
|:-----|:------------:|:------:|
| BLCA | ✅ | ✅ |
| BRCA | ✅ | ✅ |
| LUAD | ✅ | ✅ |
| LUSC | ✅ | ✅ |
| SKCM | ✅ | ✅ |
| **COADREAD** | ❌ | ✅ **新增** |
| **HNSC** | ❌ | ✅ **新增** |
| **KIRC** | ❌ | ✅ **新增** |
| **STAD** | ❌ | ✅ **新增** |
| **UCEC** | ❌ | ✅ **新增** |

---

## 4. 需要远端确认的事项

### 4.1 必须确认
1. **tensor 维度**: 提取一个 `.h5` 文件确认 `shape` 和 `dtype`
   ```python
   import h5py, tarfile, io
   tar = tarfile.open('TCGA-BLCA.tar.gz', 'r:gz')
   # 需要提取单个文件查看
   ```

2. **num_patches 数量**: 是否与当前 UNI v1 (4096) 一致

### 4.2 需要代码改动（按优先级）

1. **添加 h5py 依赖** (`requirements.txt` 或 `environment.yml`)
2. **新增 H5WSIDataset** 或修改 `load_wsi()` 支持 `.h5` 读取
3. **解压策略**: 预解压（需要额外 ~300-500 GB 磁盘）vs 运行时按需从 tar.gz 读取
4. **encoding_dim**: 如果维度不是 768，需要更新所有相关配置和模型参数
5. **新增癌种配置**: 为 COADREAD/HNSC/KIRC/STAD/UCEC 创建 config yaml
6. **更新 WSI 路径自动检测**: `dataset_survival.py` 第 276-282 行的路径推断逻辑

### 4.3 预处理脚本建议

```bash
# 方案 A: 预解压（简单但需大量磁盘）
for tar in /data1/TCGA-UNI2-h-features/*/uni2-h/pt_files/*.tar.gz; do
    cancer=$(echo $tar | grep -oP 'TCGA-\w+' | head -1)
    mkdir -p /data/TCGA-PatchFeature/${cancer}
    tar -xzf $tar -C /data/TCGA-PatchFeature/${cancer}/
done

# 方案 B: 解压为 .pt 格式（兼容现有代码）
# 需要读取 .h5 → 转为 torch.tensor → 保存为 .pt
```

---

## 5. 风险评估

| 风险 | 影响 | 缓解 |
|:-----|:-----|:-----|
| encoding_dim ≠ 768 | 模型不兼容，需重新训练 | 先确认维度再决定 |
| 磁盘不足 (解压后~400GB+) | 无法预解压 | 方案 B: 按需读取 |
| num_patches ≠ 4096 | 需要修改模型配置 | 检查后可调整 `num_patches` 参数 |
| H5 vs PT 数据类型差异 | NaN 或精度问题 | 统一转为 float32 pt |
| COADREAD 有两个 tar.gz | 需要合并或特殊处理 | 统一目录结构 |

---

## 6. 远端下一步

1. **提取一个 sample .h5**，确认 tensor shape
2. **决定解压策略** (预解压 vs 按需读取)
3. **修改 `dataset_survival.py`** 的 `load_wsi()` 支持 h5
4. **更新 encoding_dim** 和相关配置
5. **为 5 个新癌种创建 config yaml**
6. **在 SurvOT-Rank 上 DM**

---
