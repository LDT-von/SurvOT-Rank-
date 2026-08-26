import os
import torch
import pandas as pd

DATASET_CSV_DIR = "/home/ubuntu/newSlotSPE/SlotSPE/dataset_csv/clinical/all"
DATA_ROOT_DIR = "/data/CPathPatchFeature"
NUM_PATCHES = 2048
ENCODING_DIM = 1024

datasets = ["hnsc", "skcm", "stad"]

for dataset in datasets:
    csv_path = os.path.join(DATASET_CSV_DIR, f"{dataset}.csv")
    if not os.path.exists(csv_path):
        print(f"跳过 {dataset}: CSV文件不存在")
        continue
    
    df = pd.read_csv(csv_path)
    pt_files_dir = os.path.join(DATA_ROOT_DIR, dataset, "uni", "pt_files")
    
    os.makedirs(pt_files_dir, exist_ok=True)
    
    placeholder = torch.randn(NUM_PATCHES, ENCODING_DIM)
    
    slide_ids = []
    for idx, row in df.iterrows():
        wsi_val = row.get("wsi", "")
        if pd.isna(wsi_val) or str(wsi_val).lower() == "nan":
            continue
        
        slides = str(wsi_val).split(", ")
        for slide in slides:
            slide = slide.strip()
            if slide.endswith(".svs"):
                slide_id = slide[:-4]
            else:
                slide_id = slide
            slide_ids.append(slide_id)
    
    slide_ids = list(set(slide_ids))
    
    count = 0
    for slide_id in slide_ids:
        pt_path = os.path.join(pt_files_dir, f"{slide_id}.pt")
        if not os.path.exists(pt_path):
            torch.save(placeholder, pt_path)
            count += 1
    
    print(f"{dataset}: 创建了 {count} 个占位符文件，共 {len(slide_ids)} 个 slide")