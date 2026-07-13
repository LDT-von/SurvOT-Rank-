#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Unified training runner for the cleaned SurvOT-Rank framework."""

import os
import sys
import gc
import time
import pickle
import traceback
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Paths
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TRAINING_DIR))

from survot_rank.training.paths import SLOTSPE_DIR, ensure_slotspe_in_path  # noqa
ensure_slotspe_in_path()

# Packaged legacy dataset, loss, and metric utilities.
from dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset, _collate_pathways
from utils.loss_func import NLLSurvLoss, SurvPLE, RankLoss, SinkhornSurvLoss
from utils.general_utils import (
    _get_start_end, _prepare_for_experiment, _save_pkl, _print_network
)


def set_global_seed(seed):
    """鍏ㄥ眬 RNG 澶嶄綅浠ヤ繚璇佸悓 seed 浜х敓鐩稿悓鍒濆鍖栨潈閲?

    NOTE: cuDNN benchmark=True (榛樿)锛屼笉寮哄埗 deterministic銆?    鍘熷 V45 (2026-06-30) 浣跨敤闈炵‘瀹氭€?cuDNN 绠楁硶璺戝嚭 0.7105锛?    寮哄埗 deterministic=True 浼氱郴缁熸€ч檷浣?~0.01-0.02锛屾晠淇濈暀闈炵‘瀹氭€с€?    DataLoader generator 宸蹭繚璇?batch 椤哄簭纭畾鎬с€?    """
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 鎭㈠鍘熷 V45 鐨?cuDNN 閰嶇疆锛歜enchmark=True 鍏佽閫夋嫨鏈€蹇畻娉?    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
from utils.core_utils import (
    _process_data_and_forward, _calculate_risk, _update_arrays,
    _calculate_metrics, _extract_survival_metadata
)
from sksurv.metrics import concordance_index_censored

# SurvOT-Rank model factory.
from survot_rank.training.model_factory import get_model
from survot_rank.training.extended_args import process_args_extended


def safe_write_line(log_file, message):
    try:
        log_file.write(message + "\n")
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 鏃ュ織鍐欏叆澶辫触锛岀鐩樼┖闂翠笉瓒? {error}")
        else:
            raise


def safe_flush(log_file):
    try:
        log_file.flush()
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 鏃ュ織 flush 澶辫触锛岀鐩樼┖闂翠笉瓒? {error}")
        else:
            raise


def safe_to_csv(records, csv_path):
    try:
        pd.DataFrame(records).to_csv(csv_path, index=False)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 璺宠繃鍐欏叆 {csv_path}锛岀鐩樼┖闂翠笉瓒? {error}")
            return False
        raise


def safe_pickle_dump(obj, output_path):
    try:
        with open(output_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 璺宠繃鍐欏叆 {output_path}锛岀鐩樼┖闂翠笉瓒? {error}")
            return False
        raise


def safe_torch_save(state_dict, output_path):
    try:
        torch.save(state_dict, output_path)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 璺宠繃鍐欏叆 {output_path}锛岀鐩樼┖闂翠笉瓒? {error}")
            return False
        raise


def get_free_space_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def ensure_min_free_space(path, min_free_gb, context):
    free_gb = get_free_space_gb(path)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"{context} has insufficient free space: {free_gb:.2f}GB available, "
            f"requires at least {min_free_gb:.2f}GB. Results path: {path}"
        )
    return free_gb


# ============================================================
# 妯″瀷涓庝紭鍖栧櫒鍒濆鍖?# ============================================================

def init_model_for_method(args, dataset_factory):
    """
    Build the selected SurvOT-Rank method.
    """
    if args.rna_format == "RNASeq":
        omics_input_dim = dataset_factory.num_genes if dataset_factory.num_genes is not None \
                          else dataset_factory.omic_sizes
    elif args.rna_format == "GeneEmbedding":
        omics_input_dim = 768
    else:
        omics_input_dim = None

    args.omic_sizes = dataset_factory.omic_sizes
    args.omic_names = dataset_factory.omic_names
    args.pathway_names = getattr(dataset_factory, 'pathway_names', None)

    print(f"[init] loading SurvOT-Rank method: {args.survot_method}")
    model = get_model(
        method=args.survot_method,
        args=args,
        omic_input_dim=omics_input_dim,
        omic_names=args.omic_names,
        pathway_names=args.pathway_names,
    )

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    _print_network(args.results_dir, model)
    return model


def init_loss_function(args):
    if args.bag_loss == 'nll_surv':
        return NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        return SurvPLE()
    elif args.bag_loss == 'rank_surv':
        return RankLoss()
    elif args.bag_loss == 'sinkhorn_surv':
        return SinkhornSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError


def init_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError


def init_scheduler(args, optimizer):
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    elif args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.max_epochs,
                                                     eta_min=args.eta_min)
    else:
        raise NotImplementedError


# ============================================================
# 鏁版嵁鍔犺浇
# ============================================================

def get_split(args, dataset_factory, fold):
    split_path = os.path.join(
        dataset_factory.data_path,
        "splits", "5fold", dataset_factory.study,
        f"fold_{fold}.csv",
    )
    split_df = pd.read_csv(split_path)
    dataset_factory.fit_label_bins(split_df["train"].dropna().tolist())

    train_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'train', fold, args.encoding_dim)
    test_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'val', fold, args.encoding_dim)

    # 鍚敤澶氳繘绋嬫暟鎹姞杞藉拰椤甸攣瀹氬唴瀛樹互鍔犻€?GPU 璁粌
    num_workers = getattr(args, 'num_workers', 4)
    pin_memory = True

    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True, collate_fn=_collate_pathways, pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(getattr(args, 'seed', 3))
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=_collate_pathways, pin_memory=pin_memory
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(getattr(args, 'seed', 3))
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    print(f"[data] train={len(train_data)}  val={len(test_data)}  num_workers={num_workers}  pin_memory={pin_memory}")
    return train_data, test_data, train_loader, test_loader


# ============================================================
# 璁粌 / 璇勪及寰幆
# ============================================================

def train_one_epoch(args, epoch, model, loader, optimizer, scheduler, loss_fn, log_file):
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    args.cur_epoch = epoch

    total_loss = 0.0
    all_risk_scores, all_censorships, all_event_times = [], [], []

    accumulation_steps = 1 if args.batch_size != 1 else 32
    total_batches = len(loader)
    smk = int(getattr(args, "max_smoke_batches", 0) or 0)
    n_batches = smk if smk > 0 else total_batches

    pbar = tqdm(enumerate(loader), total=n_batches,
                desc=f"  [Fold {args.cur_fold}] Epoch {epoch+1}/{args.max_epochs}",
                ncols=120, leave=False, dynamic_ncols=True)
    for batch_idx, data in pbar:
        out, y_disc, event_time, c = _process_data_and_forward(args, model, data, device)
        logits, slot_loss = out

        if args.bag_loss == "cox_surv":
            loss_surv = loss_fn(logits, event_time, c)
        else:
            loss_surv = loss_fn(logits, y_disc, event_time, c)
        loss_surv = loss_surv / y_disc.shape[0]

        loss = (loss_surv + slot_loss) / accumulation_steps
        loss.backward()

        if args.batch_size != 1:
            optimizer.step()
            optimizer.zero_grad()
        else:
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        risk, _ = _calculate_risk(logits)
        all_risk_scores, all_censorships, all_event_times = _update_arrays(
            all_risk_scores, all_censorships, all_event_times,
            event_time, c, risk, data
        )

        pbar.set_postfix(loss=f"{loss.item():.3f}", surv=f"{loss_surv.item():.3f}",
                         batch=f"{batch_idx+1}/{total_batches}", refresh=False)
        if batch_idx % 10 == 0:
            msg = f"  batch:{batch_idx} loss:{loss.item():.4f} surv:{loss_surv.item():.4f}"
            print(msg, flush=True)
            safe_write_line(log_file, msg)
        if smk > 0 and (batch_idx + 1) >= smk:
            pbar.set_postfix(loss=f"{loss.item():.3f}", surv=f"{loss_surv.item():.3f}",
                             batch=f"{batch_idx+1}/{total_batches}", status="smoke-stop", refresh=False)
            msg = f"  [smoke] stop train epoch after {batch_idx + 1} batch(es)"
            print(msg, flush=True)
            safe_write_line(log_file, msg)
            break
    pbar.close()

    scheduler.step()

    total_loss /= max(len(loader.dataset), 1)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times,
        all_risk_scores, tied_tol=1e-08
    )[0]

    msg = f"[Epoch {epoch}] train_loss={total_loss:.4f}  train_cindex={c_index:.4f}"
    print(msg)
    safe_write_line(log_file, msg)


def evaluate(args, dataset_factory, model, loader, loss_fn, survival_train=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.0
    all_risk_scores, all_risk_by_bin, all_censorships, all_event_times = [], [], [], []
    all_logits, all_case_ids = [], []
    case_ids = loader.dataset.label_df["case id"]
    count = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            out, y_disc, event_time, c = _process_data_and_forward(args, model, data, device, test=True)
            logits, _ = out

            if args.bag_loss == "cox_surv":
                loss = loss_fn(logits, event_time, c)
            elif args.bag_loss == "nll_surv":
                loss = loss_fn(logits, y_disc, event_time, c)
            else:
                raise ValueError(f"Loss not impl: {args.bag_loss}")

            total_loss += loss.item()
            risk, risk_by_bin = _calculate_risk(logits)
            all_risk_by_bin.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times = _update_arrays(
                all_risk_scores, all_censorships, all_event_times,
                event_time, c, risk, data
            )
            all_logits.append(logits.detach().cpu().numpy())
            all_case_ids.append(case_ids.values[count])
            count += 1

    total_loss /= max(len(loader.dataset), 1)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_risk_by_bin = np.concatenate(all_risk_by_bin, axis=0)

    patient_results = {}
    for i in range(len(all_case_ids)):
        cid = all_case_ids[i]
        patient_results[cid] = {
            "risk": all_risk_scores[i],
            "censor": all_censorships[i],
            "time": all_event_times[i],
            "logits": all_logits[i],
        }

    c_index, c_index_ipcw, BS, IBS, iauc = _calculate_metrics(
        loader, dataset_factory, survival_train,
        all_risk_scores, all_censorships, all_event_times, all_risk_by_bin
    )
    return patient_results, c_index, c_index_ipcw, BS, IBS, iauc, total_loss


# ============================================================
# 鏁存姌璁粌 + 璇勪及
# ============================================================

def train_one_fold(args, dataset_factory, fold, log_file):
    print(f"\n{'=' * 60}\n[Fold {fold}] start\n{'=' * 60}")
    safe_write_line(log_file, f"\n=== Fold {fold} start ===")
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, f"Fold {fold} 寮€濮嬪墠")

    args.cur_fold = fold

    train_data, val_data, train_loader, val_loader = get_split(args, dataset_factory, fold)
    model = init_model_for_method(args, dataset_factory)
    loss_fn = init_loss_function(args)
    optimizer = init_optimizer(args, model)
    scheduler = init_scheduler(args, optimizer)

    survival_train = _extract_survival_metadata(dataset_factory)

    args.max_cindex = 0.0
    args.max_cindex_epoch = 0

    best_results = None
    final_metrics = None
    epoch_records = []

    # 瀹炴椂鍐?epoch 鏇茬嚎 (閬垮厤宕╂簝涓㈠け)
    epoch_csv = os.path.join(args.results_dir, f"epoch_curve_fold{fold}.csv")

    # 鐏垫椿鍋滄锛氭棭鍋?patience + 鎵嬪姩涓柇 (Ctrl-C) 瀹夊叏钀界洏
    es_patience = int(getattr(args, "early_stop_patience", 0))
    es_min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
    es_metric = getattr(args, "early_stop_metric", "val_cindex")
    es_warmup = int(getattr(args, "early_stop_warmup", 0))
    es_best = -1e9
    es_bad_epochs = 0
    stopped_epoch = args.max_epochs - 1

    def _save_best(results, model, fold):
        safe_torch_save(
            model.state_dict(),
            os.path.join(args.results_dir, f"model_best_s{fold}.pth")
        )
        safe_pickle_dump(results, os.path.join(args.results_dir, f"split_{fold}_results.pkl"))

    try:
        for epoch in range(args.max_epochs):
            train_one_epoch(args, epoch, model, train_loader,
                            optimizer, scheduler, loss_fn, log_file)
            safe_flush(log_file)
            results, val_c, val_c_ipcw, val_BS, val_IBS, val_iauc, val_loss = evaluate(
                args, dataset_factory, model, val_loader, loss_fn, survival_train
            )
            msg = (f"[Epoch {epoch}] val cindex={val_c:.4f} ipcw={val_c_ipcw:.4f} "
                   f"IBS={val_IBS:.4f} iauc={val_iauc:.4f}")
            print(msg)
            safe_write_line(log_file, msg)
            safe_flush(log_file)
            epoch_records.append({
                'epoch': epoch,
                'val_cindex': val_c,
                'val_cindex_ipcw': val_c_ipcw,
                'val_IBS': val_IBS,
                'val_iauc': val_iauc,
                'val_loss': val_loss,
            })
            ensure_min_free_space(args.results_dir, args.min_free_space_gb, f"Fold {fold} epoch {epoch} before write")
            # 姣忎釜 epoch 閮借鐩栧啓涓€娆?csv (閬垮厤宕╂簝涓㈠け鏇茬嚎)
            safe_to_csv(epoch_records, epoch_csv)

            if val_c >= args.max_cindex:
                args.max_cindex = val_c
                args.max_cindex_epoch = epoch
                best_results = results
                _save_best(results, model, fold)
                final_metrics = (val_c, val_c_ipcw, val_BS, val_IBS, val_iauc, val_loss)

            # ---- 鏃╁仠鍒ゅ畾锛坧atience>0 鏃跺惎鐢紝涓旇繃浜?warmup锛?---
            if es_patience > 0:
                cur = {'val_cindex': val_c, 'val_cindex_ipcw': val_c_ipcw,
                       'val_iauc': val_iauc}.get(es_metric, val_c)
                if cur > es_best + es_min_delta:
                    es_best = cur
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                if epoch >= es_warmup and es_bad_epochs >= es_patience:
                    stopped_epoch = epoch
                    msg = (
                        f"[Fold {fold}] early stop @epoch {epoch} "
                        f"({es_metric} did not improve for {es_patience} epochs)"
                    )
                    print(msg)
                    safe_write_line(log_file, msg)
                    break
    except KeyboardInterrupt:
        stopped_epoch = len(epoch_records) - 1
        msg = (
            f"[Fold {fold}] interrupted @epoch {stopped_epoch}; "
            f"best cindex={args.max_cindex:.4f} @epoch {args.max_cindex_epoch} was preserved"
        )
        print(msg)
        safe_write_line(log_file, msg)
        safe_flush(log_file)
        safe_to_csv(epoch_records, epoch_csv)

    msg = (f"[Fold {fold}] best cindex={args.max_cindex:.4f} "
           f"@epoch {args.max_cindex_epoch} (stopped @epoch {stopped_epoch})")
    print(msg)
    safe_write_line(log_file, msg)
    safe_flush(log_file)

    print(f"[Fold {fold}] epoch curve 宸蹭繚瀛? {epoch_csv}")

    # 娓呯悊
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_results, final_metrics


# ============================================================
# 涓诲叆鍙?# ============================================================

def run(args):
    # 0. 鍏ㄥ眬绉嶅瓙澶嶄綅 (淇: --seed 涔嬪墠娌＄敤, 涓ゆ璁粌涓嶅彲澶嶇幇)
    seed = getattr(args, 'seed', 3)
    set_global_seed(seed)
    print(f"[run] set_global_seed({seed}) done")

    # 1. 鍑嗗瀹為獙鐩綍
    args.method = f"SurvOTRank_{args.survot_method}"
    # 鍘熷 SlotSPE 鐢?os.mkdir 涓嶉€掑綊, 杩欓噷鍏堢‘淇濇牴鐩綍瀛樺湪
    os.makedirs(args.results_dir, exist_ok=True)
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, "before training start")
    args = _prepare_for_experiment(args)
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, "after experiment directory initialization")

    # 2. 临床特征列解析
    clinical_feature_cols = None
    if getattr(args, "clinical_feature_cols", None):
        clinical_feature_cols = [c.strip() for c in args.clinical_feature_cols.split(",") if c.strip()]

    # 3. 数据集工厂 (一次构建，跨 fold 重用)
    dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches,
        clinical_feature_cols=clinical_feature_cols,
    )

    # 传递临床模态开关给模型
    if clinical_feature_cols and len(clinical_feature_cols) > 0:
        args.otehv2v2_use_clinical = True
        args.otehv2v2_clinical_feature_dim = len(clinical_feature_cols)

    # 4. 过滤掉 RNA 表里没有的 case_id (避免训练时 KeyError)
    if args.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
        rna_cases = set(dataset_factory.gene_data_df.columns)
        before = len(dataset_factory.clinical_df)
        dataset_factory.clinical_df = dataset_factory.clinical_df[
            dataset_factory.clinical_df['case id'].isin(rna_cases)
        ].reset_index(drop=True)
        after = len(dataset_factory.clinical_df)
        if before != after:
            print(f"[filter] 杩囨护 RNA 缂哄け鐨?case: {before} -> {after}")

    # 3. fold 鑼冨洿
    folds = _get_start_end(args)

    # 4. log
    log_path = os.path.join(args.results_dir,
                            f"log_start_{args.k_start}_end_{args.k_end}.txt")
    log_file = open(log_path, "w", buffering=1)  # 琛岀紦鍐?
    all_metrics = []
    for fold in folds:
        try:
            results, metrics = train_one_fold(args, dataset_factory, fold, log_file)
            safe_flush(log_file)
        except Exception as e:
            print(f"[ERROR] fold {fold} 澶辫触: {e}")
            safe_write_line(log_file, f"[ERROR] fold {fold} 澶辫触: {e}")
            safe_write_line(log_file, traceback.format_exc())
            safe_flush(log_file)
            continue

        if metrics is not None:
            all_metrics.append((fold, *metrics))
        # 淇濆瓨 final 缁撴灉
        if results is not None:
            safe_pickle_dump(results, os.path.join(args.results_dir, f"split_{fold}_results_final.pkl"))

    safe_flush(log_file)
    try:
        log_file.close()
    except OSError:
        pass

    # 5. 姹囨€?    if all_metrics:
        df = pd.DataFrame(all_metrics, columns=[
            "fold", "val_cindex", "val_cindex_ipcw", "val_BS",
            "val_IBS", "val_iauc", "val_loss"
        ])
        # 鍒犻櫎闈炴爣閲忓垪 (val_BS 鏄暟缁? 鏃犳硶 mean)
        if 'val_BS' in df.columns:
            df = df.drop(columns=['val_BS'])
        df.set_index("fold", inplace=True)
        try:
            df.loc["mean"] = df.mean(numeric_only=True)
            df.loc["std"] = df.std(numeric_only=True)
        except Exception as e:
            print(f"[summary] mean/std 澶辫触: {e}")
        save_name = "summary.csv" if len(folds) == args.k \
            else f"summary_partial_{args.k_start}_{args.k_end}.csv"
        safe_to_csv(df.reset_index(), os.path.join(args.results_dir, save_name))
        print(f"\n[Summary]")
        print(df)


def main():
    start = time.time()
    args = process_args_extended()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run(args)
    end = time.time()
    print(f"\nDone. Time: {end - start:.1f}s")


if __name__ == "__main__":
    main()
