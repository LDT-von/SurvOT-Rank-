#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@file: train.py（45_otehv2_rankevent 独立训练入口）
@desc: V45 (OTEHV2RankEvent) 的自包含训练脚本。

   精简自 common/train_runner.py，去掉了 model_factory（本文件夹只有一个模型，
   不需要工厂）和其余 40+ 个实验方向的分支逻辑。

   依赖:
     - 本文件夹内: model.py / backbone.py / paths.py / args.py
     - 外部: 一份 SlotSPE/ 基础仓库（数据集 / loss / 底层网络层）
             通过 paths.py 自动查找，或用环境变量 SLOTSPE_DIR 指定

   用法:
     python train.py --data_root_dir /path/to/CPathPatchFeature \
                      --data_path ./SlotSPE/dataset_csv \
                      --results_dir ./results --gpu 0

   等价于以前的:
     python common/train_runner.py --newslot_method otehv2_rankevent ...
"""

import os
import sys
import gc
import time
import pickle
import shutil
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from paths import ensure_slotspe_in_path  # noqa
ensure_slotspe_in_path()

# 复用外部 SlotSPE/ 仓库的数据集与训练工具（不复制，避免和其他实验目录的基座脱节）
from dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset, _collate_pathways  # noqa
from utils.loss_func import NLLSurvLoss, SurvPLE, RankLoss, SinkhornSurvLoss  # noqa
from utils.general_utils import (  # noqa
    _get_start_end, _prepare_for_experiment, _save_pkl, _print_network
)
from utils.core_utils import (  # noqa
    _process_data_and_forward, _calculate_risk, _update_arrays,
    _calculate_metrics, _extract_survival_metadata
)
from sksurv.metrics import concordance_index_censored  # noqa

from args import process_args  # noqa
from model import OTEHTimeLocalCompeting  # noqa: 本文件夹的方法模型（V50）


# ============================================================
# 安全写盘工具（磁盘满不崩，只警告）
# ============================================================

def safe_write_line(log_file, message):
    try:
        log_file.write(message + "\n")
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 日志写入失败，磁盘空间不足: {error}")
        else:
            raise


def safe_flush(log_file):
    try:
        log_file.flush()
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 日志 flush 失败，磁盘空间不足: {error}")
        else:
            raise


def safe_to_csv(records, csv_path):
    try:
        pd.DataFrame(records).to_csv(csv_path, index=False)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 跳过写入 {csv_path}，磁盘空间不足: {error}")
            return False
        raise


def safe_pickle_dump(obj, output_path):
    try:
        with open(output_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 跳过写入 {output_path}，磁盘空间不足: {error}")
            return False
        raise


def safe_torch_save(state_dict, output_path):
    try:
        torch.save(state_dict, output_path)
        return True
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            print(f"[WARN] 跳过写入 {output_path}，磁盘空间不足: {error}")
            return False
        raise


def get_free_space_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def ensure_min_free_space(path, min_free_gb, context):
    free_gb = get_free_space_gb(path)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"{context} 可用空间不足: 当前仅剩 {free_gb:.2f}GB，至少需要 {min_free_gb:.2f}GB。"
            f" 请清理目标盘后重试，结果目录: {path}"
        )
    return free_gb


# ============================================================
# 模型 / 优化器 / 数据
# ============================================================

def init_model(args, dataset_factory):
    if args.rna_format == "RNASeq":
        omics_input_dim = (
            dataset_factory.num_genes if dataset_factory.num_genes is not None
            else dataset_factory.omic_sizes
        )
    elif args.rna_format == "GeneEmbedding":
        omics_input_dim = 768
    else:
        omics_input_dim = None

    args.omic_sizes = dataset_factory.omic_sizes
    args.omic_names = dataset_factory.omic_names
    args.pathway_names = getattr(dataset_factory, "pathway_names", None)

    print("[init] 加载模型: OTEHTimeLocalCompeting (V50, 时间局部竞争事件危险率)")
    model = OTEHTimeLocalCompeting(
        args,
        omic_input_dim=omics_input_dim,
        omic_names=args.omic_names,
        pathway_names=args.pathway_names,
    )

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    _print_network(args.results_dir, model)
    return model


def init_loss_function(args):
    if args.bag_loss == "nll_surv":
        return NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == "cox_surv":
        return SurvPLE()
    elif args.bag_loss == "rank_surv":
        return RankLoss()
    elif args.bag_loss == "sinkhorn_surv":
        return SinkhornSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError


def init_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError


def init_scheduler(args, optimizer):
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    elif args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.eta_min
        )
    else:
        raise NotImplementedError


def get_split(args, dataset_factory, fold):
    train_data = SurvivalDataset(dataset_factory, args.data_root_dir, "train", fold, args.encoding_dim)
    test_data = SurvivalDataset(dataset_factory, args.data_root_dir, "val", fold, args.encoding_dim)

    num_workers = getattr(args, "num_workers", 4)
    pin_memory = True

    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True, collate_fn=_collate_pathways, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=_collate_pathways, pin_memory=pin_memory
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    print(f"[data] train={len(train_data)}  val={len(test_data)}  num_workers={num_workers}  pin_memory={pin_memory}")
    return train_data, test_data, train_loader, test_loader


# ============================================================
# 训练 / 评估循环
# ============================================================

def train_one_epoch(args, epoch, model, loader, optimizer, scheduler, loss_fn, log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    args.cur_epoch = epoch

    total_loss = 0.0
    all_risk_scores, all_censorships, all_event_times = [], [], []

    accumulation_steps = 1 if args.batch_size != 1 else 32

    for batch_idx, data in enumerate(loader):
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

        if batch_idx % 10 == 0:
            msg = f"  batch:{batch_idx} loss:{loss.item():.4f} surv:{loss_surv.item():.4f}"
            print(msg)
            safe_write_line(log_file, msg)
        if getattr(args, "max_smoke_batches", 0) > 0 and (batch_idx + 1) >= args.max_smoke_batches:
            msg = f"  [smoke] stop train epoch after {batch_idx + 1} batch(es)"
            print(msg)
            safe_write_line(log_file, msg)
            break

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


def train_one_fold(args, dataset_factory, fold, log_file):
    print(f"\n{'=' * 60}\n[Fold {fold}] start\n{'=' * 60}")
    safe_write_line(log_file, f"\n=== Fold {fold} start ===")
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, f"Fold {fold} 开始前")

    train_data, val_data, train_loader, val_loader = get_split(args, dataset_factory, fold)
    model = init_model(args, dataset_factory)
    loss_fn = init_loss_function(args)
    optimizer = init_optimizer(args, model)
    scheduler = init_scheduler(args, optimizer)

    survival_train = _extract_survival_metadata(dataset_factory)

    args.max_cindex = 0.0
    args.max_cindex_epoch = 0

    best_results = None
    final_metrics = None
    epoch_records = []

    epoch_csv = os.path.join(args.results_dir, f"epoch_curve_fold{fold}.csv")

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
                "epoch": epoch,
                "val_cindex": val_c,
                "val_cindex_ipcw": val_c_ipcw,
                "val_IBS": val_IBS,
                "val_iauc": val_iauc,
                "val_loss": val_loss,
            })
            ensure_min_free_space(args.results_dir, args.min_free_space_gb, f"Fold {fold} Epoch {epoch} 写盘前")
            safe_to_csv(epoch_records, epoch_csv)

            if val_c >= args.max_cindex:
                args.max_cindex = val_c
                args.max_cindex_epoch = epoch
                best_results = results
                _save_best(results, model, fold)
                final_metrics = (val_c, val_c_ipcw, val_BS, val_IBS, val_iauc, val_loss)

            if es_patience > 0:
                cur = {"val_cindex": val_c, "val_cindex_ipcw": val_c_ipcw,
                       "val_iauc": val_iauc}.get(es_metric, val_c)
                if cur > es_best + es_min_delta:
                    es_best = cur
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                if epoch >= es_warmup and es_bad_epochs >= es_patience:
                    stopped_epoch = epoch
                    msg = (f"[Fold {fold}] early stop @epoch {epoch} "
                           f"({es_metric} 连续 {es_patience} 个 epoch 未提升)")
                    print(msg)
                    safe_write_line(log_file, msg)
                    break
    except KeyboardInterrupt:
        stopped_epoch = len(epoch_records) - 1
        msg = (f"[Fold {fold}] 手动中断 @epoch {stopped_epoch}；"
               f"已保存最佳 cindex={args.max_cindex:.4f} @epoch {args.max_cindex_epoch}")
        print(msg)
        safe_write_line(log_file, msg)
        safe_flush(log_file)
        safe_to_csv(epoch_records, epoch_csv)

    msg = (f"[Fold {fold}] best cindex={args.max_cindex:.4f} "
           f"@epoch {args.max_cindex_epoch} (stopped @epoch {stopped_epoch})")
    print(msg)
    safe_write_line(log_file, msg)
    safe_flush(log_file)

    print(f"[Fold {fold}] epoch curve 已保存: {epoch_csv}")

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_results, final_metrics


# ============================================================
# 主入口
# ============================================================

def run(args):
    args.method = "SlotSPE_otehv2_timelocal_competing"
    os.makedirs(args.results_dir, exist_ok=True)
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, "训练启动前")
    args = _prepare_for_experiment(args)
    ensure_min_free_space(args.results_dir, args.min_free_space_gb, "实验目录初始化后")

    dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches,
    )

    if args.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
        rna_cases = set(dataset_factory.gene_data_df.columns)
        before = len(dataset_factory.clinical_df)
        dataset_factory.clinical_df = dataset_factory.clinical_df[
            dataset_factory.clinical_df["case id"].isin(rna_cases)
        ].reset_index(drop=True)
        after = len(dataset_factory.clinical_df)
        if before != after:
            print(f"[filter] 过滤 RNA 缺失的 case: {before} -> {after}")

    folds = _get_start_end(args)

    log_path = os.path.join(args.results_dir, f"log_start_{args.k_start}_end_{args.k_end}.txt")
    log_file = open(log_path, "w", buffering=1)

    all_metrics = []
    for fold in folds:
        try:
            results, metrics = train_one_fold(args, dataset_factory, fold, log_file)
            safe_flush(log_file)
        except Exception as e:
            print(f"[ERROR] fold {fold} 失败: {e}")
            safe_write_line(log_file, f"[ERROR] fold {fold} 失败: {e}")
            safe_write_line(log_file, traceback.format_exc())
            safe_flush(log_file)
            continue

        if metrics is not None:
            all_metrics.append((fold, *metrics))
        if results is not None:
            safe_pickle_dump(results, os.path.join(args.results_dir, f"split_{fold}_results_final.pkl"))

    safe_flush(log_file)
    try:
        log_file.close()
    except OSError:
        pass

    if all_metrics:
        df = pd.DataFrame(all_metrics, columns=[
            "fold", "val_cindex", "val_cindex_ipcw", "val_BS",
            "val_IBS", "val_iauc", "val_loss"
        ])
        if "val_BS" in df.columns:
            df = df.drop(columns=["val_BS"])
        df.set_index("fold", inplace=True)
        try:
            df.loc["mean"] = df.mean(numeric_only=True)
            df.loc["std"] = df.std(numeric_only=True)
        except Exception as e:
            print(f"[summary] mean/std 失败: {e}")
        save_name = "summary.csv" if len(folds) == args.k \
            else f"summary_partial_{args.k_start}_{args.k_end}.csv"
        safe_to_csv(df.reset_index(), os.path.join(args.results_dir, save_name))
        print("\n[Summary]")
        print(df)


def main():
    start = time.time()
    args = process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run(args)
    end = time.time()
    print(f"\nDone. Time: {end - start:.1f}s")


if __name__ == "__main__":
    main()
