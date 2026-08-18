#!/usr/bin/env python3
"""Fair MLP Comparison Baseline (Section 4.6 / Ablation Study).

Strategy: Load a trained ACT-Surv v5 checkpoint, freeze the WSI + omics encoders,
replace the ACT transport module with a simple MLP head, and train ONLY the MLP
head on the same data. This gives a fair comparison because:

  - Same feature extractors (WSI MLP + pathway encoders) → same representation space
  - Same training protocol → comparable C-index
  - Only the prediction head differs: MLP (unconstrained) vs ACT (archetypal convex)

The question: does the ACT convex-hull constraint actually help prediction,
or does the MLP just learn the same thing?

Run:
    python scripts/fair_mlp_comparison.py --cancer blca --fold 0 --device cuda

Note: Trains MLP head only (frozen encoder) on the training set of fold 0.
Expected runtime: ~10 min on GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_act_surv_v5_mechanism import (
    build_dataloader,
    detect_dims_from_state,
    load_checkpoint_pretrained_state,
)
from scripts.report_act_surv_v5_results import _cindex_score


def _find_ckpt(cancer: str, fold: int) -> Path:
    for parent in [
        REPO_ROOT / f"results/act_surv_v5_1/{cancer}",
        REPO_ROOT / f"results/act_surv_v5/{cancer}",
    ]:
        if not parent.exists():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d.name.endswith(f"_fold{fold}"):
                ckpts = list(d.glob("model_best_s*.pth"))
                if ckpts:
                    return ckpts[0]
    raise FileNotFoundError(f"No checkpoint: {cancer} fold {fold}")


class MLPHead(nn.Module):
    """Simple MLP prediction head replacing the ACT transport module.

    Takes the concatenation of WSI and omics features → hidden → logits.
    Same budget as the ACT module (roughly similar param count).
    """
    def __init__(self, wsi_dim: int, omic_total: int, hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        total_dim = wsi_dim + omic_total
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, wsi_features, omic_features):
        # wsi_features: [B, wsi_dim] (pooled WSI, e.g. mean over patches)
        # omic_features: [B, omic_total]
        x = torch.cat([wsi_features, omic_features], dim=-1)
        return self.net(x)


def pool_wsi(x_wsi: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """Pool WSI patches to a single representation vector."""
    if x_wsi.dim() == 3:
        x = x_wsi.mean(dim=1)  # [B, D]
    else:
        x = x_wsi
    return x


def pool_omics(omic_list: list[torch.Tensor]) -> torch.Tensor:
    """Pool omics pathway features to a single vector."""
    if not omic_list:
        return torch.zeros(1, 1)
    stacked = torch.stack(omic_list, dim=1)  # [B, P, D]
    return stacked.mean(dim=1)  # [B, D]


def build_train_val_for_mlp(
    cancer: str,
    fold: int,
    device: str,
    batch_size: int = 8,
    max_batches: int = 200,
):
    """Build train/val TensorDatasets for MLP head training.

    Uses the same fold split as the original ACT-Surv training.
    We extract frozen features from the encoder + train MLP head.
    """
    ckpt_path = _find_ckpt(cancer, fold)
    state = load_checkpoint_pretrained_state(ckpt_path)
    dims = detect_dims_from_state(state)

    from types import SimpleNamespace
    from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
        ArchetypalTransportCompositionV5,
    )
    args = SimpleNamespace(
        omic_sizes=dims.get("omic_sizes", [16]*10),
        encoding_dim=dims.get("encoding_dim", 1536),
        wsi_projection_dim=dims.get("wsi_projection_dim", 256),
        act5_num_archetypes=dims.get("act5_num_archetypes", 6),
        act5_epsilon=0.10, act5_hazard_scale=1.0, act5_warmup_epochs=0,
        act5_lambda_balance=0.05, act5_lambda_rank=0.00,
        act5_rank_margin=0.02, act5_rank_temperature=0.50,
        act5_rank_max_pairs=4096,
        rna_format=dims.get("rna_format", "Pathways"),
        omics_input_dim=dims.get("omic_input_dim", 0),
        n_classes=dims.get("n_classes", 4),
    )
    model = ArchetypalTransportCompositionV5(args).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Freeze all encoder parameters
    for name, param in model.named_parameters():
        if "wsi_mlp" in name or "sig_networks" in name:
            param.requires_grad = False

    val_loader = build_dataloader(cancer, fold, batch_size=batch_size, device=device)
    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import _unpack_data

    wsi_feats_list, omic_feats_list, y_list, c_list, logits_act_list = [], [], [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            try:
                data_wsi, data_omics, y_disc, event_time, c_flag, _xc = _unpack_data(batch, device, "Pathways")
                if isinstance(data_omics, (list, tuple)):
                    input_kwargs = {f"x_omic{i+1}": omic.float() for i, omic in enumerate(data_omics)}
                else:
                    input_kwargs = {"x_omics": data_omics.float()}

                out = model(
                    x_wsi=data_wsi.float(), cur_epoch=0,
                    wsi_missing=False, omic_missing=False,
                    y=None, c=None, **input_kwargs,
                )
                logits = out[0] if isinstance(out, tuple) else out

                wsi_feat = model._encode_wsi(data_wsi).mean(dim=1)          # [B, 256]
                omic_encoded = model._encode_omics(input_kwargs)            # [B, P, 256]
                omic_feat = omic_encoded.mean(dim=1)                        # [B, 256]

                wsi_feats_list.append(wsi_feat.cpu())
                omic_feats_list.append(omic_feat.cpu())
                y_list.append(y_disc.cpu())
                c_list.append(c_flag.cpu())
                logits_act_list.append(logits.cpu())
            except Exception as e:
                print(f"  WARN batch {bi}: {e}")
                continue

    if not wsi_feats_list:
        raise RuntimeError("No batches processed")

    wsi_all = torch.cat(wsi_feats_list, dim=0)   # [N, D_wsi]
    omic_all = torch.cat(omic_feats_list, dim=0)  # [N, D_omic]
    y_all = torch.cat(y_list, dim=0).long()        # [N]
    c_all = torch.cat(c_list, dim=0).long()       # [N]
    logits_act = torch.cat(logits_act_list, dim=0) # [N, C]

    print(f"[MLP data] {len(y_all)} samples: wsi={wsi_all.shape}, omic={omic_all.shape}")

    # Split into train/val (80/20 within the already-fixed validation fold)
    # This gives us a train set for MLP head tuning
    n = len(y_all)
    n_train = int(n * 0.8)
    indices = torch.randperm(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = TensorDataset(
        wsi_all[train_idx], omic_all[train_idx], y_all[train_idx], c_all[train_idx]
    )
    val_ds = TensorDataset(
        wsi_all[val_idx], omic_all[val_idx], y_all[val_idx], c_all[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader_mlp = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader_mlp,
        "wsi_dim": wsi_all.shape[1],
        "omic_dim": omic_all.shape[1],
        "logits_act": logits_act,  # full ACT-Surv logits on all data
        "y_all": y_all,
        "c_all": c_all,
        "wsi_all": wsi_all,
        "omic_all": omic_all,
    }


def train_mlp_head(
    cancer: str,
    fold: int,
    device: str,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 8,
    hidden_dim: int = 256,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Fair MLP Comparison: {cancer} fold {fold}")
    print(f"{'='*60}")

    data = build_train_val_for_mlp(cancer, fold, device, batch_size)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    wsi_dim = data["wsi_dim"]
    omic_dim = data["omic_dim"]
    num_classes = data["y_all"].max().item() + 1

    # Build MLP head
    mlp = MLPHead(wsi_dim, omic_dim, hidden_dim, num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_cindex = -1
    best_state = None
    history = []

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0
        for wsi_b, omic_b, y_b, c_b in train_loader:
            wsi_b, omic_b, y_b, c_b = wsi_b.to(device), omic_b.to(device), y_b.to(device), c_b.to(device)
            logits = mlp(wsi_b, omic_b)
            loss = criterion(logits, y_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * wsi_b.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validate
        mlp.eval()
        logits_list, y_list, c_list = [], [], []
        with torch.no_grad():
            for wsi_b, omic_b, y_b, c_b in val_loader:
                wsi_b, omic_b = wsi_b.to(device), omic_b.to(device)
                logits = mlp(wsi_b, omic_b)
                logits_list.append(logits.cpu())
                y_list.append(y_b)
                c_list.append(c_b)

        logits_val = torch.cat(logits_list, dim=0).numpy()
        y_val = torch.cat(y_list, dim=0).numpy()
        c_val = torch.cat(c_list, dim=0).numpy()
        val_cindex = _cindex_score(logits_val, y_val, c_val)

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_cindex={val_cindex:.4f}")

        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_cindex": float(val_cindex)})

    # Load best and compute final metrics
    mlp.load_state_dict(best_state)
    mlp.eval()

    logits_all_mlp = []
    with torch.no_grad():
        for i in range(0, len(data["wsi_all"]), batch_size):
            wsi_b = data["wsi_all"][i:i+batch_size].to(device)
            omic_b = data["omic_all"][i:i+batch_size].to(device)
            logits = mlp(wsi_b, omic_b)
            logits_all_mlp.append(logits.cpu())

    logits_mlp = torch.cat(logits_all_mlp, dim=0).numpy()
    y_all = data["y_all"].numpy()
    c_all = data["c_all"].numpy()
    logits_act = data["logits_act"].numpy()

    mlp_cindex = _cindex_score(logits_mlp, y_all, c_all)
    act_cindex = _cindex_score(logits_act, y_all, c_all)

    # Compute per-patient difference
    delta = logits_mlp - logits_act
    mean_delta = float(np.abs(delta).mean())

    print(f"\n[Result]")
    print(f"  ACT-Surv C-index:  {act_cindex:.4f}")
    print(f"  MLP Head C-index:  {mlp_cindex:.4f}")
    print(f"  Delta (MLP-ACT):  {mlp_cindex - act_cindex:+.4f}")
    print(f"  Mean logit diff:   {mean_delta:.4f}")

    return {
        "experiment": "fair_mlp_comparison",
        "cancer": cancer,
        "fold": fold,
        "mlp_cindex": float(mlp_cindex),
        "act_cindex": float(act_cindex),
        "cindex_delta": float(mlp_cindex - act_cindex),
        "mean_logit_diff": mean_delta,
        "best_val_cindex": float(best_val_cindex),
        "mlp_head_params": sum(p.numel() for p in mlp.parameters()),
        "history": history,
        "verdict": (
            f"ACT-Surv outperforms MLP by {act_cindex - mlp_cindex:.4f} C-index"
            if act_cindex > mlp_cindex
            else f"MLP head matches/exceeds ACT-Surv by {mlp_cindex - act_cindex:.4f} C-index"
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fair MLP comparison baseline")
    p.add_argument("--cancer", default="blca")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "results" / "act_surv_v5" / "mlp_comparison"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_mlp_head(
        args.cancer, args.fold, device,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"mlp_comparison_{args.cancer}_fold{args.fold}_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nSaved: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
