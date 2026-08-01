#!/usr/bin/env python3
"""
Generate and audit 5-fold stratified CV splits for any TCGA study.

Usage:
    python gen_splits_5fold.py --study brca
    python gen_splits_5fold.py --study brca --label_col survival_months_os
    python gen_splits_5fold.py --study luad --n_folds 5 --seed 42
    python gen_splits_5fold.py --study brca --audit_only

Output format:
    train,val
    TCGA-XX-XXXX,TCGA-YY-YYYY
    ...

Stratification key: event status x within-status survival-time quantile.

Computing time quantiles separately for observed-event and censored patients
prevents sparse cohorts from placing most events in only one or two folds.
"""
import argparse
import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv"
TCGA_CASE_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def collect_feature_stems(feature_dir: str | Path) -> set[str]:
    """Return filename stems for every supported extracted WSI feature."""
    root = Path(feature_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"missing WSI feature directory: {root}")

    stems: set[str] = set()
    for pattern in ("*.pt", "*.h5", "*.hdf5"):
        stems.update(feature_path.stem for feature_path in root.rglob(pattern))
    if not stems:
        raise RuntimeError(f"no WSI feature files found under: {root}")
    return stems


def collect_feature_case_ids(feature_dir: str | Path) -> set[str]:
    """Return TCGA case IDs represented by supported WSI feature files."""
    case_ids: set[str] = set()
    for stem in collect_feature_stems(feature_dir):
        match = TCGA_CASE_PATTERN.search(stem)
        if match:
            case_ids.add(match.group(1).upper())
    if not case_ids:
        raise RuntimeError(f"no TCGA case IDs found under: {feature_dir}")
    return case_ids


def collect_matching_feature_case_ids(
    clinical_csv: str | Path,
    feature_dir: str | Path,
) -> set[str]:
    """Return cases with at least one clinical slide that resolves to a feature."""
    clinical = pd.read_csv(clinical_csv)
    required = {"case id", "wsi"}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"clinical CSV is missing columns: {sorted(missing)}")
    stems = collect_feature_stems(feature_dir)
    matched: set[str] = set()
    for row in clinical.loc[:, ["case id", "wsi"]].dropna().itertuples(index=False):
        case_id, slides = str(row[0]).upper(), str(row[1])
        slide_stems = {
            slide.strip().removesuffix(".svs")
            for slide in slides.split(",")
            if slide.strip()
        }
        if slide_stems & stems:
            matched.add(case_id)
    return matched


def make_strat_key(
        df: pd.DataFrame,
        label_col: str,
        censor_col: str,
        n_folds: int = 5,
        n_time_bins: int = 4):
    """Return (strat_key_array, sub_df_with_case_id_and_strat)."""
    sub = df[['case id', label_col, censor_col]].dropna().copy()
    duplicated = sub[sub.duplicated(subset=['case id'], keep=False)]
    if not duplicated.empty:
        conflicting = duplicated.groupby('case id')[[label_col, censor_col]].nunique()
        conflicting = conflicting[(conflicting[label_col] > 1) | (conflicting[censor_col] > 1)]
        if not conflicting.empty:
            raise ValueError(
                "Conflicting survival labels for duplicate case IDs: "
                f"{conflicting.index.tolist()[:10]}"
            )
        sub = sub.drop_duplicates(subset=['case id'], keep='first')

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_time_bins < 1:
        raise ValueError("n_time_bins must be positive")

    sub['event'] = (sub[censor_col] == 0).astype(int)
    sub['time_q'] = 0
    for _, group in sub.groupby('event', sort=True):
        # Every joint stratum must contain at least one case per fold.
        # Ranking only resolves tied follow-up times; it does not change their
        # temporal ordering.
        group_bins = min(n_time_bins, max(1, len(group) // n_folds))
        if group_bins == 1:
            time_q = pd.Series(0, index=group.index, dtype=int)
        else:
            ranked_time = group[label_col].rank(method='first')
            time_q = pd.qcut(
                ranked_time,
                q=group_bins,
                labels=False,
                duplicates='drop',
            ).astype(int)
        sub.loc[group.index, 'time_q'] = time_q

    sub['time_q'] = sub['time_q'].astype(int)
    sub['strat'] = sub['event'].astype(str) + '_' + sub['time_q'].astype(str)
    too_small = sub['strat'].value_counts()
    too_small = too_small[too_small < n_folds]
    if not too_small.empty:
        raise ValueError(
            "Every event/time stratum must have at least one case per fold: "
            f"{too_small.to_dict()}"
        )
    return sub['strat'].values, sub


def audit_existing_splits(
        study: str,
        data_path: str,
        label_col: str,
        censor_col: str,
        n_folds: int,
        split_root: str | None = None,
        eligible_case_ids: set[str] | None = None):
    """Audit patient coverage and joint event/time balance of existing splits."""
    csv = os.path.join(data_path, "clinical", "all", f"{study}.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"missing clinical csv: {csv}")

    df = pd.read_csv(csv, index_col=0)
    clinical_valid = set(
        df.loc[df[label_col].notna() & df[censor_col].notna(), "case id"]
        .astype(str)
        .str.upper()
    )
    if eligible_case_ids is not None:
        eligible_case_ids = {str(case_id).upper() for case_id in eligible_case_ids}
        df = df[df["case id"].astype(str).str.upper().isin(eligible_case_ids)].copy()
    _, cohort = make_strat_key(
        df,
        label_col,
        censor_col,
        n_folds=n_folds,
    )
    expected_cases = set(cohort['case id'].astype(str))
    case_to_strat = cohort.set_index('case id')['strat'].to_dict()
    case_to_event = cohort.set_index('case id')['event'].to_dict()
    target_dir = os.path.join(
        split_root or os.path.join(data_path, "splits", "5fold"),
        study,
    )

    errors = []
    validation_counts = {case_id: 0 for case_id in expected_cases}
    event_counts = []
    stratum_counts = []
    fold_sizes = []
    for fold in range(n_folds):
        split_path = os.path.join(target_dir, f"fold_{fold}.csv")
        if not os.path.isfile(split_path):
            errors.append(f"missing fold file: {split_path}")
            continue
        split = pd.read_csv(split_path)
        missing_columns = {'train', 'val'} - set(split.columns)
        if missing_columns:
            errors.append(
                f"fold_{fold} missing columns: {sorted(missing_columns)}"
            )
            continue

        train = set(split['train'].dropna().astype(str))
        val = set(split['val'].dropna().astype(str))
        invalid = (train | val) - expected_cases
        missing = expected_cases - (train | val)
        overlap = train & val
        if invalid:
            errors.append(
                f"fold_{fold} contains {len(invalid)} ineligible cases"
            )
        if missing:
            errors.append(
                f"fold_{fold} misses {len(missing)} eligible cases"
            )
        if overlap:
            errors.append(
                f"fold_{fold} has {len(overlap)} train/val overlaps"
            )

        eligible_val = val & expected_cases
        for case_id in eligible_val:
            validation_counts[case_id] += 1
        fold_sizes.append(len(eligible_val))
        event_counts.append(sum(case_to_event[case_id] for case_id in eligible_val))
        counts = pd.Series(
            [case_to_strat[case_id] for case_id in eligible_val],
            dtype='object',
        ).value_counts()
        stratum_counts.append(counts.to_dict())

    invalid_frequency = {
        case_id: count
        for case_id, count in validation_counts.items()
        if count != 1
    }
    if invalid_frequency:
        errors.append(
            f"{len(invalid_frequency)} eligible cases do not appear in validation once"
        )

    strata = sorted(set(case_to_strat.values()))
    if len(stratum_counts) == n_folds:
        for stratum in strata:
            counts = [fold.get(stratum, 0) for fold in stratum_counts]
            if max(counts) - min(counts) > 1:
                errors.append(
                    f"stratum {stratum} is imbalanced across folds: {counts}"
                )

    return {
        "study": study,
        "eligible_cases": len(expected_cases),
        "clinical_valid_cases": len(clinical_valid),
        "feature_cases": None if eligible_case_ids is None else len(eligible_case_ids),
        "clinical_without_features": (
            None
            if eligible_case_ids is None
            else len(clinical_valid - eligible_case_ids)
        ),
        "observed_events": int(cohort['event'].sum()),
        "fold_sizes": fold_sizes,
        "validation_event_counts": event_counts,
        "ok": not errors,
        "errors": errors,
    }


def gen(study: str,
        data_path: str,
        label_col: str,
        censor_col: str,
        n_folds: int,
        seed: int,
        out_dir: str,
        eligible_case_ids: set[str] | None = None):
    csv = os.path.join(data_path, "clinical", "all", f"{study}.csv")
    assert os.path.isfile(csv), f"missing clinical csv: {csv}"

    df = pd.read_csv(csv, index_col=0)
    print(f"[{study}] loaded {len(df)} rows from {csv}")

    # Survival training requires both the endpoint and censoring indicator.
    valid_mask = df[label_col].notna() & df[censor_col].notna()
    sub = df.loc[valid_mask].copy()
    if eligible_case_ids is not None:
        eligible_case_ids = {str(case_id).upper() for case_id in eligible_case_ids}
        before = sub["case id"].astype(str).str.upper().nunique()
        sub = sub[
            sub["case id"].astype(str).str.upper().isin(eligible_case_ids)
        ].copy()
        after = sub["case id"].astype(str).str.upper().nunique()
        print(
            f"[{study}] feature-complete cohort: {after}/{before} clinical cases "
            f"({before - after} removed without extracted WSI features)"
        )
    strat_key, sub2 = make_strat_key(
        sub,
        label_col,
        censor_col,
        n_folds=n_folds,
    )
    print(
        f"[{study}] {len(sub2)} unique cases have valid {label_col} "
        f"({len(sub) - len(sub2)} duplicate rows removed)"
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assign = np.zeros(len(sub2), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(sub2, strat_key)):
        fold_assign[val_idx] = fold_idx

    sub2['fold'] = fold_assign

    # Write unequal-length train/val columns without a synthetic CSV index.
    # Using zip(train, val) silently truncated train to the shorter val length.
    target_dir = os.path.join(out_dir, study)
    os.makedirs(target_dir, exist_ok=True)
    n_total = len(sub2)
    print(f"[{study}] writing {n_folds} fold CSVs to {target_dir}")
    expected_cases = set(sub2['case id'])
    validation_counts = {case_id: 0 for case_id in expected_cases}

    for k in range(n_folds):
        train = sub2.loc[sub2['fold'] != k, 'case id'].tolist()
        val = sub2.loc[sub2['fold'] == k, 'case id'].tolist()
        train_set = set(train)
        val_set = set(val)
        if train_set & val_set:
            raise RuntimeError(f"{study} fold_{k} has train/val patient overlap")
        if train_set | val_set != expected_cases:
            raise RuntimeError(f"{study} fold_{k} does not cover the eligible cohort")
        for case_id in val:
            validation_counts[case_id] += 1

        out = pd.DataFrame({
            'train': pd.Series(train, dtype='object'),
            'val': pd.Series(val, dtype='object'),
        })
        out_path = os.path.join(target_dir, f"fold_{k}.csv")
        out.to_csv(out_path, index=False)
        print(f"  fold_{k}.csv : train={len(train)} val={len(val)}")

    invalid_validation_counts = {
        case_id: count for case_id, count in validation_counts.items() if count != 1
    }
    if invalid_validation_counts:
        raise RuntimeError(
            f"{study} cases must appear in validation exactly once: "
            f"{list(invalid_validation_counts.items())[:10]}"
        )

    audit = audit_existing_splits(
        study=study,
        data_path=data_path,
        label_col=label_col,
        censor_col=censor_col,
        n_folds=n_folds,
        split_root=out_dir,
        eligible_case_ids=eligible_case_ids,
    )
    if not audit['ok']:
        raise RuntimeError(
            f"{study} generated split audit failed: {audit['errors']}"
        )
    print(
        f"[{study}] audit OK: val_events={audit['validation_event_counts']} "
        f"fold_sizes={audit['fold_sizes']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--label_col", default="survival_months_dss")
    ap.add_argument("--censor_col", default="censorship_dss")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir")
    ap.add_argument(
        "--feature_dir",
        help=(
            "Optional pt_files directory. When set, splits are generated from "
            "the clinical/feature patient intersection."
        ),
    )
    ap.add_argument(
        "--audit_only",
        action="store_true",
        help="Audit existing split files without rewriting them.",
    )
    args = ap.parse_args()
    eligible_case_ids = None
    if args.feature_dir:
        clinical_csv = (
            Path(args.data_path) / "clinical" / "all" / f"{args.study}.csv"
        )
        eligible_case_ids = collect_matching_feature_case_ids(
            clinical_csv,
            args.feature_dir,
        )
    out_dir = args.out_dir or str(
        Path(args.data_path)
        / "splits"
        / ("5fold_uni2h" if args.feature_dir else "5fold")
    )
    if args.audit_only:
        report = audit_existing_splits(
            study=args.study,
            data_path=args.data_path,
            label_col=args.label_col,
            censor_col=args.censor_col,
            n_folds=args.n_folds,
            split_root=out_dir,
            eligible_case_ids=eligible_case_ids,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report['ok'] else 2)
    gen(
        args.study,
        args.data_path,
        args.label_col,
        args.censor_col,
        args.n_folds,
        args.seed,
        out_dir,
        eligible_case_ids=eligible_case_ids,
    )


if __name__ == "__main__":
    main()
