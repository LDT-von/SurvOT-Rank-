#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: dataset_survival.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/20/24 4:17 PM
'''
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _unpack_data(data, device, omics_format):
    # [img, omic_data_list, label, event_time, c]
    data_wsi = data[0].to(device)

    if omics_format == "Pathways":
        data_omics = []  # TODO: check
        for idx, item in enumerate(data[1]):
            for idy, omic in enumerate(item):
                omic = omic.to(device)
                omic = omic.unsqueeze(0)
                if idx == 0:
                    data_omics.append(omic)
                else:
                    data_omics[idy] = torch.cat((data_omics[idy], omic), dim=0)
    else:
        data_omics = data[1].to(device)

    y_disc = data[2].to(device)
    event_time = data[3].to(device)
    c = data[4].to(device)

    return data_wsi, data_omics, y_disc, event_time, c

SIGNATURES = ["all", "six", "hallmarks", "combine", "xena"]
RNA_FORMATS = ["RNASeq", "Pathways", "GeneEmbedding"]

# six: 2979 genes,
# combine: 9076 genes,
# hallmarks: 6658 genes,
# xena: 2418
# all: 17518 genes

class SurvivalDatasetFactory:
    def __init__(self,
                 study,
                 data_path,
                 rna_format,
                 label_col,
                 signature="all",
                 n_bins=4,
                 eps=1e-6,
                 num_patches=4096,
                 num_genes=None,
                 clinical_feature_cols=None,
                 binning_mode="global_qcut"):
        self.study = study
        self.data_path = data_path
        self.signature = signature
        if self.signature not in SIGNATURES:
            raise ValueError(f"Invalid signature: {self.signature}")
        self.rna_format = rna_format
        if self.rna_format not in RNA_FORMATS:
            raise ValueError(f"Invalid RNA format: {self.rna_format}")
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.num_genes = num_genes
        self.eps = eps
        self.label_col = label_col
        self.clinical_feature_cols = clinical_feature_cols
        self.use_clinical_modality = clinical_feature_cols is not None and len(clinical_feature_cols) > 0
        self.binning_mode = binning_mode  # "global_qcut" (default) or "legacy_equal_width"

        if self.label_col == "survival_months_os":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship_os"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"
        elif self.label_col == "survival_months_dfs": #TODO: donot support this
            self.survival_endpoint = "DFS"
            self.censorship_var = "censorship_dfs"

        # ---> process gene expression data
        self._setup_gene_data()  # self.omics_names, self.omic_sizes,self.gene_embedding_df,self.gene_data_df

        # ---> process clinical data
        self._setup_clinical_data()  # self.clinical_df, self.bins

    def _setup_clinical_data(self):
        clinical_path = os.path.join(self.data_path, "clinical", "all", f"{self.study}.csv")
        clinical_df = pd.read_csv(clinical_path)
        base_cols = ["case id", self.label_col, self.censorship_var, "wsi"]
        if self.use_clinical_modality:
            cols = base_cols + list(self.clinical_feature_cols)
        else:
            cols = base_cols
        clinical_df = clinical_df[cols]

        # Encode categorical clinical features
        if self.use_clinical_modality:
            for col in self.clinical_feature_cols:
                if clinical_df[col].dtype == "object":
                    clinical_df[col] = pd.Categorical(clinical_df[col]).codes

        self.clinical_df = clinical_df.dropna()
        # reindex the clinical data
        self.clinical_df = self.clinical_df.reset_index(drop=True)

        # discretize the label
        # The training runner refits these labels from each fold's training split.
        if self.binning_mode == "legacy_equal_width":
            self._legacy_disc_label(self._get_uncensored_data())
        else:
            self._disc_label(self._get_uncensored_data())

    def _get_uncensored_data(self):
        uncensored_df = self.clinical_df[self.clinical_df[self.censorship_var] < 1]
        return uncensored_df

    def _legacy_disc_label(self, df):
        """Original SlotSPE equal-width binning via pd.cut(bins=4).

        The original code first called pd.qcut, then immediately overwrote the
        result with pd.cut(bins=self.n_bins).  Since n_bins is an integer (4),
        pd.cut divides the full survival time range into 4 equal-width intervals.

        Bin boundaries are computed from *df* so that the caller controls which
        subset defines the range (all uncensored in __init__; training-fold
        uncensored in fit_label_bins).  The resulting boundaries are then applied
        to every row in self.clinical_df.
        """
        _, bins = pd.cut(
            df[self.label_col],
            bins=self.n_bins,
            retbins=True,
            labels=False,
        )
        disc_labels = pd.cut(
            self.clinical_df[self.label_col],
            bins=bins,
            labels=False,
        )
        disc_labels = disc_labels.fillna(0).clip(0, self.n_bins - 1)
        disc_labels = disc_labels.fillna(0).clip(0, self.n_bins - 1)
        self.clinical_df["label"] = disc_labels.values.astype(int)
        self.bins = bins

    def _disc_label(self, uncensored_df):
        """Fit train-only quantile bins and apply them to all clinical rows."""
        values = uncensored_df[self.label_col].dropna()
        if values.nunique() < self.n_bins:
            raise ValueError(
                f"Cannot fit {self.n_bins} survival bins from "
                f"{values.nunique()} unique uncensored times"
            )

        _, q_bins = pd.qcut(
            values, q=self.n_bins, retbins=True, labels=False,
            duplicates="drop",
        )
        if len(q_bins) - 1 != self.n_bins:
            raise ValueError(
                f"qcut produced {len(q_bins) - 1} bins; expected {self.n_bins}"
            )

        q_bins = q_bins.astype(float)
        q_bins[0] = -np.inf
        q_bins[-1] = np.inf
        disc_labels = pd.cut(
            self.clinical_df[self.label_col], bins=q_bins, labels=False,
            right=False, include_lowest=True,
        )
        if disc_labels.isna().any():
            raise ValueError("Survival binning produced NaN labels")

        self.clinical_df["label"] = disc_labels.astype("int64").to_numpy()
        self.bins = q_bins

    def fit_label_bins(self, train_case_ids):
        """Fit discrete-time labels using only uncensored training cases."""
        train_case_ids = set(train_case_ids)
        train_df = self.clinical_df[
            self.clinical_df["case id"].isin(train_case_ids)
        ]
        if train_df.empty:
            raise ValueError("Training split is empty; cannot fit survival bins")
        uncensored = train_df[train_df[self.censorship_var] < 1]
        if getattr(self, "binning_mode", "global_qcut") == "legacy_equal_width":
            self._legacy_disc_label(uncensored)
        else:
            self._disc_label(uncensored)

    def _setup_signatures(self, rna_data_df):
        if self.signature == "six":
            signature_path = os.path.join(self.data_path, "signatures", f"signatures.csv")
        elif self.signature == "all": #TODO: check this
            signature_path = os.path.join(self.data_path, "signatures", f"combine_signatures.csv")
        else:
            signature_path = os.path.join(self.data_path, "signatures", f"{self.signature}_signatures.csv")

        signature_df = pd.read_csv(signature_path)

        self.omic_names = []
        self.pathway_names = []  # only keep pathways with non-empty omic sets

        for col in signature_df.columns:
            omic = signature_df[col].dropna().unique()
            omic = sorted(set(omic).intersection(set(rna_data_df.index)))

            if len(omic) == 0:
                continue  # skip empty omics

            self.omic_names.append(omic)
            self.pathway_names.append(col)  # keep corresponding pathway name only

        self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("pathway size: ", len(self.omic_sizes))


    def _setup_gene_embeddings(self):
        gane_embedding_path = os.path.join(self.data_path, "gene_embedding_inter", f"genes_embedding_768.csv")
        self.gene_embedding_df = pd.read_csv(gane_embedding_path, index_col=0)


    def _setup_gene_data(self):
        rna_file = os.path.join(self.data_path, "raw_rna_data_inter", f"{self.study}_rna_inter.csv")
        rna_data_df = pd.read_csv(rna_file, index_col=0)
        self.gene_data_df = rna_data_df
        self._setup_signatures(rna_data_df)
        if self.rna_format == "RNASeq":
            if self.signature != "all": # flatten the self.omic_names
                self.omic_names = [item for sublist in self.omic_names for item in sublist]
                self.gene_data_df = self.gene_data_df.loc[self.omic_names]
            self.omic_sizes = self.gene_data_df.shape[0]
            self.gene_embedding_df = None
        elif self.rna_format == "Pathways":
            self.gene_embedding_df = None
        elif self.rna_format == "GeneEmbedding":
            self._setup_gene_embeddings()
            if self.signature != "all": # flatten the self.omic_names
                self.omic_names = [item for sublist in self.omic_names for item in sublist]
                self.gene_data_df = self.gene_data_df.loc[self.omic_names]
                self.gene_embedding_df = self.gene_embedding_df.loc[self.omic_names]
            self.omic_sizes = self.gene_data_df.shape[0]
            print("gene embedding shape: ", self.gene_embedding_df.shape)
        else:
            raise ValueError(f"Invalid RNA format: {self.rna_format}")


    def _print_info(self):
        print("Study: ", self.study)
        print("Signature: ", self.signature)
        print("RNA format: ", self.rna_format)
        print("Label column: ", self.label_col)
        print("Number of bins: ", self.n_bins)
        print("Number of patches: ", self.num_patches)
        print("Number of genes: ", self.num_genes)
        print("Censorship variable: ", self.censorship_var)
        print("omic sizes: ", self.omic_sizes) # length of the genes
        # print("omic names: ", self.omic_names)
        if self.rna_format == "Pathways":
            print("pathway names: ", self.pathway_names)


class SurvivalDataset(Dataset):
    def __init__(self, dataset_factory, wsi_path, split_key: str = 'train', fold=None, encoding_dim=768):
        self.dataset_factory = dataset_factory
        # Auto-detect path format: if wsi_path is like /data/CPathPatchFeature/{study}
        # add the missing {uni}/pt_files/ level
        self.wsi_path = wsi_path
        study = dataset_factory.study
        if os.path.isdir(wsi_path) and not os.path.exists(os.path.join(wsi_path, 'TCGA-1.pt')):
            # Try to detect if we need to add path levels
            potential_path = os.path.join(wsi_path, study, 'uni', 'pt_files')
            if os.path.isdir(potential_path):
                self.wsi_path = potential_path
        self.split_key = split_key
        self.fold = fold  # which fold to use
        self.encoding_dim = encoding_dim
        self._wsi_feature_index = None

        if split_key in ['train', 'val']:
            self.label_df = self._load_split()
        else:
            raise ValueError(f"Invalid split key: {split_key}")

    def _load_split(self):
        split_path = os.path.join(self.dataset_factory.data_path, "splits", "5fold", f"{self.dataset_factory.study}",
                                  f"fold_{self.fold}.csv")
        all_splits = pd.read_csv(split_path)
        split = self._get_split_from_df(all_splits, self.split_key)
        return split

    def _get_split_from_df(self, all_splits, split_key: str = 'train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        # change splits to list
        split = split.tolist()

        clinical_df_splits = self.dataset_factory.clinical_df[self.dataset_factory.clinical_df['case id'].isin(split)]

        # reset the index
        clinical_df_splits = clinical_df_splits.reset_index(drop=True)

        return clinical_df_splits

    def _build_wsi_feature_index(self):
        """Index supported feature files once, including nested archive layouts."""
        root = Path(self.wsi_path)
        index = {}
        if root.is_dir():
            for suffix in ("*.pt", "*.h5", "*.hdf5"):
                for feature_path in root.rglob(suffix):
                    index.setdefault(feature_path.stem, feature_path)
        self._wsi_feature_index = index

    def _resolve_wsi_feature_path(self, slide_id):
        stem = str(slide_id).removesuffix(".svs")
        root = Path(self.wsi_path)
        for suffix in (".pt", ".h5", ".hdf5"):
            candidate = root / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        if self._wsi_feature_index is None:
            self._build_wsi_feature_index()
        return self._wsi_feature_index.get(stem)

    def _load_wsi_feature(self, feature_path):
        feature_path = Path(feature_path)
        if feature_path.suffix == ".pt":
            features = torch.load(feature_path, map_location="cpu")
        elif feature_path.suffix in {".h5", ".hdf5"}:
            try:
                import h5py
            except ImportError as error:
                raise ImportError(
                    "Loading UNI2-h .h5 features requires h5py. "
                    "Install the project requirements before training."
                ) from error
            with h5py.File(feature_path, "r") as handle:
                if "features" not in handle:
                    raise KeyError(f"missing HDF5 dataset 'features': {feature_path}")
                features = torch.from_numpy(handle["features"][...])
        else:
            raise ValueError(f"unsupported WSI feature format: {feature_path}")

        if isinstance(features, dict):
            if "features" not in features:
                raise KeyError(f"missing tensor key 'features': {feature_path}")
            features = features["features"]
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        if features.ndim == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(
                f"WSI features must have shape (patches, dim), got "
                f"{tuple(features.shape)} from {feature_path}"
            )
        if features.size(1) != self.encoding_dim:
            raise ValueError(
                f"WSI encoding dimension mismatch for {feature_path}: "
                f"expected {self.encoding_dim}, got {features.size(1)}"
            )
        return features.to(dtype=torch.float32)

    def load_wsi(self, slides):
        if str(slides) == "nan":
            return torch.zeros((1))
        else:
            slide_ids = slides.split(", ")
            wsi = []
            for slide_id in slide_ids:
                feature_path = self._resolve_wsi_feature_path(slide_id)
                if feature_path is not None:
                    wsi.append(self._load_wsi_feature(feature_path))
                else:
                    wsi.append(torch.zeros((self.dataset_factory.num_patches, self.encoding_dim)))
                    print("missing file: ", slide_id)
            wsi = torch.cat(wsi, dim=0).type(torch.float32)  # TODO: check the torch.float32
            return wsi

    def load_genes(self, case_id):
        patient_genes = self.dataset_factory.gene_data_df[case_id]

        if self.dataset_factory.rna_format == "RNASeq":
            patient_genes = torch.from_numpy(patient_genes.values.astype(np.float32))
            return patient_genes
        elif self.dataset_factory.rna_format == "Pathways":
            omic_list = []
            for omic in self.dataset_factory.omic_names:
                omic_data = patient_genes[omic].values
                omic_data = torch.from_numpy(omic_data.astype(np.float32))
                # print(omic_data.size(0))
                # # pad the omic data to the shared size
                # if omic_data.size(0) < 195:
                #     omic_data = torch.cat([omic_data, torch.zeros(195 - omic_data.size(0))], dim=0)
                omic_list.append(omic_data)
            return omic_list
        elif self.dataset_factory.rna_format == "GeneEmbedding":
            patient_genes = torch.from_numpy(patient_genes.values.astype(np.float32))
            rna = patient_genes.unsqueeze(1)
            gene_data = self.dataset_factory.gene_embedding_df.values
            gene_data = torch.from_numpy(gene_data.astype(np.float32))
            gene_embedding = rna * gene_data
            return gene_embedding
        else:
            raise ValueError(f"Invalid RNA format: {self.dataset_factory.rna_format}")

    def get_label(self, case_id):
        label = self.label_df[self.label_df['case id'] == case_id]['label']
        event_time = self.label_df[self.label_df['case id'] == case_id][self.dataset_factory.label_col]
        censorship = self.label_df[self.label_df['case id'] == case_id][self.dataset_factory.censorship_var]
        # convert to tensor
        label = torch.tensor(label.values[0], dtype=torch.long)
        event_time = torch.tensor(event_time.values[0], dtype=torch.float32)
        censorship = torch.tensor(censorship.values[0], dtype=torch.float32)
        return label, event_time, censorship

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, batch_idx):
        case_id = self.label_df.loc[batch_idx, 'case id']
        slides = self.label_df.loc[batch_idx, 'wsi']
        label, event_time, censorship = self.get_label(case_id)
        wsi = self.load_wsi(slides)
        genes = self.load_genes(case_id)

        # Keep train and evaluation inputs at the same token count. Training uses
        # random patches; evaluation uses deterministic evenly spaced patches.
        if self.dataset_factory.num_patches is not None:
            n_samples = min(self.dataset_factory.num_patches, wsi.size(0))
            if self.split_key == 'train':
                patch_idx = np.sort(np.random.choice(wsi.size(0), n_samples, replace=False))
            else:
                patch_idx = np.floor(
                    np.arange(n_samples) * wsi.size(0) / n_samples
                ).astype(np.int64)
            wsi = wsi[patch_idx, :]

            if n_samples < self.dataset_factory.num_patches:
                wsi = torch.cat([wsi, torch.zeros(self.dataset_factory.num_patches - n_samples, wsi.size(1))], dim=0)
        if self.dataset_factory.num_genes is not None and self.split_key == 'train':
            if self.dataset_factory.rna_format != "Pathways":
                n_genes = min(self.dataset_factory.num_genes, genes.size(0))
                gene_idx = np.sort(np.random.choice(genes.size(0), n_genes, replace=False))
                genes = genes[gene_idx]
                if n_genes < self.dataset_factory.num_genes:
                    genes = torch.cat([genes, torch.zeros(self.dataset_factory.num_genes - n_genes)], dim=0)

        if getattr(self.dataset_factory, "use_clinical_modality", False):
            clinical_values = self.label_df.loc[batch_idx, self.dataset_factory.clinical_feature_cols].values.astype(np.float32)
            clinical_tensor = torch.from_numpy(clinical_values)
            return wsi, genes, label, event_time, censorship, clinical_tensor
        return wsi, genes, label, event_time, censorship


def _collate_pathways(batch):

    img = torch.stack([item[0] for item in batch])

    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    result = [img, omic_data_list, label, event_time, c]
    if len(batch[0]) > 5:
        clinical = torch.stack([item[5] for item in batch])
        result.append(clinical)
    return result

if __name__ == '__main__':
    from torch.utils.data import DataLoader, SubsetRandomSampler

    study = "blca"
    data_path = "./dataset_csv"
    rna_format = "Pathways"  # "RNASeq", "Pathways", "GeneEmbedding"
    label_col = "survival_months_dss"
    signature = "combine"

    dataset_factory = SurvivalDatasetFactory(study, data_path, rna_format, label_col, signature, num_genes=None)
    dataset_factory._print_info()

    wsi_path = f"/Data/Pathology/UNI/{study}/pt_files/"
    split_key = 'train'
    fold = 0
    dataset = SurvivalDataset(dataset_factory, wsi_path, split_key, fold)

    if rna_format == "Pathways":
        collate_fn = _collate_pathways
    else:
        collate_fn = None

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # for i, (wsi, genes, label, event_time, censorship) in enumerate(train_loader):
    #     print(wsi.shape, label, event_time, censorship)
        # for gene in genes:
        #     print(gene.shape)

    for i, data in enumerate(train_loader):
        wsi, genes, label, event_time, censorship = _unpack_data(data, device="cpu", omics_format=rna_format)
        # print(wsi.shape, label, event_time, censorship)
        for gene in genes:
            print(gene.shape)

