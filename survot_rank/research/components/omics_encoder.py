"""Small encoders used by SurvOT-Rank methods."""

import torch.nn as nn


def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-normalizing MLP block for omics features."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False),
    )


def WSI_Mlp(dim_in, feat_dim):
    """Projection MLP for WSI patch embeddings."""
    return nn.Sequential(
        nn.Linear(dim_in, dim_in),
        nn.ReLU(inplace=False),
        nn.Linear(dim_in, feat_dim),
    )

