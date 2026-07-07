#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy model initialization stub.

The SlotSPE baseline was intentionally removed from this project. This module is
kept only so old imports fail with a clear message instead of an ImportError.
"""


def _init_model(args, dataset_factory):
    raise RuntimeError(
        "SlotSPE baseline initialization has been removed. "
        "Use survot_rank.training.model_factory with the PET method instead."
    )
