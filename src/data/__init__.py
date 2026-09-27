"""Dataset acquisition, preprocessing, and in-memory shuffling."""

from .loader import (
    LoadedDataset,
    generate_shuffled_datasets,
    load_dataset,
    shuffle_dataset,
)
from .preprocessing import normalize_rows_to_unit_ball, prepare_contexts

__all__ = [
    "LoadedDataset",
    "generate_shuffled_datasets",
    "load_dataset",
    "normalize_rows_to_unit_ball",
    "prepare_contexts",
    "shuffle_dataset",
]
