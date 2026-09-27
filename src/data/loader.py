"""Load NumPy datasets and generate reproducible, in-memory permutations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .numpy_dataset import load_numpy_problem
from .preprocessing import prepare_contexts


DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: int
    num_actions: int
    contexts: np.ndarray
    labels: np.ndarray
    raw_context_dimension: int
    source_path: Path
    use_bias: bool
    project_unitball: bool


def load_dataset(
    dataset_id: int,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    input_path: Path | None = None,
    use_bias: bool = True,
    project_unitball: bool = True,
) -> LoadedDataset:
    """Load one dataset by OpenML ID and apply the requested preprocessing.

    With the defaults, action blocks formed from ``contexts`` have the same
    entries as the earlier ``e_a tensor (1, project(x))/sqrt(2)`` map.
    """

    if dataset_id <= 0:
        raise ValueError("dataset_id must be positive")
    if input_path is None:
        matches = sorted(Path(datasets_dir).glob(f"ds_{dataset_id}_*.npz"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one NumPy dataset for OpenML {dataset_id}; "
                f"found {len(matches)} in {datasets_dir}"
            )
        source_path = matches[0]
    else:
        source_path = Path(input_path)
    metadata_path = source_path.with_suffix(".source.json")
    if not metadata_path.is_file():
        raise ValueError(f"missing source metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if "feature_hashing" in metadata:
        raise ValueError("feature-hashed datasets are not allowed")
    if int(metadata["openml_dataset_id"]) != dataset_id:
        raise ValueError("dataset ID disagrees with source metadata")
    num_actions = int(metadata["num_actions"])
    raw, labels = load_numpy_problem(source_path, num_actions)
    if raw.shape != (int(metadata["rows"]), int(metadata["context_dimension"])):
        raise ValueError("NumPy dimensions disagree with source metadata")
    return LoadedDataset(
        dataset_id=dataset_id,
        num_actions=num_actions,
        contexts=prepare_contexts(
            raw, use_bias=use_bias, project_unitball=project_unitball
        ),
        labels=labels,
        raw_context_dimension=raw.shape[1],
        source_path=source_path,
        use_bias=use_bias,
        project_unitball=project_unitball,
    )


def shuffle_dataset(
    contexts: np.ndarray,
    labels: np.ndarray,
    *,
    horizon: int,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a seeded permutation prefix and its original row indices.

    There is no replacement and no shuffled copy is written to disk.
    """

    source_rows = int(contexts.shape[0])
    if labels.shape != (source_rows,):
        raise ValueError("contexts and labels must have the same row count")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if source_rows < horizon:
        raise ValueError(
            f"source dataset has {source_rows} rows; horizon {horizon} "
            "requires sampling with replacement, which is disabled"
        )
    indices = np.random.default_rng(shuffle_seed).permutation(source_rows)[:horizon]
    return contexts[indices], labels[indices], indices


def generate_shuffled_datasets(
    dataset: LoadedDataset,
    *,
    count: int,
    seed_base: int,
    horizon: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``count`` reproducible NumPy streams, without saved VW files."""

    if count <= 0:
        raise ValueError("count must be positive")
    stream_length = len(dataset.labels) if horizon is None else horizon
    for offset in range(count):
        yield shuffle_dataset(
            dataset.contexts,
            dataset.labels,
            horizon=stream_length,
            shuffle_seed=seed_base + offset,
        )
