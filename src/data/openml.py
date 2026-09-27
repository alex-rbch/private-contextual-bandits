"""Download full numeric OpenML classification datasets as NumPy arrays."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import openml

from .numpy_dataset import numpy_problem_bytes


DEFAULT_MIN_ROWS = 2048


def download(
    dataset_id: int,
    output_directory: Path,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> Path:
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    features, targets, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    matrix = np.asarray(features.to_numpy(), dtype=np.float64)
    target_strings = np.asarray(targets).astype(str)
    classes, labels = np.unique(target_strings, return_inverse=True)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"OpenML {dataset_id} has malformed features")
    if matrix.shape[0] < min_rows:
        raise ValueError(
            f"OpenML {dataset_id} has {matrix.shape[0]} rows; "
            f"at least {min_rows} are required"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"OpenML {dataset_id} is not finite numeric data")
    action_count = int(classes.size)
    regression_dimension = action_count * (matrix.shape[1] + 1)
    if action_count < 2:
        raise ValueError(f"OpenML {dataset_id} is not a classification problem")

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"ds_{dataset_id}_{action_count}"
    destination = output_directory / f"{stem}.npz"
    payload = numpy_problem_bytes(matrix, labels)
    destination.write_bytes(payload)

    counts = Counter(int(label) + 1 for label in labels)
    metadata = {
        "class_counts": {str(key): counts[key] for key in sorted(counts)},
        "class_values_in_encoded_order": classes.tolist(),
        "context_dimension": int(matrix.shape[1]),
        "feature_names": [str(name) for name in features.columns],
        "dataset": dataset.name,
        "num_actions": action_count,
        "openml_dataset_id": dataset_id,
        "openml_url": f"https://www.openml.org/d/{dataset_id}",
        "regression_dimension": regression_dimension,
        "minimum_required_rows": min_rows,
        "rows": int(matrix.shape[0]),
    }
    (output_directory / f"{stem}.source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return destination
