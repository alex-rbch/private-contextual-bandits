"""Reuse local datasets or download missing datasets by OpenML ID."""
from __future__ import annotations

from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def prepare_input(dataset_id, directory, *, cache_directory=None):
    """Reuse local inputs or download once; never overwrite existing data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archives = sorted(directory.glob(f"ds_{dataset_id}_*.npz"))
    metadata = sorted(directory.glob(f"ds_{dataset_id}_*.source.json"))
    if archives or metadata:
        if len(archives) != 1 or len(metadata) != 1 or archives[0].with_suffix(".source.json") != metadata[0]:
            raise ValueError(f"OpenML {dataset_id}: require one complete archive/metadata pair")
        return archives[0]
    from .openml import download
    import openml
    openml.config.set_root_cache_directory(str(cache_directory or ROOT / ".cache" / "openml"))
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as temporary:
        temporary = Path(temporary)
        prepared = download(dataset_id, temporary, min_rows=1)
        destination = directory / prepared.name
        sidecar = destination.with_suffix(".source.json")
        if destination.exists() or sidecar.exists():
            raise FileExistsError("dataset appeared during preparation; prepare inputs once")
        prepared.with_suffix(".source.json").rename(sidecar)
        prepared.rename(destination)
    return destination
