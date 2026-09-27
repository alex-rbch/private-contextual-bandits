"""Run one algorithm on one NumPy dataset with one public benchmark seed.

The seed fixes the in-memory permutation, action draws, and (when private
mechanisms are simulated) the noise draws. Because the noise seed is public,
private-regime output from this function is *not* a DP release.
"""

from __future__ import annotations

from .data import LoadedDataset, shuffle_dataset
from .oracles import OracleConfiguration
from .policies import PolicyConfiguration
from .protocol import run_batch_protocol, stable_seed


def run_setting(
    dataset: LoadedDataset,
    *,
    policy: PolicyConfiguration,
    loss: str,
    ridge_lambda: float,
    epsilon: float | None,
    delta: float,
    seed: int,
    horizon: int | None = None,
) -> dict[str, object]:
    """Execute one seeded benchmark setting for the config runner."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    stream_length = len(dataset.labels) if horizon is None else horizon
    permutation_seed = stable_seed(seed, "permutation", dataset.dataset_id)
    contexts, labels, indices = shuffle_dataset(
        dataset.contexts,
        dataset.labels,
        horizon=stream_length,
        shuffle_seed=permutation_seed,
    )
    oracle = OracleConfiguration(
        ridge_lambda=ridge_lambda,
        epsilon_reg=4.0 if epsilon is None else epsilon,
        # The non-private oracle does not consume delta; its configuration
        # still requires an interior placeholder value.
        delta_reg=1e-5 if epsilon is None and delta == 0 else delta,
        prefer_primal_ridge_when_lower_dimension=True,
    )
    result = run_batch_protocol(
        contexts,
        labels,
        num_actions=dataset.num_actions,
        oracle_configuration=oracle,
        private=epsilon is not None,
        loss=loss,
        algorithms=(policy.algorithm,),
        policy_configurations={policy.algorithm: policy},
        seed=seed,
        include_intercept=False,
        include_nonprivate_benchmark_metrics=True,
        benchmark_common_randomness=epsilon is not None,
    )
    result["dataset"] = {
        "dataset_id": dataset.dataset_id,
        "source_path": str(dataset.source_path),
        "source_rows": len(dataset.labels),
        "horizon": stream_length,
        "master_seed": seed,
        "permutation_seed": permutation_seed,
        "permutation_indices": indices.tolist(),
        "use_bias": dataset.use_bias,
        "project_unitball": dataset.project_unitball,
    }
    return result
