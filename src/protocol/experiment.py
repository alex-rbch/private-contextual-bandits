"""Drive every selected policy over one context stream and report the runs."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Iterable, Mapping

import numpy as np

from ..oracles import LOSS_MODELS, OracleConfiguration, build_oracles, oracle_name
from ..policies import (
    PolicyConfiguration,
    SUPPORTED_POLICIES,
    fastcb_supports_loss,
    fastcb_unsupported_run,
    policy_needs_exploration_geometry,
)
from ..privacy import split_objective_gram_budget
from .batching import geometric_epochs
from .features import ActionBlockFeatureMap
from .runner import BatchLocalBandit


def stable_seed(seed: int, *parts: object) -> int:
    """Derive a deterministic independent NumPy seed."""

    payload = "|".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def paired_algorithm_seeds(seed: int, algorithm: str) -> tuple[int, int]:
    """Return oracle/action seeds coupled across privacy regimes."""

    return (
        stable_seed(seed, algorithm, "oracle"),
        stable_seed(seed, algorithm, "actions"),
    )


def validate_problem(
    contexts: np.ndarray,
    optimal_actions: np.ndarray,
    num_actions: int,
    batch_sizes: tuple[int, ...],
    algorithms: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(contexts, dtype=np.float64)
    labels = np.asarray(optimal_actions)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("contexts must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contexts must be finite")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("optimal_actions must contain one label per context")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError("optimal_actions must be integer-valued")
    labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= num_actions):
        raise ValueError("optimal_actions must be zero-based action indices")
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive sizes")
    if sum(batch_sizes) != matrix.shape[0]:
        raise ValueError(
            "batch_sizes must partition the full horizon; "
            "partial tails are not privately released"
        )
    if not algorithms or len(algorithms) != len(set(algorithms)):
        raise ValueError("algorithms must be nonempty and unique")
    unsupported = set(algorithms).difference(SUPPORTED_POLICIES)
    if unsupported:
        raise ValueError(f"unsupported algorithms: {sorted(unsupported)}")
    return matrix, labels


def _logistic_privacy_manifest(
    loss: str,
    private: bool,
    algorithms: Iterable[str],
    configuration: OracleConfiguration,
    *,
    num_actions: int,
    scheduled_fit_count: int,
) -> dict[str, object] | None:
    """Record the policy-driven logistic budget split, changing no policy."""

    if loss != "logistic":
        return None
    manifest: dict[str, object] = {}
    for name in algorithms:
        explores = policy_needs_exploration_geometry(name)
        if not private:
            manifest[name] = {
                "needs_gram_release": explores,
                "objective_perturbation": "zero",
                "gram_release": "exact" if explores else "unused",
            }
        elif scheduled_fit_count == 0:
            manifest[name] = {
                "needs_gram_release": explores,
                "scheduled_fit_count": 0,
                "objective_epsilon": None,
                "objective_delta": None,
                "gram_epsilon": None,
                "gram_delta": None,
                "epsilon_fit": None,
            }
        else:
            budget = split_objective_gram_budget(
                configuration.epsilon_reg,
                configuration.delta_reg,
                needs_gram=explores,
            )
            group = num_actions if name == "supervised" else 1
            manifest[name] = {
                "needs_gram_release": explores,
                "objective_epsilon": budget.objective_epsilon,
                "objective_delta": budget.objective_delta,
                "gram_epsilon": budget.gram_epsilon,
                "gram_delta": budget.gram_delta,
                "epsilon_fit": budget.objective_epsilon / group,
                "scalar_examples_per_original_row": group,
                "scheduled_fit_count": scheduled_fit_count,
            }
    return manifest


def run_batch_protocol(
    contexts: np.ndarray,
    optimal_actions: np.ndarray,
    *,
    num_actions: int,
    oracle_configuration: OracleConfiguration,
    private: bool,
    loss: str,
    algorithms: Iterable[str] = SUPPORTED_POLICIES,
    policy_configurations: Mapping[str, PolicyConfiguration] | None = None,
    seed: int = 1,
    include_intercept: bool = True,
    action_feature_scale: float = 1.0,
    epoch_growth_factor: float = 2.0,
    include_nonprivate_benchmark_metrics: bool = False,
    benchmark_common_randomness: bool = False,
) -> dict[str, object]:
    """Compare policies under the batch-local protocol on one context stream.

    Private output defaults to model releases, not raw observations. Public
    benchmark drivers may explicitly request losses and actions for tuning;
    that artifact is not itself a differentially private release.
    """

    if loss not in LOSS_MODELS:
        raise ValueError(f"loss must be one of {LOSS_MODELS}")
    if benchmark_common_randomness and private and not include_nonprivate_benchmark_metrics:
        raise ValueError(
            "seeded private noise is only available for public-data benchmarks"
        )
    names = tuple(algorithms)
    epochs = geometric_epochs(len(contexts), epoch_growth_factor)
    scheduled_fit_count = max(0, len(epochs) - 1)
    # Epoch boundaries depend only on the public horizon and growth factor.
    # Each original index is fitted once, with all K supervised examples
    # together, so the configured per-run budget is also each batch's budget.
    matrix, labels = validate_problem(
        contexts, optimal_actions, num_actions, tuple(e.size for e in epochs), names
    )
    feature_map = ActionBlockFeatureMap(
        matrix.shape[1], num_actions, include_intercept, action_feature_scale
    )
    unsupported_runs = {
        name: fastcb_unsupported_run(len(matrix))
        for name in names if name == "fastcb" and not fastcb_supports_loss(loss)
    }
    active_names = tuple(name for name in names if name not in unsupported_runs)
    oracles = build_oracles(
        active_names,
        oracle_configuration,
        private=private,
        loss=loss,
        needs_exploration_statistic=policy_needs_exploration_geometry,
    ) if active_names else {}
    # A genuine private logistic release keeps fresh secret entropy. Public
    # benchmarks can instead couple pseudorandom noise across algorithms.
    independent_private_fits = (
        private and loss == "logistic" and not benchmark_common_randomness
    )

    configured_policies = dict(policy_configurations or {})
    unknown = set(configured_policies).difference(names)
    if unknown:
        raise ValueError(
            f"policy configurations supplied for unselected algorithms: {sorted(unknown)}"
        )
    resolved = {
        name: configured_policies.get(name, PolicyConfiguration(name)) for name in names
    }

    runs: dict[str, dict[str, object]] = {}
    for name in names:
        policy = resolved[name]
        if policy.algorithm != name:
            raise ValueError(f"policy configuration key {name} contains {policy.algorithm}")
        if name in unsupported_runs:
            runs[name] = unsupported_runs[name]
            continue
        oracle_rng_seed, action_rng_seed = paired_algorithm_seeds(seed, name)
        if benchmark_common_randomness:
            oracle_rng_seed = stable_seed(seed, "oracle", "common_batches")
        bandit = BatchLocalBandit(
            feature_map=feature_map,
            oracle=oracles[name],
            horizon=len(matrix),
            policy=policy,
            oracle_rng=np.random.default_rng(
                None if independent_private_fits else oracle_rng_seed
            ),
            epoch_growth_factor=epoch_growth_factor,
            independent_private_fits=independent_private_fits,
            oracle_batch_seed=(
                oracle_rng_seed if benchmark_common_randomness else None
            ),
        )
        action_rng = np.random.default_rng(action_rng_seed)
        actions: list[int] = []
        probabilities: list[list[float]] = []
        losses: list[float] = []
        full_action_mse: list[float | None] = []
        argmin_contains_optimal: list[float | None] = []
        for sample_index, (context, optimal_action) in enumerate(zip(matrix, labels)):
            if private or bandit.active_predictor is None:
                full_action_mse.append(None)
                argmin_contains_optimal.append(None)
            else:
                features = feature_map.transform_all(context)
                predictions = np.asarray(
                    bandit.active_predictor.predict(features), dtype=np.float64
                )
                true_losses = np.ones(num_actions, dtype=np.float64)
                true_losses[int(optimal_action)] = 0.0
                full_action_mse.append(
                    float(np.mean((predictions - true_losses) ** 2))
                )
                minimizers = np.isclose(
                    predictions, float(np.min(predictions)), rtol=1e-12, atol=1e-15
                )
                argmin_contains_optimal.append(float(minimizers[int(optimal_action)]))
            action, probability = bandit.sample_action(context, action_rng)
            loss_value = float(action != int(optimal_action))
            bandit.observe(
                sample_index,
                context,
                action,
                loss_value,
                optimal_action=(int(optimal_action) if name == "supervised" else None),
            )
            actions.append(action)
            probabilities.append([float(value) for value in probability])
            losses.append(loss_value)
        bandit.assert_batch_locality()
        if len(bandit.releases) != scheduled_fit_count:
            raise AssertionError("protocol made an unexpected number of oracle calls")
        runs[name] = {
            "randomness": {
                "oracle_rng_seed": oracle_rng_seed,
                "action_rng_seed": action_rng_seed,
                **(
                    {"oracle_noise_coupling": "same_base_draws_per_batch_across_algorithms"}
                    if benchmark_common_randomness
                    else {}
                ),
            },
            "actions": actions,
            "action_probabilities": probabilities,
            "observed_losses": losses,
            "cumulative_loss": float(sum(losses)),
            "oracle_calls": len(bandit.releases),
            "oracle_training_epochs": [r.training_epoch for r in bandit.releases],
            "oracle_predictor_epochs": [r.predictor_epoch for r in bandit.releases],
            "oracle_sample_indices": [list(r.sample_indices) for r in bandit.releases],
            "epoch_diagnostics": list(bandit.released_epoch_diagnostics()),
            "sampled_virtual_indices": list(bandit.sampled_virtual_indices),
            "recorded_probability_semantics": (
                "conditional_on_sampled_virtual_index"
                if name == "vpo"
                else "policy_distribution"
            ),
            "oracle_full_action_mse": full_action_mse,
            "oracle_argmin_contains_optimal": argmin_contains_optimal,
        }
        if independent_private_fits:
            runs[name]["randomness"] = {
                "action_rng_seed": action_rng_seed,
                "privacy_randomness": "independent_entropy_per_fit_not_released",
            }
            runs[name].pop("oracle_sample_indices")
            if not include_nonprivate_benchmark_metrics:
                for field in (
                    "actions",
                    "action_probabilities",
                    "observed_losses",
                    "cumulative_loss",
                    "sampled_virtual_indices",
                    "recorded_probability_semantics",
                    "oracle_full_action_mse",
                    "oracle_argmin_contains_optimal",
                ):
                    runs[name].pop(field)
                runs[name]["model_releases"] = [
                    {
                        "training_epoch": r.training_epoch,
                        "predictor_epoch": r.predictor_epoch,
                        "original_record_count": len(r.sample_indices),
                        "scalar_example_count": r.predictor.sample_count,
                        "theta": r.predictor.theta.tolist(),
                        "exploration_statistic": (
                            r.predictor.exploration_statistic.tolist()
                            if r.predictor.exploration_statistic is not None
                            else None
                        ),
                    }
                    for r in bandit.releases
                ]

    return {
        "configuration": {
            "benchmark_only_not_dp_release": bool(
                private and (
                    benchmark_common_randomness
                    or include_nonprivate_benchmark_metrics
                )
            ),
            "benchmark_common_randomness": benchmark_common_randomness,
            "public_seed_controls_privacy_noise": bool(
                private and benchmark_common_randomness
            ),
            "loss": loss,
            "regression_oracle": oracle_name(loss, private),
            "private": private,
            "predictor_scope": "previous_epoch_only",
            "exploration_geometry_scope": "all_released_batches",
            "exploration_ridge_rule": "fixed_unit_identity",
            "confidence_rule": "pooled_released_batch_statistics_no_log_factor",
            "regression_update_inside_epoch": False,
            "regression_update_frequency": "once_per_completed_epoch",
            "raw_regression_rows_retained_after_release": False,
            "final_epoch_fit": False,
            "prediction_floors_enabled": False,
            "supervised_feedback": (
                "complete_zero_one_action_loss_vector_per_original_row"
                if "supervised" in names
                else None
            ),
            "privacy_composition_across_epochs": (
                "adaptive_parallel_disjoint_original_records" if private else None
            ),
            "privacy_budget_divided_by_epoch_count": False,
            "scheduled_fit_count": scheduled_fit_count,
            "logistic_privacy_manifest": _logistic_privacy_manifest(
                loss,
                private,
                names,
                oracle_configuration,
                num_actions=num_actions,
                scheduled_fit_count=scheduled_fit_count,
            ),
            "ridge_lambda": oracle_configuration.ridge_lambda,
            "quadratic_ridge_multiplier": (
                (
                    "fitting_rows"
                    if oracle_configuration.quadratic_ridge_scale_by_batch_records
                    else "fixed_legacy_comparison"
                )
                if loss == "quadratic"
                else None
            ),
            "epsilon_reg": oracle_configuration.epsilon_reg if private else None,
            "delta_reg": oracle_configuration.delta_reg if private else None,
            "adjacency": oracle_configuration.adjacency.value,
            "feature_norm_bound": oracle_configuration.feature_norm_bound,
            "target_bound": oracle_configuration.target_bound,
            "feature_map": "exact_action_block",
            "feature_dimension": feature_map.dimension,
            "include_intercept": include_intercept,
            "action_feature_scale": action_feature_scale,
            "epoch_growth_factor": epoch_growth_factor,
            "horizon": len(matrix),
            "epochs": [
                {
                    "epoch": e.epoch,
                    "start_round": e.start_round,
                    "end_round": e.end_round,
                    "size": e.size,
                }
                for e in epochs
            ],
            "policy_configurations": {
                name: asdict(policy) for name, policy in resolved.items()
            },
            "seed": seed,
        },
        "algorithms": runs,
    }
