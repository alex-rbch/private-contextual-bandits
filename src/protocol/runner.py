"""The batch-local protocol: play a frozen policy, then fit the next predictor.

Chronology for every non-final batch ``B_m``:

    play B_m with a frozen predictor and policy
        -> build regression rows only from B_m
        -> release one new predictor (and, if the policy explores, one
           exploration statistic)
        -> freeze them for B_(m+1).

There is no cumulative fitting, no within-batch update, and no reuse of an
older raw row. The predictor for ``B_(m+1)`` is trained on exactly ``B_m``.
The pooled exploration geometry is the one thing that spans batches, and it
sums only already-released statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterator

import numpy as np

from ..oracles import (
    FitContext, Oracle, OracleDiagnostics, Predictor,
    NonPrivateQuadraticLossOracle, PrivateQuadraticLossOracle,
)
from ..policies import (
    CumulativeExplorationGeometry,
    VPOParameters,
    PolicyConfiguration,
    SUPPORTED_POLICIES,
    vpo_parameters,
    averaged_virtual_policy,
    bpo_policy,
    confidence_survivors,
    fastcb_allocation,
    gamma_schedule,
    inverse_gap_allocation,
    linucb_allocation,
    policy_needs_exploration_geometry,
    predicted_gaps,
    regcb_allocation,
    sample_virtual_mixture_component,
    uniform_minimizers,
    validate_probability,
)
from ..policies.policy_optimization import PowerLawPopulationErrorEnvelope
from .batching import GeometricEpoch, geometric_epochs
from .features import ActionBlockFeatureMap

PopulationErrorEnvelope = PowerLawPopulationErrorEnvelope


@dataclass(frozen=True)
class BatchRelease:
    """One oracle call: which batch trained it, and what it released."""

    training_epoch: int
    predictor_epoch: int
    expected_sample_indices: tuple[int, ...]
    sample_indices: tuple[int, ...]
    predictor: Predictor


@dataclass(frozen=True)
class BatchDiagnostics:
    """Per-batch policy and oracle diagnostics."""

    algorithm: str
    epoch: int
    epoch_start: int
    epoch_end: int
    epoch_size: int
    oracle_type: str
    oracle_training_batch_index: int | None
    oracle_training_sample_count: int
    oracle_sample_indices: tuple[int, ...]
    epsilon_reg: float | None
    delta_reg: float | None
    feature_norm_bound: float
    ridge_lambda: float
    learning_rate: float | None
    gamma_m: float | None
    ucb_scale: float | None
    theta_norm: float | None
    feature_norm_max: float | None
    mean_predicted_gap: float
    max_predicted_gap: float
    min_action_probability: float
    max_action_probability: float
    policy_entropy: float
    virtual_rollout_length: int
    pooled_exploration_releases: int | None
    pooled_exploration_records: int | None
    gram_sensitivity: float | None
    response_sensitivity: float | None
    gram_noise_sigma: float | None
    response_noise_sigma: float | None
    gram_min_eig_before_noise: float | None
    gram_min_eig_after_noise: float | None
    optimizer_success: bool | None
    optimizer_iterations: int | None
    optimizer_gradient_norm: float | None
    feature_dimension: int | None
    theta_hat_norm: float | None
    theta_private_norm: float | None
    dual_matrix_dimension: int | None
    dual_matrix_condition_number: float | None
    prediction_out_of_range_fraction: float | None
    mean_abs_private_minus_nonprivate_prediction: float | None
    max_abs_private_minus_nonprivate_prediction: float | None
    ridge_parameterization: str | None = None

    _PRIVATE_BATCH_FACTS = (
        "feature_norm_max",
        "gram_min_eig_before_noise",
        "optimizer_success",
        "optimizer_iterations",
        "optimizer_gradient_norm",
        "theta_hat_norm",
        "dual_matrix_condition_number",
        "prediction_out_of_range_fraction",
        "mean_abs_private_minus_nonprivate_prediction",
        "max_abs_private_minus_nonprivate_prediction",
    )
    _PRIVATE_CONTEXT_FACTS = (
        "oracle_sample_indices",
        "mean_predicted_gap",
        "max_predicted_gap",
        "min_action_probability",
        "max_action_probability",
        "policy_entropy",
    )

    def debug_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["oracle_sample_indices"] = list(self.oracle_sample_indices)
        return payload

    def release_dict(
        self, *, private: bool, protect_private_contexts: bool = False
    ) -> dict[str, object]:
        payload = self.debug_dict()
        if private:
            for field in self._PRIVATE_BATCH_FACTS:
                payload[field] = None
        if private and protect_private_contexts:
            for field in self._PRIVATE_CONTEXT_FACTS:
                payload[field] = None
        return payload


class BatchLocalBandit:
    """Run one algorithm over the geometric batch schedule.

    The first batch is uniform. Every later batch uses one frozen predictor
    fitted on exactly the preceding batch. ``bpo`` additionally
    post-processes the sequence of already released predictors; it never
    refits on or reuses older raw rows.
    """

    def __init__(
        self,
        feature_map: ActionBlockFeatureMap,
        oracle: Oracle,
        horizon: int,
        policy: PolicyConfiguration,
        oracle_rng: np.random.Generator,
        population_error_envelope: PopulationErrorEnvelope | None = None,
        epoch_growth_factor: float = 2.0,
        independent_private_fits: bool = False,
        oracle_batch_seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, Oracle):
            raise TypeError("oracle must implement the Oracle interface")
        if not isinstance(oracle_rng, np.random.Generator):
            raise TypeError("oracle_rng must be numpy.random.Generator")
        if policy.algorithm not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy {policy.algorithm!r}")
        if policy.algorithm == "fastcb" and isinstance(
            oracle, (NonPrivateQuadraticLossOracle, PrivateQuadraticLossOracle)
        ):
            raise ValueError("FastCB only supports logistic loss")
        self.feature_map = feature_map
        self.oracle = oracle
        self.policy_configuration = policy
        self.epochs = geometric_epochs(horizon, epoch_growth_factor)
        self.epoch_growth_factor = float(epoch_growth_factor)
        # Legacy constructor argument kept for historical audit callers.
        # Active VPO does not read a population-error envelope.
        del population_error_envelope
        self.oracle_rng = oracle_rng
        # Privacy plumbing only: this does not change any policy or data usage.
        self.independent_private_fits = bool(independent_private_fits)
        # Public benchmark coupling only: restart the oracle stream for each
        # batch so an optional Gram draw cannot shift the next batch's noise.
        self.oracle_batch_seed = oracle_batch_seed
        self.explores = policy_needs_exploration_geometry(policy.algorithm)
        self.exploration_geometry = (
            CumulativeExplorationGeometry(feature_map.dimension)
            if self.explores
            else None
        )
        self.active_predictor: Predictor | None = None
        self.active_parameters: VPOParameters | None = None
        self._active_training_diagnostic: OracleDiagnostics | None = None
        self._active_training_indices: tuple[int, ...] = ()
        self._epoch_offset = 0
        self._rounds_observed = 0
        self._features: list[np.ndarray] = []
        self._targets: list[float] = []
        self._indices: list[int] = []
        self._seen_indices: set[int] = set()
        self._releases: list[BatchRelease] = []
        self._epoch_diagnostics: list[BatchDiagnostics] = []
        self._mean_gaps: list[float] = []
        self._max_gaps: list[float] = []
        self._minimum_probabilities: list[float] = []
        self._maximum_probabilities: list[float] = []
        self._entropies: list[float] = []
        self._pending_action: int | None = None
        self._pending_virtual_index: int | None = None
        self._sampled_virtual_indices: list[int | None] = []
        self._initial_oracle_debug_count = len(oracle.debug_diagnostics)

    @property
    def algorithm(self) -> str:
        return self.policy_configuration.algorithm

    @property
    def active_epoch(self) -> GeometricEpoch:
        if self._epoch_offset >= len(self.epochs):
            raise RuntimeError("the horizon has completed")
        return self.epochs[self._epoch_offset]

    @property
    def releases(self) -> tuple[BatchRelease, ...]:
        return tuple(self._releases)

    @property
    def epoch_diagnostics(self) -> tuple[BatchDiagnostics, ...]:
        """Trusted diagnostics, including facts measured on private rows."""

        return tuple(self._epoch_diagnostics)

    @property
    def sampled_virtual_indices(self) -> tuple[int | None, ...]:
        """The sampled zero-based K for VPO rounds; ``None`` otherwise."""

        return tuple(self._sampled_virtual_indices)

    def released_epoch_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.release_dict(
                private=self.oracle.private,
                protect_private_contexts=self.independent_private_fits,
            )
            for diagnostic in self._epoch_diagnostics
        )

    def _gamma_m(self) -> float:
        if not self._active_training_indices:
            raise AssertionError("gamma_m requires a previous training epoch")
        # Freeze the exponential-batch clock for the whole active batch. At its
        # start, ``start_round - 1`` rounds have been completed.
        return gamma_schedule(
            self.policy_configuration.gamma_scale,
            self.policy_configuration.gamma_exponent,
            max(1, self.active_epoch.start_round - 1),
        )

    def _historical_estimates(
        self, features: np.ndarray
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Predict from historical releases without changing the BPO rule."""

        for release in self._releases:
            yield (
                np.asarray(release.predictor.predict(features), dtype=np.float64),
                len(release.sample_indices),
            )

    def _exploration_widths(self, features: np.ndarray, width_scale: float) -> np.ndarray:
        if self.exploration_geometry is None:
            raise AssertionError("this policy has no exploration geometry")
        return np.asarray(
            self.exploration_geometry.width(features, width_scale), dtype=np.float64
        )

    def _evaluate_policy(
        self, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.active_epoch  # Fail loudly after the configured horizon.
        features = self.feature_map.transform_all(context)
        predictor = self.active_predictor
        if predictor is None:
            probability = np.full(
                self.feature_map.num_actions, 1.0 / self.feature_map.num_actions
            )
            predictions = np.full(self.feature_map.num_actions, 0.5)
            return probability, predictions, np.zeros_like(predictions)

        losses = np.asarray(predictor.predict(features), dtype=np.float64)
        if losses.shape != (self.feature_map.num_actions,) or not np.all(
            np.isfinite(losses)
        ):
            raise AssertionError(f"{self.algorithm} received invalid predictions")

        if self.algorithm == "linucb":
            if not self._active_training_indices:
                raise AssertionError("LinUCB requires a released previous-batch model")
            radii = self._exploration_widths(features, 1.0)
            probability = linucb_allocation(
                losses, radii, self.policy_configuration.ucb_scale
            )
            # Reported predictions retain the reward round trip the selection
            # rule performs, so diagnostics match the scores actually ranked.
            predictions = 1.0 - (1.0 - losses)
            return (
                validate_probability(probability),
                predictions,
                predicted_gaps(predictions),
            )

        predictions = losses
        gaps = predicted_gaps(predictions)
        if self.algorithm == "vpo":
            parameters = self.active_parameters
            if parameters is None:
                raise AssertionError("a released model is missing VPO parameters")
            probability = averaged_virtual_policy(
                predictions, parameters.learning_rate, parameters.rollout_length
            )
        elif self.algorithm == "bpo":
            probability = bpo_policy(
                self._historical_estimates(features),
                num_actions=self.feature_map.num_actions,
                eta=self.policy_configuration.learning_rate_scale,
                gamma=self.policy_configuration.gamma_scale,
            )
        elif self.algorithm == "squarecb":
            probability = inverse_gap_allocation(predictions, self._gamma_m())
        elif self.algorithm == "fastcb":
            probability = fastcb_allocation(predictions, self._gamma_m())
        elif self.algorithm == "supervised":
            probability = uniform_minimizers(predictions)
        elif self.algorithm in ("regcb", "adacb"):
            widths = self._exploration_widths(
                features, self.policy_configuration.width_scale
            )
            survivors = confidence_survivors(
                np.asarray(predictor.raw_predict(features), dtype=np.float64),
                widths,
            )
            probability = (
                regcb_allocation(survivors)
                if self.algorithm == "regcb"
                else inverse_gap_allocation(
                    predictions, self._gamma_m(), eligible=survivors
                )
            )
        else:
            raise AssertionError(f"unhandled policy {self.algorithm}")
        return validate_probability(probability), predictions, gaps

    def action_probabilities(self, context: np.ndarray) -> np.ndarray:
        probability, _, _ = self._evaluate_policy(context)
        return probability

    def _sample_distribution(
        self, context: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Return the distribution, displayed gaps, and optional virtual index."""

        if (
            self.algorithm == "vpo"
            and self.active_predictor is not None
        ):
            self.active_epoch
            features = self.feature_map.transform_all(context)
            predictions = np.asarray(
                self.active_predictor.predict(features), dtype=np.float64
            )
            gaps = predicted_gaps(predictions)
            parameters = self.active_parameters
            if parameters is None:
                raise AssertionError("a released model is missing VPO parameters")
            virtual_index, probability = sample_virtual_mixture_component(
                predictions, parameters.learning_rate, parameters.rollout_length, rng
            )
            return probability, gaps, virtual_index
        probability, _, gaps = self._evaluate_policy(context)
        return probability, gaps, None

    def sample_action(
        self, context: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, np.ndarray]:
        if self._pending_action is not None:
            raise RuntimeError("observe the pending action before sampling again")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        probability, gaps, virtual_index = self._sample_distribution(context, rng)
        action = int(rng.choice(self.feature_map.num_actions, p=probability))
        self._pending_action = action
        self._pending_virtual_index = virtual_index
        self._sampled_virtual_indices.append(virtual_index)
        self._mean_gaps.append(float(np.mean(gaps)))
        self._max_gaps.append(float(np.max(gaps)))
        self._minimum_probabilities.append(float(np.min(probability)))
        self._maximum_probabilities.append(float(np.max(probability)))
        positive = probability[probability > 0.0]
        self._entropies.append(float(-np.sum(positive * np.log(positive))))
        return action, probability

    def _finalize_epoch_diagnostics(self, epoch: GeometricEpoch) -> None:
        if len(self._mean_gaps) != epoch.size:
            raise AssertionError("each epoch round must sample exactly one policy")
        source = self._active_training_diagnostic
        parameters = self.active_parameters
        geometry = self.exploration_geometry
        has_gamma = source is not None and self.algorithm in (
            "squarecb",
            "fastcb",
            "adacb",
        )
        diagnostics = BatchDiagnostics(
            algorithm=self.algorithm,
            epoch=epoch.epoch,
            epoch_start=epoch.start_round,
            epoch_end=epoch.end_round,
            epoch_size=epoch.size,
            oracle_type=type(self.oracle).__name__,
            oracle_training_batch_index=(None if source is None else source.batch_index),
            oracle_training_sample_count=0 if source is None else source.batch_size,
            oracle_sample_indices=self._active_training_indices,
            epsilon_reg=(
                self.oracle.configuration.epsilon_reg if self.oracle.private else None
            ),
            delta_reg=(
                self.oracle.configuration.delta_reg if self.oracle.private else None
            ),
            feature_norm_bound=self.oracle.configuration.feature_norm_bound,
            ridge_lambda=(
                self.oracle.configuration.ridge_lambda
                if source is None
                else source.ridge_lambda
            ),
            learning_rate=(
                self.policy_configuration.learning_rate_scale
                if source is not None and self.algorithm == "bpo"
                else (None if parameters is None else parameters.learning_rate)
            ),
            gamma_m=(
                self.policy_configuration.gamma_scale
                if source is not None and self.algorithm == "bpo"
                else (self._gamma_m() if has_gamma else None)
            ),
            ucb_scale=(
                self.policy_configuration.ucb_scale
                if source is not None and self.algorithm == "linucb"
                else None
            ),
            theta_norm=None if source is None else source.theta_norm,
            feature_norm_max=(
                None if source is None else source.max_observed_feature_norm
            ),
            mean_predicted_gap=float(np.mean(self._mean_gaps)),
            max_predicted_gap=float(np.max(self._max_gaps)),
            min_action_probability=float(np.min(self._minimum_probabilities)),
            max_action_probability=float(np.max(self._maximum_probabilities)),
            policy_entropy=float(np.mean(self._entropies)),
            virtual_rollout_length=(0 if parameters is None else parameters.rollout_length),
            pooled_exploration_releases=(
                None if geometry is None else geometry.release_count
            ),
            pooled_exploration_records=(
                None if geometry is None else geometry.record_count
            ),
            gram_sensitivity=None if source is None else source.gram_sensitivity,
            response_sensitivity=(None if source is None else source.response_sensitivity),
            gram_noise_sigma=None if source is None else source.gram_noise_sigma,
            response_noise_sigma=(None if source is None else source.response_noise_sigma),
            gram_min_eig_before_noise=(
                None if source is None else source.gram_min_eigenvalue_before_noise
            ),
            gram_min_eig_after_noise=(
                None if source is None else source.gram_min_eigenvalue_after_noise
            ),
            optimizer_success=None if source is None else source.optimizer_success,
            optimizer_iterations=(None if source is None else source.optimizer_iterations),
            optimizer_gradient_norm=(
                None if source is None else source.optimizer_gradient_norm
            ),
            feature_dimension=None if source is None else source.feature_dimension,
            theta_hat_norm=None if source is None else source.theta_hat_norm,
            theta_private_norm=None if source is None else source.theta_private_norm,
            dual_matrix_dimension=(None if source is None else source.dual_matrix_dimension),
            dual_matrix_condition_number=(
                None if source is None else source.dual_matrix_condition_number
            ),
            prediction_out_of_range_fraction=(
                None if source is None else source.prediction_out_of_range_fraction
            ),
            mean_abs_private_minus_nonprivate_prediction=(
                None
                if source is None
                else source.mean_abs_private_minus_nonprivate_prediction
            ),
            max_abs_private_minus_nonprivate_prediction=(
                None
                if source is None
                else source.max_abs_private_minus_nonprivate_prediction
            ),
            ridge_parameterization=(
                None if source is None else source.ridge_parameterization
            ),
        )
        self._epoch_diagnostics.append(diagnostics)
        self._mean_gaps.clear()
        self._max_gaps.clear()
        self._minimum_probabilities.clear()
        self._maximum_probabilities.clear()
        self._entropies.clear()

    def _release_predictor_for_next_epoch(self, epoch: GeometricEpoch) -> Predictor:
        expected_indices = epoch.sample_indices
        indices = tuple(self._indices)
        if indices != expected_indices:
            raise AssertionError(
                f"oracle input for predictor g_{epoch.epoch + 1} is not exactly B_{epoch.epoch}"
            )
        before = len(self.oracle.debug_diagnostics)
        if self.oracle_batch_seed is not None:
            fit_rng = np.random.default_rng(
                np.random.SeedSequence([self.oracle_batch_seed, epoch.epoch])
            )
        elif self.independent_private_fits:
            fit_rng = np.random.default_rng()
        else:
            fit_rng = self.oracle_rng
        predictor = self.oracle.fit(
            np.stack(self._features),
            np.asarray(self._targets, dtype=np.float64),
            fit_rng,
            context=FitContext(
                algorithm=self.algorithm,
                batch_index=epoch.epoch,
                sample_indices=indices,
                full_information_action_count=(
                    self.feature_map.num_actions
                    if self.algorithm == "supervised"
                    else None
                ),
            ),
        )
        if len(self.oracle.debug_diagnostics) != before + 1:
            raise AssertionError("one completed epoch must cause exactly one oracle call")
        predictor_epoch = epoch.epoch + 1
        parameters: VPOParameters | None = None
        if self.algorithm == "vpo":
            parameters = vpo_parameters(
                predictor_epoch=predictor_epoch,
                training_epoch=epoch.epoch,
                training_epoch_size=epoch.size,
                num_actions=self.feature_map.num_actions,
                eta0=self.policy_configuration.learning_rate_scale,
            )
        self._releases.append(
            BatchRelease(
                training_epoch=epoch.epoch,
                predictor_epoch=predictor_epoch,
                expected_sample_indices=expected_indices,
                sample_indices=indices,
                predictor=predictor,
            )
        )
        if self.exploration_geometry is not None:
            if predictor.exploration_statistic is None:
                raise AssertionError(
                    "an exploring policy requires a released batch statistic"
                )
            self.exploration_geometry.accumulate(
                predictor.exploration_statistic, record_count=len(indices)
            )
        self.active_predictor = predictor
        self.active_parameters = parameters
        self._active_training_diagnostic = self.oracle.debug_diagnostics[-1]
        self._active_training_indices = indices
        return predictor

    def observe(
        self,
        sample_index: int,
        context: np.ndarray,
        action: int,
        loss: float,
        *,
        optimal_action: int | None = None,
    ) -> Predictor | None:
        if self._pending_action is None:
            raise RuntimeError("sample an action before observing its loss")
        if action != self._pending_action:
            raise RuntimeError("observed action differs from the sampled action")
        if sample_index != self._rounds_observed:
            raise RuntimeError(
                f"expected sample index {self._rounds_observed}, got {sample_index}"
            )
        epoch = self.active_epoch
        if not epoch.start_round - 1 <= sample_index < epoch.end_round:
            raise RuntimeError("sample index is outside the active epoch")
        if sample_index in self._seen_indices or sample_index in self._indices:
            raise RuntimeError("a raw row was assigned to more than one epoch")
        if not math.isfinite(loss) or loss < 0.0 or loss > 1.0:
            raise ValueError("loss must lie in [0, 1]")
        if self.algorithm == "supervised":
            if optimal_action is None or not (
                0 <= optimal_action < self.feature_map.num_actions
            ):
                raise ValueError("supervised feedback requires a valid optimal action")
            full_losses = np.ones(self.feature_map.num_actions, dtype=np.float64)
            full_losses[int(optimal_action)] = 0.0
            self._features.extend(self.feature_map.transform_all(context))
            self._targets.extend(float(value) for value in full_losses)
        else:
            self._features.append(self.feature_map.transform(context, action))
            self._targets.append(float(loss))
        self._indices.append(int(sample_index))
        self._pending_action = None
        self._pending_virtual_index = None
        self._rounds_observed += 1
        if len(self._indices) < epoch.size:
            return None

        self._finalize_epoch_diagnostics(epoch)
        predictor: Predictor | None = None
        if self._epoch_offset + 1 < len(self.epochs):
            predictor = self._release_predictor_for_next_epoch(epoch)
        self._seen_indices.update(self._indices)
        self._features.clear()
        self._targets.clear()
        self._indices.clear()
        self._epoch_offset += 1
        return predictor

    def assert_batch_locality(self) -> None:
        """Verify the disjointness the parallel-composition argument assumes."""

        expected_calls = max(0, len(self.epochs) - 1)
        if len(self._releases) != expected_calls:
            raise AssertionError("expected exactly one oracle call per non-final epoch")
        for release, epoch in zip(self._releases, self.epochs[:-1]):
            if release.training_epoch != epoch.epoch:
                raise AssertionError("oracle training epoch indexing is inconsistent")
            if release.sample_indices != epoch.sample_indices:
                raise AssertionError("an oracle received rows outside its training epoch")
            if release.sample_indices != release.expected_sample_indices:
                raise AssertionError("an oracle input differs from its expected epoch")
        flattened = [
            index for release in self._releases for index in release.sample_indices
        ]
        if len(flattened) != len(set(flattened)):
            raise AssertionError("oracle inputs are not pairwise disjoint")
        expected_private_rows = [
            index for epoch in self.epochs[:-1] for index in epoch.sample_indices
        ]
        if flattened != expected_private_rows:
            raise AssertionError("eligible raw rows were not accessed exactly once")
        oracle_calls = len(self.oracle.debug_diagnostics) - self._initial_oracle_debug_count
        if oracle_calls != expected_calls:
            raise AssertionError("oracle call accounting differs from epoch accounting")
        if self._epoch_offset != len(self.epochs):
            raise AssertionError("the horizon has not completed")
        if self._features or self._targets or self._indices:
            raise AssertionError("raw rows remain buffered after the horizon")
        if len(self._sampled_virtual_indices) != self._rounds_observed:
            raise AssertionError("virtual-index accounting differs from round accounting")
        if self.exploration_geometry is not None and (
            self.exploration_geometry.release_count != expected_calls
        ):
            raise AssertionError("pooled geometry did not absorb every release")
        if self.algorithm == "vpo":
            for offset, epoch in enumerate(self.epochs):
                indices = self._sampled_virtual_indices[
                    epoch.start_round - 1 : epoch.end_round
                ]
                if offset == 0:
                    if any(index is not None for index in indices):
                        raise AssertionError("uniform epoch must not sample a virtual index")
                    continue
                rollout_length = self.epochs[offset - 1].size
                if any(
                    index is None or not 0 <= index < rollout_length
                    for index in indices
                ):
                    raise AssertionError("a sampled virtual index is out of range")
