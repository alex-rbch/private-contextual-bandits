"""Action-selection rules. One module per family of update rule."""

from .base import (
    BANDIT_POLICIES,
    EXPLORATION_POLICIES,
    SUPPORTED_POLICIES,
    PolicyConfiguration,
    gamma_schedule,
    policy_needs_exploration_geometry,
    predicted_gaps,
    uniform_minimizers,
    validate_probability,
)
from .confidence import (
    CumulativeExplorationGeometry,
    confidence_bounds,
    confidence_survivors,
    linucb_allocation,
    regcb_allocation,
)
from .inverse_gap import (
    fastcb_allocation, fastcb_supports_loss, fastcb_unsupported_run,
    inverse_gap_allocation,
)
from .policy_optimization import (
    ConstantPopulationErrorEnvelope,
    PowerLawPopulationErrorEnvelope,
    VPOParameters,
    averaged_virtual_policy,
    bpo_policy,
    direct_virtual_trajectory,
    recursive_virtual_trajectory,
    sample_virtual_mixture_component,
    virtual_vpo_update,
    vpo_parameters,
)

__all__ = [
    "BANDIT_POLICIES",
    "ConstantPopulationErrorEnvelope",
    "CumulativeExplorationGeometry",
    "EXPLORATION_POLICIES",
    "PolicyConfiguration",
    "PowerLawPopulationErrorEnvelope",
    "SUPPORTED_POLICIES",
    "VPOParameters",
    "averaged_virtual_policy",
    "bpo_policy",
    "confidence_bounds",
    "confidence_survivors",
    "direct_virtual_trajectory",
    "fastcb_allocation",
    "fastcb_supports_loss",
    "fastcb_unsupported_run",
    "gamma_schedule",
    "inverse_gap_allocation",
    "linucb_allocation",
    "policy_needs_exploration_geometry",
    "predicted_gaps",
    "recursive_virtual_trajectory",
    "regcb_allocation",
    "sample_virtual_mixture_component",
    "uniform_minimizers",
    "validate_probability",
    "virtual_vpo_update",
    "vpo_parameters",
]
