"""Public API for inverse fitting of power weights from pairwise constraints."""

from __future__ import annotations

from ._powerfit_constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from ._powerfit_model import (
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)
from ._powerfit_active import (
    ActiveSetIteration,
    ActiveSetOptions,
    PairConstraintDiagnostics,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)
from ._powerfit_realize import RealizedPairDiagnostics, match_realized_pairs
from ._powerfit_solver import PowerWeightFitResult, fit_power_weights, radii_to_weights, weights_to_radii

__all__ = [
    'PairBisectorConstraints',
    'resolve_pair_bisector_constraints',
    'SquaredLoss',
    'HuberLoss',
    'Interval',
    'FixedValue',
    'SoftIntervalPenalty',
    'ExponentialBoundaryPenalty',
    'ReciprocalBoundaryPenalty',
    'L2Regularization',
    'FitModel',
    'PowerWeightFitResult',
    'RealizedPairDiagnostics',
    'ActiveSetOptions',
    'ActiveSetIteration',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'fit_power_weights',
    'match_realized_pairs',
    'solve_self_consistent_power_weights',
    'radii_to_weights',
    'weights_to_radii',
]
