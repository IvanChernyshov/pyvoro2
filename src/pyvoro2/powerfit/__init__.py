"""Public API for inverse fitting of power weights from pairwise constraints."""

from __future__ import annotations

from .constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from .model import (
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
from .active import (
    ActiveSetIteration,
    ActiveSetOptions,
    ActiveSetPathSummary,
    PairConstraintDiagnostics,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)
from .realize import (
    RealizedPairDiagnostics,
    UnaccountedRealizedPair,
    UnaccountedRealizedPairError,
    match_realized_pairs,
)
from .report import (
    build_active_set_report,
    build_fit_report,
    build_realized_report,
    dumps_report_json,
    write_report_json,
)
from .solver import (
    ConnectivityDiagnostics,
    ConnectivityDiagnosticsError,
    ConstraintGraphDiagnostics,
    HardConstraintConflict,
    HardConstraintConflictTerm,
    PowerWeightFitResult,
    fit_power_weights,
    radii_to_weights,
    weights_to_radii,
)

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
    'ConstraintGraphDiagnostics',
    'ConnectivityDiagnostics',
    'ConnectivityDiagnosticsError',
    'HardConstraintConflictTerm',
    'HardConstraintConflict',
    'PowerWeightFitResult',
    'RealizedPairDiagnostics',
    'UnaccountedRealizedPair',
    'UnaccountedRealizedPairError',
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
    'ActiveSetOptions',
    'ActiveSetIteration',
    'ActiveSetPathSummary',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'fit_power_weights',
    'match_realized_pairs',
    'solve_self_consistent_power_weights',
    'radii_to_weights',
    'weights_to_radii',
]
