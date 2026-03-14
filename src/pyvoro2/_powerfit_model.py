"""Objective models for inverse fitting of power weights.

The inverse-fit API is intentionally generic: downstream code specifies which
pairs matter, which periodic image is used for each pair, and which scalar
separator target should be matched. This module defines the objective pieces
used to fit power weights from those constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class ScalarMismatch:
    """Base class for mismatch terms applied to predicted separator positions."""


@dataclass(frozen=True, slots=True)
class SquaredLoss(ScalarMismatch):
    """Quadratic mismatch penalty: ``(predicted - target)^2``."""


@dataclass(frozen=True, slots=True)
class HuberLoss(ScalarMismatch):
    """Huber mismatch penalty in the chosen measurement space.

    The penalty is quadratic near zero and linear for large residuals.
    """

    delta: float = 1.0

    def __post_init__(self) -> None:
        if float(self.delta) <= 0.0:
            raise ValueError('HuberLoss.delta must be > 0')


class HardConstraint:
    """Base class for hard feasibility restrictions."""


@dataclass(frozen=True, slots=True)
class Interval(HardConstraint):
    """Hard interval restriction in the chosen measurement space."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not float(self.upper) > float(self.lower):
            raise ValueError('Interval requires upper > lower')


@dataclass(frozen=True, slots=True)
class FixedValue(HardConstraint):
    """Hard equality restriction in the chosen measurement space."""

    value: float


class ScalarPenalty:
    """Base class for additional scalar penalties."""


@dataclass(frozen=True, slots=True)
class SoftIntervalPenalty(ScalarPenalty):
    """Quadratic penalty for leaving a preferred interval.

    The penalty is zero within ``[lower, upper]`` and quadratic outside.
    """

    lower: float
    upper: float
    strength: float

    def __post_init__(self) -> None:
        if not float(self.upper) > float(self.lower):
            raise ValueError('SoftIntervalPenalty requires upper > lower')
        if float(self.strength) < 0.0:
            raise ValueError('SoftIntervalPenalty.strength must be >= 0')


@dataclass(frozen=True, slots=True)
class ExponentialBoundaryPenalty(ScalarPenalty):
    """Repulsive penalty near the boundaries of an interval.

    The penalty is based on exponentials measured from an inner interval
    ``[lower + margin, upper - margin]``.
    """

    lower: float = 0.0
    upper: float = 1.0
    margin: float = 0.02
    strength: float = 1.0
    tau: float = 0.01

    def __post_init__(self) -> None:
        if not float(self.upper) > float(self.lower):
            raise ValueError('ExponentialBoundaryPenalty requires upper > lower')
        if float(self.margin) < 0.0:
            raise ValueError('ExponentialBoundaryPenalty.margin must be >= 0')
        if float(self.strength) < 0.0:
            raise ValueError('ExponentialBoundaryPenalty.strength must be >= 0')
        if float(self.tau) <= 0.0:
            raise ValueError('ExponentialBoundaryPenalty.tau must be > 0')
        if float(self.lower) + float(self.margin) > float(self.upper) - float(
            self.margin
        ):
            raise ValueError('ExponentialBoundaryPenalty margin is too large')


@dataclass(frozen=True, slots=True)
class ReciprocalBoundaryPenalty(ScalarPenalty):
    """Reciprocal repulsion near interval boundaries.

    This penalty is intended to be used together with a hard interval or a
    strong outside penalty. It penalizes separator positions that enter the
    boundary layers ``[lower, lower + margin]`` and ``[upper - margin, upper]``.
    """

    lower: float = 0.0
    upper: float = 1.0
    margin: float = 0.05
    strength: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not float(self.upper) > float(self.lower):
            raise ValueError('ReciprocalBoundaryPenalty requires upper > lower')
        if float(self.margin) < 0.0:
            raise ValueError('ReciprocalBoundaryPenalty.margin must be >= 0')
        if float(self.strength) < 0.0:
            raise ValueError('ReciprocalBoundaryPenalty.strength must be >= 0')
        if float(self.epsilon) <= 0.0:
            raise ValueError('ReciprocalBoundaryPenalty.epsilon must be > 0')
        if float(self.lower) + float(self.margin) > float(self.upper) - float(
            self.margin
        ):
            raise ValueError('ReciprocalBoundaryPenalty margin is too large')


@dataclass(frozen=True, slots=True)
class L2Regularization:
    """Optional L2 regularization on the weight vector."""

    strength: float = 0.0
    reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        if float(self.strength) < 0.0:
            raise ValueError('L2Regularization.strength must be >= 0')
        ref = self.reference
        if ref is not None:
            arr = np.asarray(ref, dtype=float)
            if arr.ndim != 1:
                raise ValueError('L2Regularization.reference must be 1D')
            object.__setattr__(self, 'reference', arr)


@dataclass(frozen=True, slots=True)
class FitModel:
    """Complete objective definition for inverse power-weight fitting.

    The objective consists of:
      - one required mismatch term,
      - an optional hard feasibility set,
      - zero or more extra penalties,
      - optional L2 regularization on the weights.
    """

    mismatch: ScalarMismatch = field(default_factory=SquaredLoss)
    feasible: HardConstraint | None = None
    penalties: tuple[ScalarPenalty, ...] = ()
    regularization: L2Regularization = field(default_factory=L2Regularization)

    def __post_init__(self) -> None:
        if not isinstance(self.mismatch, ScalarMismatch):
            raise TypeError('FitModel.mismatch must be a ScalarMismatch instance')
        if self.feasible is not None and not isinstance(self.feasible, HardConstraint):
            raise TypeError('FitModel.feasible must be a HardConstraint or None')
        penalties = self.penalties
        if isinstance(penalties, Sequence) and not isinstance(penalties, tuple):
            penalties = tuple(penalties)
            object.__setattr__(self, 'penalties', penalties)
        if not all(isinstance(p, ScalarPenalty) for p in penalties):
            raise TypeError('FitModel.penalties must contain ScalarPenalty instances')
        if not isinstance(self.regularization, L2Regularization):
            raise TypeError(
                'FitModel.regularization must be an L2Regularization instance'
            )
