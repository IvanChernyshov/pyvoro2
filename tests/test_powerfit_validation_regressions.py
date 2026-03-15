import numpy as np
import pytest


def test_powerfit_rejects_nonfinite_points_values_and_confidence():
    from pyvoro2 import fit_power_weights, resolve_pair_bisector_constraints

    pts_bad = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts_bad, [(0, 1, 0.5)])
    with pytest.raises(ValueError, match='finite'):
        fit_power_weights(pts_bad, [(0, 1, 0.5)])

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts, [(0, 1, np.nan)])
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts, [(0, 1, 0.5)], confidence=[np.inf])



def test_powerfit_constraint_ids_must_match_points_and_be_unique():
    from pyvoro2 import resolve_pair_bisector_constraints

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )

    with pytest.raises(ValueError, match='unique'):
        resolve_pair_bisector_constraints(
            pts,
            [(10, 20, 0.5)],
            ids=[10, 10, 20],
            index_mode='id',
        )

    with pytest.raises(ValueError, match='length n_points'):
        resolve_pair_bisector_constraints(
            pts,
            [(10, 20, 0.5)],
            ids=[10, 20],
            index_mode='id',
        )



def test_zero_confidence_constraints_do_not_crash_quadratic_fit():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        confidence=[0.0],
    )

    assert res.status == 'optimal'
    assert np.allclose(res.weights, np.array([0.0, 0.0]))
    assert np.allclose(res.predicted_fraction, np.array([0.5]))
    assert any('zero-confidence' in msg for msg in res.warnings)



def test_zero_confidence_rows_do_not_join_effective_components():
    from pyvoro2 import fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25), (1, 2, 0.9)],
        measurement='fraction',
        confidence=[1.0, 0.0],
    )

    assert res.status == 'optimal'
    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.weights[2], 0.0, atol=1e-12)



def test_empty_resolved_constraints_use_regularization_only_solution():
    from pyvoro2 import FitModel, L2Regularization, fit_power_weights
    from pyvoro2.powerfit.constraints import resolve_pair_bisector_constraints

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [],
        measurement='fraction',
        allow_empty=True,
    )
    model = FitModel(
        regularization=L2Regularization(
            strength=1.0,
            reference=np.array([3.0, 5.0], dtype=float),
        )
    )

    res = fit_power_weights(pts, constraints, model=model)

    assert res.status == 'optimal'
    assert np.allclose(res.weights, np.array([3.0, 5.0]))
    assert any('regularization-only' in msg for msg in res.warnings)



def test_weight_radius_conversions_reject_nonfinite_values():
    from pyvoro2 import radii_to_weights, weights_to_radii

    with pytest.raises(ValueError, match='finite'):
        radii_to_weights(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match='finite'):
        weights_to_radii(np.array([0.0, np.inf]))
