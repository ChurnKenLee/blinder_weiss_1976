import numpy as np
import pytest
from blinder_weiss.bellman_smoothness import retirement_euler_diagnostics
from blinder_weiss.model import benchmark_params
from blinder_weiss.retirement import retirement_reference


def test_exact_retirement_and_alternating_jitter_have_independent_known_errors():
    params = benchmark_params()
    path = retirement_reference(params, periods=8, step=0.5).path(5.0)
    perturbation = 0.003 * (-1.0) ** np.arange(8)
    consumption = np.stack((path.consumption, path.consumption * np.exp(perturbation)), axis=1)
    diagnostic = retirement_euler_diagnostics(
        path.time,
        consumption,
        np.zeros_like(consumption),
        np.broadcast_to(path.assets[:, None], (9, 2)),
        params,
        asset_minimum=1e-4,
    )
    np.testing.assert_array_equal(diagnostic.count, [7, 7])
    np.testing.assert_allclose(diagnostic.residuals[:, 0], 0.0, atol=1e-13)
    np.testing.assert_allclose(diagnostic.residuals[:, 1], np.diff(perturbation) / 0.5, atol=1e-13)
    np.testing.assert_allclose(diagnostic.root_mean_square, [0.0, 0.012], atol=1e-13)
    np.testing.assert_allclose(diagnostic.maximum_absolute, [0.0, 0.012], atol=1e-13)


def test_excludes_working_periods_and_every_endpoint_of_floor_contacts():
    diagnostic = retirement_euler_diagnostics(
        np.arange(7),
        np.ones((6, 2)),
        np.array([[0, 0.1], [0, 0.1], [0.1, 0.1], [0, 0.1], [0, 0.1], [0, 0.1]]),
        np.array([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1e-4, 2], [2, 2]]),
        benchmark_params(),
        asset_minimum=1e-4,
    )
    np.testing.assert_array_equal(diagnostic.eligible[:, 0], [True, False, False, False, False])
    np.testing.assert_array_equal(diagnostic.count, [1, 0])
    assert diagnostic.root_mean_square[0] == pytest.approx(0.01)
    assert np.isnan(diagnostic.root_mean_square[1])
    assert np.isnan(diagnostic.maximum_absolute[1])


@pytest.mark.parametrize("time", [[0, 0.5, 1.1], [0, 0, 1], [0, np.nan, 1]])
def test_rejects_unsupported_time_grids(time):
    with pytest.raises(ValueError):
        retirement_euler_diagnostics(
            time, [1, 1], [0, 0], [2, 2, 2], benchmark_params(), asset_minimum=1e-4
        )
