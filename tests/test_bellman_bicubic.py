from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss.bellman_bicubic import (
    bicubic_constraint_violations,
    monotone_bicubic_interpolate,
    prepare_monotone_bicubic,
)


def _check_cell_inequalities(packed: np.ndarray, assets: np.ndarray, logs: np.ndarray) -> None:
    """Check the published sufficient inequalities independently, cell by cell."""

    values, dx, dy, twists = packed
    tolerance = 2e-10
    for axis in (0, 1):
        grid = assets if axis == 0 else logs
        other_grid = logs if axis == 0 else assets
        table = values if axis == 0 else values.T
        slopes = dx if axis == 0 else dy.T
        other_slopes = dy if axis == 0 else dx.T
        mixed = twists if axis == 0 else twists.T
        for i in range(grid.size - 1):
            width = grid[i + 1] - grid[i]
            for j in range(other_grid.size):
                secant = (table[i + 1, j] - table[i, j]) / width
                for derivative in (slopes[i, j], slopes[i + 1, j]):
                    assert min(0.0, 3.0 * secant) - tolerance <= derivative
                    assert derivative <= max(0.0, 3.0 * secant) + tolerance
            for j in range(other_grid.size - 1):
                other_width = other_grid[j + 1] - other_grid[j]
                differences = np.diff(table[i : i + 2, j : j + 2], axis=0)[0]
                if np.all(differences >= 0.0):
                    sign = 1.0
                elif np.all(differences <= 0.0):
                    sign = -1.0
                else:
                    continue
                low, high = sign * differences
                low_cross = sign * (other_slopes[i + 1, j] - other_slopes[i, j])
                high_cross = sign * (other_slopes[i + 1, j + 1] - other_slopes[i, j + 1])
                assert low_cross >= -3.0 * low / other_width - tolerance
                assert high_cross <= 3.0 * high / other_width + tolerance
                assert (
                    low_cross
                    >= -(3.0 * low - width * max(sign * slopes[i, j], sign * slopes[i + 1, j]))
                    / other_width
                    - tolerance
                )
                assert (
                    high_cross
                    <= (
                        3.0 * high
                        - width * max(sign * slopes[i, j + 1], sign * slopes[i + 1, j + 1])
                    )
                    / other_width
                    + tolerance
                )
                for endpoint in (i, i + 1):
                    lower = -3.0 * sign * slopes[endpoint, j] / other_width
                    upper = 3.0 * (
                        low_cross / width
                        + 3.0 * low / (width * other_width)
                        - sign * slopes[endpoint, j] / other_width
                    )
                    assert lower - tolerance <= sign * mixed[endpoint, j] <= upper + tolerance
                    lower = 3.0 * (
                        high_cross / width
                        - 3.0 * high / (width * other_width)
                        + sign * slopes[endpoint, j + 1] / other_width
                    )
                    upper = 3.0 * sign * slopes[endpoint, j + 1] / other_width
                    assert lower - tolerance <= sign * mixed[endpoint, j + 1] <= upper + tolerance


@pytest.mark.parametrize(
    "case", ["random", "retirement", "counterexample", "decreasing", "constant"]
)
def test_monotone_bicubic_constraints_and_dense_monotonicity(case: str) -> None:
    assets = np.array([0.0, 0.03, 0.2, 1.0, 3.0, 8.0])
    logs = np.array([-1.0, -0.5, 0.1, 0.7, 1.5])
    generator = np.random.default_rng(73)
    values = np.cumsum(np.cumsum(generator.exponential(size=(6, 5)), axis=0), axis=1)
    direction = np.array([1.0, 1.0])
    if case == "retirement":
        values = 3.0 * np.log1p(assets[:, None]) + (
            0.05 * np.maximum(1.0 - assets[:, None], 0.0) ** 2 * np.exp(logs[None, :])
        )
    elif case == "counterexample":
        assets = np.array([0.0, 1.0, 2.0])
        logs = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array(
            [[0.0, 1.0, 2.0, 3.0], [0.1, 1.001, 2.001, 3.001], [0.2, 1.002, 2.002, 3.002]]
        )
    elif case == "decreasing":
        values = values[::-1]
        direction[0] = -1.0
    elif case == "constant":
        values[:] = -3.0
    prepared = jax.jit(prepare_monotone_bicubic)(
        jnp.asarray(values), jnp.asarray(assets), jnp.asarray(logs)
    )
    assert np.all(np.isfinite(np.asarray(prepared)))
    np.testing.assert_array_equal(prepared[0], values)
    _check_cell_inequalities(np.asarray(prepared), assets, logs)
    residuals = bicubic_constraint_violations(prepared, jnp.asarray(assets), jnp.asarray(logs))
    for name, residual in residuals.items():
        if name != "value_monotonicity":
            assert float(residual) < 2e-10, name
    asset_mesh, log_mesh = np.meshgrid(
        np.linspace(assets[0], assets[-1], 101), np.linspace(logs[0], logs[-1], 97), indexing="ij"
    )
    queries = jnp.asarray(np.stack((asset_mesh.ravel(), log_mesh.ravel()), axis=1))

    def evaluate(query):
        return monotone_bicubic_interpolate(
            prepared, jnp.asarray(assets), jnp.asarray(logs), query[0], query[1]
        )

    gradients = jax.jit(jax.vmap(jax.grad(evaluate)))(queries)
    assert np.min(np.asarray(gradients) * direction) >= -2e-10
    node_assets, node_logs = np.meshgrid(assets, logs, indexing="ij")
    reconstructed = monotone_bicubic_interpolate(
        prepared,
        jnp.asarray(assets),
        jnp.asarray(logs),
        jnp.asarray(node_assets),
        jnp.asarray(node_logs),
    )
    np.testing.assert_allclose(reconstructed, values, atol=2e-13, rtol=2e-13)
    if case == "retirement":
        late_gradient = jax.grad(evaluate)(jnp.asarray([5.0, 0.3]))
        assert abs(float(late_gradient[1])) < 1e-13


@pytest.mark.parametrize("asset_nodes,log_nodes", [(2, 2), (2, 5), (6, 2), (6, 5)])
def test_bicubic_reproduces_bilinear_functions(asset_nodes: int, log_nodes: int) -> None:
    assets = jnp.linspace(0.0, 3.0, asset_nodes) ** 1.5
    logs = jnp.linspace(-1.0, 1.0, log_nodes)
    values = (
        2.0 + 1.1 * assets[:, None] + 0.7 * logs[None, :] + 0.2 * assets[:, None] * logs[None, :]
    )
    packed = prepare_monotone_bicubic(values, assets, logs)
    queries = jnp.asarray([[0.01, -0.7], [0.8, 0.15], [2.4, 0.9]])
    actual = monotone_bicubic_interpolate(packed, assets, logs, queries[:, 0], queries[:, 1])
    expected = 2.0 + 1.1 * queries[:, 0] + 0.7 * queries[:, 1] + 0.2 * queries[:, 0] * queries[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_bicubic_has_continuous_first_derivatives_across_both_cell_edges() -> None:
    assets = jnp.asarray([0.0, 0.1, 0.4, 1.0, 3.0])
    logs = jnp.asarray([-1.0, -0.3, 0.5, 1.5])
    values = jnp.log1p(assets[:, None]) + 0.2 * jnp.exp(logs[None, :]) * (1.0 + assets[:, None])
    packed = prepare_monotone_bicubic(values, assets, logs)

    @jax.jit
    @jax.grad
    def gradient(query):
        return monotone_bicubic_interpolate(packed, assets, logs, query[0], query[1])

    for asset in np.asarray(assets)[1:-1]:
        np.testing.assert_allclose(
            gradient(jnp.asarray([asset - 1e-8, 0.2])),
            gradient(jnp.asarray([asset + 1e-8, 0.2])),
            rtol=1e-6,
            atol=2e-7,
        )
    for log_value in np.asarray(logs)[1:-1]:
        np.testing.assert_allclose(
            gradient(jnp.asarray([0.8, log_value - 1e-8])),
            gradient(jnp.asarray([0.8, log_value + 1e-8])),
            rtol=1e-6,
            atol=2e-7,
        )


def test_bicubic_reports_algebraic_violation_and_handles_nonmonotone_data() -> None:
    assets = jnp.asarray([0.0, 0.1, 1.0, 3.0])
    logs = jnp.asarray([-1.0, 0.0, 1.0])
    values = jnp.asarray([[0.0, 1.0, 0.5], [0.2, -0.1, 2.0], [1.0, 0.3, -1.0], [2.0, 1.0, 3.0]])
    packed = prepare_monotone_bicubic(values, assets, logs)
    assert np.all(np.isfinite(np.asarray(packed)))
    _check_cell_inequalities(np.asarray(packed), np.asarray(assets), np.asarray(logs))
    diagnostics = bicubic_constraint_violations(packed, assets, logs)
    assert float(diagnostics["value_monotonicity"]) > 0.0
    broken = packed.at[3].set(1e6)
    diagnostics = bicubic_constraint_violations(broken, assets, logs)
    assert float(diagnostics["mixed_derivative"]) > 1e5


def test_bicubic_direct_lookup_and_parameter_autodiff_are_finite() -> None:
    assets = 1e-4 + 4.0 * jnp.linspace(0.0, 1.0, 9) ** 2.5
    logs = jnp.linspace(-1.0, 1.0, 7)
    values = jnp.log1p(assets[:, None]) + 0.1 * jnp.maximum(logs[None, :], 0.0) ** 2
    packed = prepare_monotone_bicubic(values, assets, logs)
    queries = jnp.asarray([[0.03, -0.7], [0.8, 0.15], [2.4, 0.9]])
    general = monotone_bicubic_interpolate(packed, assets, logs, queries[:, 0], queries[:, 1])
    direct = monotone_bicubic_interpolate(
        packed,
        assets,
        logs,
        queries[:, 0],
        queries[:, 1],
        asset_grid_curvature=2.5,
        uniform_log_grid=True,
    )
    np.testing.assert_allclose(direct, general, rtol=2e-14, atol=2e-14)

    def objective(scale):
        prepared = prepare_monotone_bicubic(values * scale, assets, logs)
        return jnp.sum(
            monotone_bicubic_interpolate(prepared, assets, logs, queries[:, 0], queries[:, 1])
        )

    assert np.isfinite(float(jax.grad(objective)(1.0)))


def test_one_ulp_retirement_plateau_noise_cannot_disable_shape_constraints() -> None:
    """A reduced age-69 table used to amplify roundoff into dV/dK=-0.001426."""

    assets = jnp.asarray(
        [4.70564211111111, 5.600084000000001, 6.572303444444444, 7.622300444444444]
    )
    logs = jnp.asarray([1.541666666666667, 1.71875, 1.8958333333333335, 2.072916666666667, 2.25])
    # These retained discrepancies are only one or two ULPs within retirement
    # plateaus. The older exact-sign mask skipped the surrounding mixed bounds.
    values = jnp.asarray(
        [
            [
                -1.7858269429731248,
                -1.7858269429731253,
                -1.7849345935395777,
                -1.778108976164038,
                -1.7649497462319579,
            ],
            [
                -1.657943147153974,
                -1.657943147153974,
                -1.657943147153974,
                -1.6579431471539745,
                -1.6570268484564896,
            ],
            [
                -1.5584188824158896,
                -1.558418882415889,
                -1.5584188824158896,
                -1.558418882415889,
                -1.5584188824158896,
            ],
            [
                -1.4794490509259512,
                -1.4794490509259517,
                -1.4794490509259512,
                -1.4794490509259512,
                -1.4794490509259512,
            ],
        ]
    )
    asset_mesh, log_mesh = jnp.meshgrid(
        jnp.linspace(assets[0], assets[-1], 91), jnp.linspace(logs[0], logs[-1], 91), indexing="ij"
    )
    queries = jnp.stack((asset_mesh.ravel(), log_mesh.ravel()), axis=1)

    @jax.jit
    def evaluate_gradients(packed):
        def evaluate(query):
            return monotone_bicubic_interpolate(packed, assets, logs, query[0], query[1])

        return jax.vmap(jax.grad(evaluate))(queries)

    reference = prepare_monotone_bicubic(values, assets, logs)
    for scale in (1.0, 1e-20, 1e20):
        scaled_values = values * scale
        packed = prepare_monotone_bicubic(scaled_values, assets, logs)
        np.testing.assert_array_equal(packed[0], scaled_values)
        assert np.all(np.isfinite(np.asarray(packed)))
        gradients = np.asarray(evaluate_gradients(packed)) / scale
        assert np.min(gradients[:, 0]) >= -2e-13
        assert np.min(gradients[:, 1]) >= -2e-13
        # Relative scaling, rather than an absolute tolerance floor, keeps real
        # derivatives when the entire value table has a very small magnitude.
        np.testing.assert_allclose(
            np.asarray(packed[1:]) / scale, reference[1:], rtol=2e-11, atol=2e-13
        )
        diagnostics = bicubic_constraint_violations(packed, assets, logs)
        assert float(diagnostics["value_monotonicity"]) > 0.0
        assert float(diagnostics["mixed_derivative"]) / scale < 2e-12
        assert float(diagnostics["mixed_bound_intersection"]) / scale < 2e-12
