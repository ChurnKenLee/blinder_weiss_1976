from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss.bellman_interpolation import pchip_slopes, tensor_pchip_interpolate
from scipy.interpolate import PchipInterpolator


@pytest.mark.parametrize("axis", [0, 1])
def test_pchip_slopes_match_scipy_on_nonuniform_grid(axis: int) -> None:
    grid = np.array([0.0, 0.01, 0.2, 0.8, 2.0, 5.0])
    values = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, -0.2],
        [0.3, 0.8, -0.7],
        [0.2, 1.5, -0.7],
        [0.6, 2.0, -0.9],
        [0.9, 2.1, -2.0],
    ])
    if axis == 1:
        values = values.T
    actual = pchip_slopes(jnp.asarray(grid), jnp.asarray(values), axis=axis)
    expected = PchipInterpolator(grid, values, axis=axis).derivative()(grid)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)


@pytest.mark.parametrize("asset_nodes,log_nodes", [(2, 2), (2, 5), (6, 2), (6, 5)])
def test_tensor_pchip_matches_sequential_scipy(asset_nodes: int, log_nodes: int) -> None:
    generator = np.random.default_rng(73)
    assets = 0.001 + 7.0 * np.linspace(0.0, 1.0, asset_nodes)**2.3
    log_grid = -1.0 + 3.0 * np.linspace(0.0, 1.0, log_nodes)**1.4
    values = generator.normal(size=(asset_nodes, log_nodes))
    queries = generator.uniform([assets[0], log_grid[0]], [assets[-1], log_grid[-1]], (24, 2))
    queries[:4] = np.array([
        [assets[0] - 1.0, log_grid[0] - 1.0],
        [assets[-1] + 1.0, log_grid[-1] + 1.0],
        [assets[0], log_grid[-1]],
        [assets[-1], log_grid[0]],
    ])
    evaluate = jax.jit(tensor_pchip_interpolate)
    actual = evaluate(
        jnp.asarray(values), jnp.asarray(assets), jnp.asarray(log_grid),
        jnp.asarray(queries[:, 0]), jnp.asarray(queries[:, 1]),
    )
    asset_interpolator = PchipInterpolator(assets, values, axis=0)
    expected = np.array([
        PchipInterpolator(log_grid, asset_interpolator(np.clip(asset, assets[0], assets[-1])))(
            np.clip(log_human_capital, log_grid[0], log_grid[-1])
        )
        for asset, log_human_capital in queries
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    node_assets, node_logs = np.meshgrid(assets, log_grid, indexing="ij")
    at_nodes = evaluate(
        jnp.asarray(values), jnp.asarray(assets), jnp.asarray(log_grid),
        jnp.asarray(node_assets), jnp.asarray(node_logs),
    )
    np.testing.assert_allclose(at_nodes, values, rtol=1e-13, atol=1e-13)


def test_precomputed_slopes_and_direct_lookup_match_general_lookup() -> None:
    curvature = 3.7
    asset_grid = 1e-5 + 25.0 * np.linspace(0.0, 1.0, 31)**curvature
    log_grid = np.linspace(-1.3, 2.6, 25)
    values = jnp.asarray(np.log1p(asset_grid[:, None]) + np.exp(0.2 * log_grid[None, :]))
    assets = np.concatenate((asset_grid, np.nextafter(asset_grid, -np.inf),
                             np.nextafter(asset_grid, np.inf)))
    logs = np.linspace(log_grid[0], log_grid[-1], assets.size)
    derivatives = pchip_slopes(jnp.asarray(asset_grid), values)
    expected = tensor_pchip_interpolate(values, jnp.asarray(asset_grid), jnp.asarray(log_grid),
                                        jnp.asarray(assets), jnp.asarray(logs))
    actual = tensor_pchip_interpolate(
        values, jnp.asarray(asset_grid), jnp.asarray(log_grid), jnp.asarray(assets), jnp.asarray(logs),
        asset_derivatives=derivatives, asset_grid_curvature=curvature, uniform_log_grid=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_gradients_are_continuous_across_knots_in_monotone_region() -> None:
    assets = jnp.asarray([0.001, 0.03, 0.2, 1.0, 3.0, 8.0])
    log_grid = jnp.asarray([-1.0, -0.5, 0.1, 0.7, 1.5])
    values = (
        jnp.log1p(assets[:, None]) + 0.3 * jnp.exp(log_grid[None, :])
        + 0.05 * assets[:, None] * jnp.exp(0.3 * log_grid[None, :])
    )
    derivatives = pchip_slopes(assets, values)

    @jax.jit
    @jax.grad
    def gradient(query):
        return tensor_pchip_interpolate(
            values, assets, log_grid, query[0], query[1], asset_derivatives=derivatives
        )

    for asset in np.asarray(assets)[1:-1]:
        left = gradient(jnp.asarray([asset - 1e-8, 0.3]))
        right = gradient(jnp.asarray([asset + 1e-8, 0.3]))
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-7)
    for log_human_capital in np.asarray(log_grid)[1:-1]:
        left = gradient(jnp.asarray([0.5, log_human_capital - 1e-8]))
        right = gradient(jnp.asarray([0.5, log_human_capital + 1e-8]))
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-7)

    along_assets = PchipInterpolator(np.asarray(assets), np.asarray(values), axis=0)(0.5)
    expected_log_derivative = PchipInterpolator(np.asarray(log_grid), along_assets).derivative()(0.3)
    assert float(gradient(jnp.asarray([0.5, 0.3]))[1]) == pytest.approx(
        float(expected_log_derivative), rel=2e-13
    )


@pytest.mark.parametrize("data", [[0.0, 0.0, 0.5, 0.6, 2.0], [0.0, 0.5, -0.2, -0.2, 2.0]])
def test_one_dimensional_shape_and_finite_autodiff(data: list[float]) -> None:
    assets = jnp.asarray([0.0, 0.02, 0.3, 1.0, 3.0])
    log_grid = jnp.asarray([-1.0, 1.0])
    values = jnp.asarray(data)[:, None] + jnp.zeros((5, 2))
    queries = jnp.linspace(0.0, 3.0, 401)
    result = tensor_pchip_interpolate(values, assets, log_grid, queries, 0.0)
    indices = np.clip(np.searchsorted(np.asarray(assets), np.asarray(queries)) - 1, 0, 3)
    lower = np.minimum(np.asarray(data)[indices], np.asarray(data)[indices + 1])
    upper = np.maximum(np.asarray(data)[indices], np.asarray(data)[indices + 1])
    assert np.all(np.asarray(result) >= lower - 1e-13)
    assert np.all(np.asarray(result) <= upper + 1e-13)
    derivative = jax.grad(lambda table: jnp.sum(tensor_pchip_interpolate(
        table, assets, log_grid, queries, 0.0
    )))(values)
    assert np.all(np.isfinite(np.asarray(derivative)))
    query_derivatives = jax.vmap(jax.grad(lambda query: tensor_pchip_interpolate(
        values, assets, log_grid, query, 0.0, asset_grid_curvature=None
    )))(assets)
    assert np.all(np.isfinite(np.asarray(query_derivatives)))
