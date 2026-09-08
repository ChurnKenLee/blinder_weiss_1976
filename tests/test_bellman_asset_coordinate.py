"""Independent shape, CRRA marginal utility, and solver consistency checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, diagnose_bellman, solve_bellman, value_at
from blinder_weiss.bellman_bicubic import (
    asset_utility_coordinate,
    bicubic_constraint_violations,
    monotone_bicubic_interpolate,
    prepare_monotone_bicubic,
)
from blinder_weiss.retirement import retirement_reference


@pytest.mark.parametrize("power", [-1.0, -0.3, 0.4])
def test_reproduces_exact_retirement_value_and_marginal_utility(power):
    params = benchmark_params(consumption_power=power, bequest_power=power)
    reference = retirement_reference(params, periods=30, step=0.5)
    assets = jnp.asarray([0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 35.0])
    logs = jnp.linspace(-4.0, 1.0, 4)
    values = jnp.broadcast_to(jnp.asarray(reference.value(assets))[:, None], (8, 4))
    prepared = prepare_monotone_bicubic(values, assets, logs, asset_power=power)
    queries = jnp.stack((jnp.linspace(0.11, 34.9, 200), jnp.full(200, -1.0)), axis=1)

    def evaluate(query):
        return monotone_bicubic_interpolate(
            prepared, assets, logs, query[0], query[1], asset_power=power
        )

    actual = jax.jit(jax.vmap(evaluate))(queries)
    gradients = jax.jit(jax.vmap(jax.grad(evaluate)))(queries)
    np.testing.assert_allclose(actual, reference.value(queries[:, 0]), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        gradients[:, 0], reference.marginal_asset_value(queries[:, 0]), rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(gradients[:, 1], 0.0, atol=2e-12)


@pytest.mark.parametrize("power", [-1.0, 0.0, 0.4])
def test_transformed_coordinate_preserves_both_shapes_and_fast_lookup(power):
    assets = 1e-3 + 4.0 * jnp.linspace(0.0, 1.0, 9) ** 2.5
    logs = jnp.linspace(-1.0, 1.0, 7)
    values = jnp.log1p(assets[:, None]) + 0.1 * jnp.exp(logs[None, :])
    packed = prepare_monotone_bicubic(values, assets, logs, asset_power=power)
    coordinate = asset_utility_coordinate(assets, power)
    assert np.all(np.diff(coordinate) > 0.0)
    residuals = bicubic_constraint_violations(packed, assets, logs, asset_power=power)
    assert max(map(float, residuals.values())) < 2e-10
    mesh = jnp.meshgrid(jnp.linspace(0.002, 4.0, 97), jnp.linspace(-0.99, 0.99, 73))
    queries = jnp.stack([axis.ravel() for axis in mesh], axis=1)

    def evaluate(query, fast):
        return monotone_bicubic_interpolate(
            packed,
            assets,
            logs,
            query[0],
            query[1],
            asset_power=power,
            asset_grid_curvature=2.5 if fast else None,
            uniform_log_grid=fast,
        )

    fast = jax.jit(jax.vmap(lambda query: evaluate(query, True)))(queries)
    general = jax.jit(jax.vmap(lambda query: evaluate(query, False)))(queries)
    np.testing.assert_allclose(fast, general, rtol=2e-13, atol=2e-13)
    gradients = jax.jit(jax.vmap(jax.grad(lambda query: evaluate(query, True))))(queries)
    assert np.all(np.isfinite(gradients))
    assert np.min(gradients) >= -2e-12


def test_transformed_recursion_queries_and_bellman_identity_agree():
    config = BellmanConfig(
        periods=3,
        asset_nodes=7,
        human_capital_nodes=5,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=5,
        refinement_steps=4,
        value_interpolation="monotone_bicubic",
        bicubic_asset_power=-1.0,
        compute_platform="cpu",
    )
    solution = solve_bellman(benchmark_params(horizon=3.0), config)
    diagnostics = diagnose_bellman(solution)
    assert diagnostics.accepted_node_solution
    assert diagnostics.maximum_node_bellman_residual < 1e-9
    assert diagnostics.simulation_stayed_in_domain
    assets, log_k = jnp.asarray(solution.asset_grid), jnp.asarray(solution.log_human_capital_grid)
    packed = prepare_monotone_bicubic(solution.values[0], assets, log_k, asset_power=-1.0)
    expected = monotone_bicubic_interpolate(packed, assets, log_k, 2.345, 0.134, asset_power=-1.0)
    assert float(value_at(solution, 0, 2.345, np.exp(0.134))) == pytest.approx(float(expected))
    terminal_config = replace(config, periods=1)
    transformed = solve_bellman(benchmark_params(horizon=1.0), terminal_config)
    identity = solve_bellman(
        benchmark_params(horizon=1.0), replace(terminal_config, bicubic_asset_power=1.0)
    )
    np.testing.assert_allclose(transformed.values, identity.values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"bicubic_asset_power": float("nan")}, "finite"),
        ({"bicubic_asset_power": -1.0, "value_interpolation": "bilinear"}, "monotone_bicubic"),
    ],
)
def test_invalid_asset_coordinate_cannot_silently_change_representation(overrides, match):
    with pytest.raises(ValueError, match=match):
        solve_bellman(config=BellmanConfig(**overrides))
