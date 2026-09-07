from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import (
    BellmanConfig,
    benchmark_params,
    constant_control_transition,
    solve_bellman,
)
from blinder_weiss.bellman import _make_control_optimizer, bellman_state_grids


def _retirement_setup():
    params = benchmark_params(
        horizon=1.0,
        rho=0.0,
        interest_rate=0.0,
        human_capital_depreciation=0.0,
        human_capital_productivity=0.0,
        leisure_weight=100.0,
        initial_assets=4.0,
    )
    config = BellmanConfig(
        periods=1,
        asset_nodes=3,
        human_capital_nodes=9,
        asset_minimum=1e-4,
        asset_maximum=6.0,
        asset_grid_curvature=1.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=2,
        investment_nodes=2,
        consumption_nodes=2,
        refinement_steps=0,
        refinement_starts=1,
        control_batch_size=8,
        value_interpolation="monotone_bicubic",
        compute_platform="cpu",
    )
    return params, config


@pytest.mark.parametrize("adaptive_tolerance", [0.0, 1e-13])
def test_adaptive_neighbors_propagate_a_feasible_retirement_policy_across_the_batch(
    adaptive_tolerance: float,
) -> None:
    params, base = _retirement_setup()
    asset_grid_np, log_grid_np = bellman_state_grids(base)
    asset_grid, log_grid = jnp.asarray(asset_grid_np), jnp.asarray(log_grid_np)
    # V(A,K)=A makes every retirement state's objective identical. With the
    # zero rates above, c=1 uniquely maximizes -1/c + (4-c) - leisure_weight.
    continuation = jnp.broadcast_to(
        asset_grid[:, None], (base.asset_nodes, base.human_capital_nodes)
    )
    states = jnp.stack((jnp.full((1, base.human_capital_nodes), 4.0), log_grid[None, :]), axis=-1)
    incumbent = jnp.zeros((1, base.human_capital_nodes, 3))
    incumbent = incumbent.at[..., 0].set(1.4).at[0, 0, 0].set(1.0)
    results = []
    for sweeps, tolerance in ((2, None), (16, None), (64, adaptive_tolerance)):
        config = replace(base, neighbor_policy_sweeps=sweeps, neighbor_policy_tolerance=tolerance)
        optimizer = _make_control_optimizer(
            params, config, asset_grid, log_grid, neighbor_shape=(1, base.human_capital_nodes)
        )
        result = jax.device_get(jax.jit(optimizer)(states, continuation, incumbent, True, False))
        results.append(result)

    short, fixed, adaptive = results
    # The fixed two sweeps move the superior candidate only two columns. This
    # reproduces the propagation failure without running an entire lifecycle.
    np.testing.assert_allclose(short[1][0, :3], 1.0, atol=1e-13)
    assert short[1][0, -1] == pytest.approx(1.4)
    assert adaptive[0][0, -1] > short[0][0, -1] + 0.1
    for fixed_array, adaptive_array in zip(fixed, adaptive, strict=True):
        np.testing.assert_allclose(adaptive_array, fixed_array, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(adaptive[1], 1.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(adaptive[0], -98.0, rtol=0.0, atol=1e-12)

    for value, consumption, hours, training in results:
        assert all(np.all(np.isfinite(array)) for array in (value, consumption, hours, training))
        assert np.all(consumption >= base.consumption_floor)
        np.testing.assert_array_equal(hours, np.zeros_like(hours))
        np.testing.assert_array_equal(training, np.zeros_like(training))
        controls = jnp.stack(
            (jnp.asarray(consumption), jnp.asarray(hours), jnp.asarray(training)), axis=-1
        )
        next_states = np.asarray(constant_control_transition(states, controls, params, 1.0))
        expected_value = (
            -1.0 / consumption - params.leisure_weight / (1.0 - hours) + next_states[..., 0]
        )
        np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=1e-12)
        assert np.all(next_states[..., 0] >= asset_grid_np[0])
        assert np.all(next_states[..., 0] <= asset_grid_np[-1])
        np.testing.assert_array_equal(next_states[..., 1], np.asarray(states[..., 1]))
        # Independent within-period check: for h=q=r=0, A(s)=4-c*s exactly.
        checkpoint_assets = 4.0 - consumption[..., None] * np.linspace(0.0, 1.0, 33)
        assert np.min(checkpoint_assets) >= base.asset_minimum


def test_exact_neighbor_stopping_propagates_improvements_below_relative_tolerance() -> None:
    params, base = _retirement_setup()
    asset_grid_np, log_grid_np = bellman_state_grids(base)
    asset_grid, log_grid = jnp.asarray(asset_grid_np), jnp.asarray(log_grid_np)
    continuation = jnp.broadcast_to(
        asset_grid[:, None], (base.asset_nodes, base.human_capital_nodes)
    )
    states = jnp.stack((jnp.full((1, base.human_capital_nodes), 4.0), log_grid[None, :]), axis=-1)
    # The exact maximizer c=1 improves the other incumbents by about 1e-12:
    # larger than roundoff, but smaller than 1e-13 * abs(V) for V near -98.
    # A tolerance-based stop can leave this candidate only partly propagated,
    # introducing real nodal declines into an exactly flat continuation.
    inferior_consumption = 1.0 + 1e-6
    incumbent = jnp.zeros((1, base.human_capital_nodes, 3))
    incumbent = incumbent.at[..., 0].set(inferior_consumption).at[0, 0, 0].set(1.0)
    results = []
    for tolerance in (1e-13, 0.0):
        config = replace(base, neighbor_policy_sweeps=64, neighbor_policy_tolerance=tolerance)
        optimizer = _make_control_optimizer(
            params, config, asset_grid, log_grid, neighbor_shape=(1, base.human_capital_nodes)
        )
        results.append(
            jax.device_get(jax.jit(optimizer)(states, continuation, incumbent, True, False))
        )

    tolerant, exact = results
    np.testing.assert_allclose(tolerant[1][0, :2], 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(tolerant[1][0, 2:], inferior_consumption, rtol=0.0, atol=1e-14)
    gap = float(exact[0][0, -1] - tolerant[0][0, -1])
    assert 1e-13 < gap < 1e-13 * abs(tolerant[0][0, -1])
    assert np.min(np.diff(tolerant[0], axis=1)) < -1e-13
    np.testing.assert_allclose(exact[1], 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(exact[0], np.full_like(exact[0], exact[0][0, 0]))
    for value, consumption, hours, training in results:
        assert all(np.all(np.isfinite(array)) for array in (value, consumption, hours, training))
        np.testing.assert_array_equal(hours, np.zeros_like(hours))
        np.testing.assert_array_equal(training, np.zeros_like(training))
        expected_value = -1.0 / consumption - params.leisure_weight + (4.0 - consumption)
        np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=3e-14)
        assert np.all(consumption >= base.consumption_floor)
        assert np.all(4.0 - consumption >= base.asset_minimum)
        assert np.all(4.0 - consumption <= base.asset_maximum)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_neighbor_tolerance_is_rejected_before_solving(tolerance: float) -> None:
    params, config = _retirement_setup()
    with pytest.raises(
        ValueError, match="neighbor_policy_tolerance must be finite and nonnegative"
    ):
        solve_bellman(params, replace(config, neighbor_policy_tolerance=tolerance))
