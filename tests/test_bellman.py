from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import (
    BellmanConfig,
    benchmark_params,
    bequest_utility,
    constant_control_transition,
    diagnose_bellman,
    greedy_policy_at,
    policy_at,
    simulate_policy,
    solve_bellman,
)
from blinder_weiss.bellman import maximum_feasible_consumption
from blinder_weiss.model import effective_earnings_share


@pytest.fixture(scope="module")
def small_bellman_solution():
    params = benchmark_params(horizon=8.0)
    config = BellmanConfig(
        periods=8,
        asset_nodes=9,
        human_capital_nodes=7,
        asset_minimum=1e-3,
        asset_maximum=20.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.5,
        hours_nodes=7,
        investment_nodes=7,
        consumption_nodes=9,
        refinement_steps=4,
        compute_platform="cpu",
    )
    return solve_bellman(params, config)


def test_exact_transition_matches_constant_human_capital_solution() -> None:
    params = benchmark_params()
    step = 1.75
    hours = 0.6
    training_time = params.human_capital_depreciation / params.human_capital_productivity
    consumption = 0.8
    state = jnp.asarray([4.0, np.log(1.4)])
    control = jnp.asarray([consumption, hours, training_time])
    transition = constant_control_transition(state, control, params, step)

    asset_factor = np.exp(params.interest_rate * step)
    annuity_factor = np.expm1(params.interest_rate * step) / params.interest_rate
    constant_earnings = float(
        effective_earnings_share(jnp.asarray(hours), jnp.asarray(training_time)) * jnp.exp(state[1])
    )
    expected_assets = asset_factor * state[0] + annuity_factor * (constant_earnings - consumption)
    assert float(transition[0]) == pytest.approx(float(expected_assets), rel=1e-12)
    assert float(transition[1]) == pytest.approx(float(state[1]), abs=1e-12)


def test_consumption_capacity_enforces_all_path_checkpoints() -> None:
    params = benchmark_params()
    step = 2.0
    asset_minimum = 1e-3
    checkpoints = 8
    assets = 0.5
    log_human_capital = np.log(1.2)
    hours = 0.55
    training_time = 0.2
    capacity = float(
        maximum_feasible_consumption(
            jnp.asarray(assets),
            jnp.asarray(log_human_capital),
            jnp.asarray(hours),
            jnp.asarray(training_time),
            params,
            step,
            asset_minimum,
            checkpoints,
        )
    )
    checkpoint_assets = []
    for duration in np.linspace(step / checkpoints, step, checkpoints):
        next_state = constant_control_transition(
            jnp.asarray([assets, log_human_capital]),
            jnp.asarray([capacity, hours, training_time]),
            params,
            float(duration),
        )
        checkpoint_assets.append(float(next_state[0]))
    assert min(checkpoint_assets) >= asset_minimum - 1e-11
    assert min(checkpoint_assets) == pytest.approx(asset_minimum, abs=1e-10)


def test_requiring_unavailable_gpu_fails_before_solving() -> None:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    gpu_devices = [device for device in gpu_devices if device.platform == "gpu"]
    if not gpu_devices:
        with pytest.raises(RuntimeError, match="could not initialize a CUDA device"):
            solve_bellman(config=BellmanConfig(compute_platform="gpu"))
    else:
        pytest.skip("CUDA is available, so the unavailable-GPU branch does not apply")


def test_bellman_solution_contains_finite_values_and_feasible_policies(
    small_bellman_solution,
) -> None:
    solution = small_bellman_solution
    expected_shape = (
        solution.config.periods + 1,
        solution.config.asset_nodes,
        solution.config.human_capital_nodes,
    )
    assert solution.values.shape == expected_shape
    assert solution.backend == "cpu"
    assert solution.device.startswith("cpu:0")
    assert solution.consumption_policy.shape == (
        solution.config.periods,
        solution.config.asset_nodes,
        solution.config.human_capital_nodes,
    )
    assert np.all(np.isfinite(solution.values))
    assert np.all(solution.consumption_policy > 0.0)
    assert np.all(solution.hours_policy >= 0.0)
    assert np.all(solution.hours_policy < 1.0)
    assert np.all(solution.training_time_policy >= 0.0)
    assert np.all(solution.training_time_policy <= solution.hours_policy + 1e-12)
    assert np.min(np.diff(solution.values, axis=1)) >= -1e-10

    terminal_assets = jnp.asarray(solution.asset_grid[:, None])
    expected_terminal = np.broadcast_to(
        np.asarray(bequest_utility(terminal_assets, solution.params)),
        solution.values[-1].shape,
    )
    np.testing.assert_allclose(solution.values[-1], expected_terminal)


def test_greedy_and_interpolated_policies_simulate_inside_domain(
    small_bellman_solution,
) -> None:
    solution = small_bellman_solution
    simulation = simulate_policy(solution)
    interpolated_simulation = simulate_policy(solution, policy_method="interpolate")
    consumption, hours, training_time = policy_at(
        solution,
        0,
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([0.9, 1.0, 1.1]),
    )
    assert consumption.shape == (3,)
    assert np.all(hours >= 0.0)
    assert np.all(training_time >= 0.0)
    assert np.all(training_time <= hours + 1e-12)
    greedy_consumption, greedy_hours, greedy_training = greedy_policy_at(
        solution,
        0,
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([0.9, 1.0, 1.1]),
    )
    assert greedy_consumption.shape == (3,)
    assert np.all(greedy_consumption > 0.0)
    assert np.all(greedy_hours >= 0.0)
    assert np.all(greedy_training >= 0.0)
    assert np.all(greedy_training <= greedy_hours + 1e-12)

    assert simulation.policy_method == "greedy"
    assert simulation.stayed_in_domain
    assert simulation.minimum_assets >= solution.config.asset_minimum - 1e-9
    assert np.all(simulation.consumption > 0.0)
    assert np.all(simulation.training_time <= simulation.hours + 1e-12)
    assert np.isfinite(simulation.lifetime_utility)
    assert np.isfinite(simulation.initial_value)
    assert np.isfinite(simulation.value_gap)
    assert interpolated_simulation.policy_method == "interpolate"
    assert interpolated_simulation.stayed_in_domain
    assert np.isfinite(interpolated_simulation.lifetime_utility)
    diagnostics = diagnose_bellman(solution, simulation)
    assert diagnostics.accepted_node_solution
    assert diagnostics.maximum_node_bellman_residual < 1e-10
    assert diagnostics.minimum_consumption_capacity_slack >= -1e-10
