from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import (
    BellmanConfig,
    BellmanConvergenceConfig,
    bellman_refinement_config,
    benchmark_params,
    bequest_utility,
    constant_control_transition,
    diagnose_bellman,
    greedy_policy_at,
    greedy_policy_value_at,
    policy_at,
    simulate_policy,
    solve_bellman,
    solve_bellman_converged,
    value_at,
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"refinement_backtracking_steps": 0}, "backtracking_steps must be positive"),
        ({"refinement_backtracking_factor": 0.0}, "backtracking_factor must lie"),
        ({"refinement_backtracking_factor": 1.0}, "backtracking_factor must lie"),
        ({"refinement_starts": 0}, "refinement_starts must be positive"),
        ({"control_batch_size": 0}, "control_batch_size must be positive"),
    ],
)
def test_invalid_refinement_backtracking_config_is_rejected(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_bellman(config=BellmanConfig(compute_platform="cpu", **overrides))


def test_backtracked_refinement_weakly_improves_discrete_controls() -> None:
    params = benchmark_params(horizon=1.0)
    coarse_config = BellmanConfig(
        periods=1,
        asset_nodes=7,
        human_capital_nodes=5,
        asset_minimum=1e-3,
        asset_maximum=12.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=7,
        refinement_steps=0,
        neighbor_policy_sweeps=0,
        compute_platform="cpu",
    )
    coarse = solve_bellman(params, coarse_config)
    refined = solve_bellman(
        params,
        replace(
            coarse_config,
            refinement_steps=8,
            refinement_learning_rate=0.01,
        ),
    )

    improvement = refined.values[0] - coarse.values[0]
    assert np.min(improvement) >= -1e-11
    assert np.max(improvement) > 1e-8


def test_backtracking_recovers_from_an_oversized_refinement_step() -> None:
    params = benchmark_params(horizon=1.0)
    config = BellmanConfig(
        periods=1,
        asset_nodes=7,
        human_capital_nodes=5,
        asset_minimum=1e-3,
        asset_maximum=12.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=7,
        refinement_steps=1,
        refinement_learning_rate=0.5,
        refinement_backtracking_steps=1,
        neighbor_policy_sweeps=0,
        compute_platform="cpu",
    )
    full_step = solve_bellman(params, config)
    backtracked = solve_bellman(
        params,
        replace(config, refinement_backtracking_steps=10),
    )

    improvement = backtracked.values[0] - full_step.values[0]
    assert np.min(improvement) >= -1e-11
    assert np.max(improvement) > 1e-8


def test_batched_control_search_matches_single_batch() -> None:
    params = benchmark_params(horizon=1.0)
    config = BellmanConfig(
        periods=1,
        asset_nodes=5,
        human_capital_nodes=4,
        asset_minimum=1e-3,
        asset_maximum=12.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=7,
        refinement_steps=0,
        refinement_starts=1,
        neighbor_policy_sweeps=0,
        compute_platform="cpu",
    )
    batched = solve_bellman(params, replace(config, control_batch_size=14))
    single_batch = solve_bellman(params, replace(config, control_batch_size=10_000))

    np.testing.assert_allclose(batched.values, single_batch.values, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        batched.consumption_policy,
        single_batch.consumption_policy,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(batched.hours_policy, single_batch.hours_policy, atol=1e-12)
    np.testing.assert_allclose(
        batched.training_time_policy,
        single_batch.training_time_policy,
        atol=1e-12,
    )


def test_multistart_refinement_never_lowers_node_values() -> None:
    params = benchmark_params(horizon=1.0)
    config = BellmanConfig(
        periods=1,
        asset_nodes=7,
        human_capital_nodes=5,
        asset_minimum=1e-3,
        asset_maximum=12.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=7,
        refinement_steps=8,
        refinement_learning_rate=0.01,
        refinement_starts=1,
        neighbor_policy_sweeps=0,
        compute_platform="cpu",
    )
    single_start = solve_bellman(params, config)
    multi_start = solve_bellman(params, replace(config, refinement_starts=4))

    assert np.min(multi_start.values - single_start.values) >= -1e-11


def test_value_and_greedy_value_queries_return_consistent_shapes(
    small_bellman_solution,
) -> None:
    solution = small_bellman_solution
    assets = np.asarray([4.0, 5.0, 6.0])
    human_capital = np.asarray([0.9, 1.0, 1.1])
    interpolated_value = value_at(solution, 0, assets, human_capital)
    greedy_value, consumption, hours, training_time = greedy_policy_value_at(
        solution,
        0,
        assets,
        human_capital,
    )

    assert interpolated_value.shape == assets.shape
    assert greedy_value.shape == assets.shape
    assert np.all(np.isfinite(interpolated_value))
    assert np.all(np.isfinite(greedy_value))
    assert np.all(consumption > 0.0)
    assert np.all(training_time <= hours + 1e-12)
    terminal_assets = np.asarray([4.25, 5.5, 7.75])
    terminal_off_grid = value_at(
        solution,
        solution.config.periods,
        terminal_assets,
        np.ones_like(terminal_assets),
    )
    np.testing.assert_allclose(
        terminal_off_grid,
        np.asarray(bequest_utility(jnp.asarray(terminal_assets), solution.params)),
    )


def test_refinement_schedule_jointly_expands_domain_and_resolution() -> None:
    params = benchmark_params()
    target = BellmanConfig(compute_platform="cpu")
    convergence = BellmanConvergenceConfig()
    level_zero = bellman_refinement_config(params, target, convergence, 0)
    level_two = bellman_refinement_config(params, target, convergence, 2)

    assert level_zero.periods == target.periods
    assert level_zero.asset_minimum < target.asset_minimum
    assert level_zero.asset_maximum > target.asset_maximum
    assert level_two.periods == 4 * target.periods
    assert level_two.asset_nodes == 1 + 3 * (target.asset_nodes - 1)
    assert level_two.human_capital_nodes == 1 + 3 * (target.human_capital_nodes - 1)
    assert level_two.hours_nodes == target.hours_nodes + 4
    assert level_two.consumption_nodes == target.consumption_nodes + 8
    assert level_two.refinement_starts == 4
    assert level_two.path_checkpoints == 4 * target.path_checkpoints
    assert level_two.asset_minimum < level_zero.asset_minimum
    assert level_two.asset_maximum > level_zero.asset_maximum
    assert level_zero.asset_grid_curvature > target.asset_grid_curvature


def test_small_convergence_driver_reports_finest_solution() -> None:
    params = benchmark_params(horizon=1.0)
    target = BellmanConfig(
        periods=1,
        asset_nodes=3,
        human_capital_nodes=3,
        asset_minimum=1e-3,
        asset_maximum=10.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=3,
        investment_nodes=3,
        consumption_nodes=3,
        path_checkpoints=1,
        refinement_steps=1,
        neighbor_policy_sweeps=0,
        compute_platform="cpu",
    )
    convergence = BellmanConvergenceConfig(
        max_levels=2,
        required_consecutive_passes=1,
        value_tolerance=1e9,
        policy_regret_tolerance=1e9,
        monotonicity_tolerance=1e9,
        control_batch_size=16,
    )
    result = solve_bellman_converged(params, target, convergence)

    assert result.converged
    assert len(result.levels) == 2
    assert result.solution is result.levels[-1].solution
    assert result.levels[0].normalized_value_change is None
    assert result.levels[1].normalized_value_change is not None
    assert result.levels[1].maximum_policy_regret is not None
    assert result.levels[1].target_next_states_inside_domain
    assert result.levels[1].passed


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
