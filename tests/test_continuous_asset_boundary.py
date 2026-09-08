from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, diagnose_bellman, solve_bellman
from blinder_weiss.bellman import (
    constant_control_transition,
    endpoint_consumption_capacity,
    maximum_feasible_consumption,
    minimum_assets_during_step,
)
from blinder_weiss.model import effective_earnings_share
from scipy.integrate import quad
from scipy.optimize import OptimizeResult, brentq, minimize_scalar


def exprel(value):
    return float(np.expm1(value) / value) if value != 0 else 1.0


def reference_capacity(assets, log_k, hours, training, params, step, floor):
    growth = params.human_capital_productivity * training - params.human_capital_depreciation
    earnings = float(effective_earnings_share(jnp.asarray(hours), jnp.asarray(training)))
    earnings *= np.exp(log_k)
    rate = params.interest_rate
    endpoint = (
        rate * floor + (assets - floor) / (step * exprel(-rate * step))
        + earnings * exprel((growth - rate) * step) / exprel(-rate * step)
    )
    if growth <= 0 or earnings == 0:
        return endpoint
    if assets == floor:
        return rate * floor + earnings

    # Independently integrate a nonnegative discounted-income difference,
    # avoiding the implementation's subtraction of exponential averages.
    def buffer_needed(time):
        return earnings * quad(
            lambda s: np.exp((growth - rate) * s) * np.expm1(growth * (time - s)),
            0, time, epsabs=1e-25, epsrel=1e-12,
        )[0]

    buffer = assets - floor
    if buffer_needed(step) <= buffer:
        return endpoint
    touching_time = brentq(lambda t: buffer_needed(t) - buffer, 0, step, xtol=1e-15)
    return rate * floor + earnings * np.exp(growth * touching_time)


@pytest.mark.parametrize(
    "assets,hours,training,rate,step",
    [
        (1e-4, 0.7, 0.45, 0.05, 0.5),
        (5e-4, 0.7, 0.45, 0.05, 0.5),
        (2.0, 0.7, 0.45, 0.05, 0.5),
        (5e-4, 0.8, 0.1 / 0.22, 0.05, 0.5),
        (5e-4, 0.8, (0.1 + 1e-12) / 0.22, 0.05, 0.5),
        (5e-4, 0.8, (0.1 - 1e-12) / 0.22, 0.05, 0.5),
        (5e-4, 0.7, 0.45, 0.0, 0.5),
        (5e-4, 0.7, 0.45, -0.02, 0.5),
        (1.0, 0.7, 0.05 / 0.22, 0.0, 0.5),
        (1.0, 0.7, 0.05 / 0.22, 0.05, 0.5),
        (1.0, 0.0, 0.0, 0.0, 0.5),
        (1.0, 0.7, 0.7, 0.05, 0.5),
        (1.0, 0.7, 0.7, -0.02, 0.5),
        (1e-4 + 1e-14, 0.7, (0.05 + 1e-12) / 0.22, 0.05, 2.0),
    ],
)
def test_continuous_capacity_matches_independent_integral_root(
    assets, hours, training, rate, step
):
    params = benchmark_params(interest_rate=rate)
    floor = 1e-4
    log_k = 0.13
    capacity = float(maximum_feasible_consumption(
        jnp.asarray(assets), jnp.asarray(log_k), jnp.asarray(hours), jnp.asarray(training),
        params, step, floor, 4,
    ))
    reference = reference_capacity(assets, log_k, hours, training, params, step, floor)
    assert capacity == pytest.approx(reference, abs=3e-13, rel=3e-13)
    state = jnp.array([assets, log_k])
    control = jnp.array([capacity, hours, training])
    minimum = float(minimum_assets_during_step(state, control, params, step))
    assert minimum >= floor - 3e-13
    assert minimum == pytest.approx(floor, abs=3e-12)
    excessive = control.at[0].add(1e-4 * max(1, abs(capacity)))
    assert float(minimum_assets_during_step(state, excessive, params, step)) < floor - 1e-10


def test_previous_checkpoint_floor_policy_has_a_real_between_checkpoint_dip():
    params = benchmark_params()
    floor = 1e-4
    state = jnp.array([floor, 0.0])
    h, q = jnp.asarray(0.7072046322267), jnp.asarray(0.4536915522068)
    legacy = maximum_feasible_consumption(state[0], state[1], h, q, params, 0.5, floor, 4,
                                           method="checkpoints")
    exact = maximum_feasible_consumption(state[0], state[1], h, q, params, 0.5, floor, 4)
    legacy_minimum = minimum_assets_during_step(state, jnp.array([legacy, h, q]), params, 0.5)
    assert float(legacy_minimum) < floor - 3e-5
    assert float(exact) < float(legacy) - 9e-4
    assert float(minimum_assets_during_step(state, jnp.array([exact, h, q]), params, 0.5)) >= floor
    checkpoint_states = jax.vmap(
        lambda duration: constant_control_transition(state, jnp.array([legacy, h, q]), params,
                                                      duration)
    )(jnp.array([0.125, 0.25, 0.375, 0.5]))
    assert np.min(np.asarray(checkpoint_states[:, 0])) >= floor - 1e-14
    for checkpoints in (1, 4, 32):
        np.testing.assert_allclose(
            maximum_feasible_consumption(state[0], state[1], h, q, params, 0.5, floor, checkpoints),
            exact, atol=0, rtol=0,
        )


@pytest.mark.parametrize("rate", [-0.02, 0.0, 0.049, 0.049 + 1e-12, 0.1])
def test_analytical_minimum_matches_independent_scalar_search(rate):
    params = benchmark_params(interest_rate=rate)
    state = np.array([0.03, 0.1])
    controls = [np.array([c, 0.7, q]) for c, q in [(0.35, 0.45), (0.5, 0), (0.01, 0.7)]]
    for control in controls:
        w = float(effective_earnings_share(jnp.asarray(control[1]), jnp.asarray(control[2])))
        w *= np.exp(state[1])
        growth = params.human_capital_productivity * control[2] - params.human_capital_depreciation

        def assets_at(time, w=w, growth=growth, control=control):
            return (np.exp(rate * time) * (state[0] + w * time * exprel((growth-rate)*time))
                    - control[0] * time * exprel(rate*time))

        result = minimize_scalar(assets_at, bounds=(0.0, 2.0), method="bounded",
                                 options={"xatol": 1e-13})
        reference = min(assets_at(0), assets_at(2), float(cast(OptimizeResult, result).fun))
        actual = float(minimum_assets_during_step(jnp.asarray(state), jnp.asarray(control),
                                                  params, 2.0))
        assert actual == pytest.approx(reference, abs=2e-13)


@pytest.mark.parametrize("assets,training", [(0.0005, 0.45), (2.0, 0.1)])
def test_capacity_envelope_gradient_matches_directional_finite_differences(assets, training):
    baseline = benchmark_params()
    point = jnp.array([assets, 0.1, 0.7, training, 0.05, 0.22, 0.05, 0.5, 1e-4])
    direction = jnp.array([0.002, 0.2, 0.1, -0.08, 0.01, 0.02, -0.01, 0.1, 2e-5])

    @jax.jit
    def capacity(x):
        params = baseline._replace(interest_rate=x[4], human_capital_productivity=x[5],
                                   human_capital_depreciation=x[6])
        return maximum_feasible_consumption(x[0], x[1], x[2], x[3], params, x[7], x[8], 4)

    gradient = jax.jit(jax.grad(capacity))(point)
    assert np.all(np.isfinite(gradient))
    derivative = float(gradient @ direction)
    errors = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        finite_difference = float((capacity(point + epsilon * direction)
                                   - capacity(point - epsilon * direction)) / (2 * epsilon))
        errors.append(abs(finite_difference - derivative))
    assert min(errors) < 2e-7
    assert errors[-1] < 2e-6


def test_batched_shape_and_endpoint_api_remain_distinct():
    params = benchmark_params()
    assets = jnp.array([[1e-4], [5e-4], [1.0]])
    h, q = jnp.array([[0.7, 0.8]]), jnp.array([[0.45, 0.1]])
    exact = jax.jit(maximum_feasible_consumption, static_argnums=7)(
        assets, jnp.zeros_like(assets), h, q, params, 0.5, 1e-4, 4
    )
    assert exact.shape == (3, 2)
    endpoint = endpoint_consumption_capacity(assets, jnp.zeros_like(assets), h, q, params,
                                             0.5, 1e-4)
    assert float(endpoint[0, 0]) > float(exact[0, 0]) + 1e-3
    assert float(endpoint[0, 1]) == pytest.approx(float(exact[0, 1]), abs=1e-14)


def test_default_solution_diagnoses_entire_interval():
    config = BellmanConfig(
        periods=2, asset_nodes=5, human_capital_nodes=5, asset_minimum=1e-4,
        asset_maximum=12, log_human_capital_minimum=-1, log_human_capital_maximum=1,
        hours_nodes=3, investment_nodes=3, consumption_nodes=5, refinement_steps=1,
        neighbor_policy_sweeps=0, control_batch_size=16, compute_platform="cpu",
    )
    assert config.asset_feasibility == "continuous"
    solution = solve_bellman(benchmark_params(horizon=1), config)
    diagnostics = diagnose_bellman(solution)
    assert diagnostics.minimum_node_path_assets >= config.asset_minimum - 1e-12
    assert diagnostics.simulation_minimum_assets >= config.asset_minimum - 1e-12
    assert diagnostics.accepted_node_solution
    assert "minimum_node_path_assets" in diagnostics.as_dict()
    with pytest.raises(ValueError, match="asset_feasibility"):
        solve_bellman(solution.params, replace(config, asset_feasibility="unknown"))
