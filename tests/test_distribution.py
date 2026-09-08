from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, solve_bellman
from blinder_weiss import distribution as distribution_module
from blinder_weiss.bellman import _exprel, constant_control_transition, maximum_feasible_consumption
from blinder_weiss.boundary import endpoint_floor_contact
from blinder_weiss.distribution import (
    DistributionDomainError,
    DistributionGrid,
    build_transport,
    initialize_distribution,
    simulate_distribution,
)
from blinder_weiss.population import cohort_moments, simulate_cohort


@pytest.fixture
def grid():
    return DistributionGrid(0.01, np.array([0.02, 0.2, 0.7, 1.5]), np.array([-1.0, 0.0, 1.0]))


def test_identity_nonuniform_grid_and_adjoint(grid):
    transport = build_transport(grid, grid.states)
    rng = np.random.default_rng(21)
    mass = rng.uniform(size=len(grid.states))
    mass /= mass.sum()
    values = rng.normal(size=mass.size)
    np.testing.assert_allclose(transport.weights.sum(axis=1), 1, atol=1e-15)
    assert np.min(transport.weights) >= 0
    np.testing.assert_allclose(transport.push(mass), mass, atol=1e-15)
    np.testing.assert_allclose(transport.pull(values), values, atol=1e-15)

    destinations = grid.states.copy()
    destinations[:, 0] = 0.3
    destinations[:, 1] *= 0.75
    translated = build_transport(grid, destinations)
    np.testing.assert_allclose(
        translated.push(mass) @ values, mass @ translated.pull(values), atol=1e-15
    )
    np.testing.assert_allclose(translated.push(mass).sum(), 1.0, atol=1e-15)


def test_boundary_enter_stay_along_leave_and_corner(grid):
    # Four sources: arrival from interior, staying corner, moving along face,
    # leaving the face. Destinations classify constraint contact independently
    # of ordinary interpolation onto the first interior cell.
    destinations = np.array([[0.01, 0.5], [0.01, -1], [0.01, 0.25], [0.11, 0]])
    mass = np.array([0.2, 0.3, 0.1, 0.4])
    result = build_transport(grid, destinations).push(mass).reshape(grid.shape)
    np.testing.assert_allclose(result[0].sum(), 0.6, atol=1e-15)
    np.testing.assert_allclose(result[0], [0.3, 0.175, 0.125], atol=1e-15)
    np.testing.assert_allclose(result[1:].sum(), 0.4, atol=1e-15)
    np.testing.assert_allclose(result.sum(), 1, atol=1e-15)


def test_first_cell_interior_does_not_manufacture_floor_atom(grid):
    near_floor = np.nextafter(grid.asset_floor, np.inf)
    destinations = np.array([[near_floor, 0.0], [grid.asset_floor, 0.0]])
    stencil = build_transport(grid, destinations)
    result = stencil.push([0.7, 0.3]).reshape(grid.shape)
    np.testing.assert_allclose(result[0].sum(), 0.3, atol=1e-15)
    np.testing.assert_allclose(result[1].sum(), 0.7, atol=1e-15)
    np.testing.assert_array_equal(stencil.lower_interior_remap, [True, False])
    np.testing.assert_allclose(stencil.asset_remap_error, [0.02 - near_floor, 0.0])
    initial = initialize_distribution(grid, destinations[:, 0], np.ones(2), [0.7, 0.3])
    np.testing.assert_allclose(initial, result)


def test_explicit_floor_contact_corrects_roundoff_only(grid):
    states = np.array([[np.nextafter(grid.asset_floor, np.inf), 0.0]])
    transport = build_transport(grid, states, on_asset_floor=[True])
    result = transport.push([1.0]).reshape(grid.shape)
    assert result[0, 1] == 1.0
    with pytest.raises(DistributionDomainError, match="invalid floor tag"):
        build_transport(grid, [[0.1, 0.0]], on_asset_floor=[True])


@pytest.mark.parametrize("destination,face", [([0.0, 0], 0), ([2, 0], 1), ([0.2, -2], 2),
                                              ([0.2, 2], 3)])
def test_material_domain_exits_raise_before_remapping(grid, destination, face):
    with pytest.raises(DistributionDomainError, match="left the distribution domain") as error:
        build_transport(grid, [destination])
    assert error.value.diagnostics["attempted_exit_distances"][0, face] > 1e-10


def test_translation_mean_and_diffusion_under_refinement():
    variances = []
    for nodes in (51, 101, 201):
        grid = DistributionGrid(0.0, np.linspace(0.01, 2.0, nodes), np.array([-1.0, 1.0]))
        mass = initialize_distribution(grid, 0.5, 1.0).ravel()
        destination = grid.states.copy()
        # Translation is exact on the reachable support. Unreachable rightmost
        # rows stay fixed, avoiding an artificial exit unrelated to this test.
        destination[:, 0] += np.where(destination[:, 0] < 1.8, 0.013, 0.0)
        operator = build_transport(grid, destination)
        for _ in range(10):
            mass = operator.push(mass)
        mean = mass @ grid.states[:, 0]
        variance = mass @ (grid.states[:, 0] - mean) ** 2
        variances.append(variance)
        np.testing.assert_allclose(mean, 0.63, atol=1e-14)
        np.testing.assert_allclose(mass.sum(), 1, atol=1e-14)
        assert np.min(mass) >= 0
    assert variances[2] < variances[1] < variances[0]


@pytest.fixture(scope="module")
def solution():
    return solve_bellman(
        benchmark_params(horizon=0.5),
        BellmanConfig(
            periods=2, asset_nodes=5, human_capital_nodes=5,
            asset_minimum=0.001, asset_maximum=12.0,
            log_human_capital_minimum=-1.0, log_human_capital_maximum=1.0,
            hours_nodes=3, investment_nodes=3, consumption_nodes=5,
            refinement_steps=2, control_batch_size=16, neighbor_policy_sweeps=0,
            compute_platform="cpu",
        ),
    )


def test_full_lifecycle_conservation_and_cohort_first_period(solution):
    grid = DistributionGrid(
        solution.config.asset_minimum, np.array([0.001001, 0.5, 2.0, 5.0, 8.0, 12.0]),
        np.linspace(-1, 1, 7),
    )
    mass = initialize_distribution(grid, [5.0, 8.0], 1.0, [0.4, 0.6])
    result = simulate_distribution(
        solution, grid, mass, participation_hours_threshold=0.01, store_snapshots=True
    )
    assert result.masses is not None
    assert result.masses.shape == (3, *grid.shape)
    np.testing.assert_allclose(result.masses.sum(axis=(1, 2)), 1, atol=1e-14)
    assert result.diagnostics["maximum_mass_drift"] < 1e-12
    assert result.diagnostics["maximum_row_sum_error"] < 1e-12
    assert result.diagnostics["minimum_probability_mass"] >= 0
    assert result.diagnostics["completed_periods"] == 2
    assert result.diagnostics["outer_cell_mass"].shape == (3, 3)
    np.testing.assert_allclose(result.diagnostics["outer_cell_mass"][0], [0.6, 0, 0])
    assert not result.diagnostics["accepted_for_calibration_default"]
    np.testing.assert_array_equal(result.state_moments.time, solution.time)
    cohort = simulate_cohort(solution, [5.0, 8.0], 1.0, weights=[0.4, 0.6])
    reference = cohort_moments(cohort, participation_hours_threshold=0.01)
    for name in ("hours", "participation", "training_time", "consumption", "earnings"):
        np.testing.assert_allclose(getattr(result.moments, name)[0], getattr(reference, name)[0],
                                   atol=1e-11)
    states = grid.states
    np.testing.assert_allclose(
        result.state_moments.assets, result.masses.reshape(3, -1) @ states[:, 0]
    )
    no_snapshots = simulate_distribution(
        solution, grid, mass, participation_hours_threshold=0.01, store_snapshots=False
    )
    assert no_snapshots.masses is None
    np.testing.assert_allclose(no_snapshots.terminal_mass, result.terminal_mass, atol=1e-15)


def test_scan_reports_occupied_artificial_exit_and_ignores_inactive_nan(solution, monkeypatch):
    grid = DistributionGrid(0.001, np.array([0.1, 1.0, 2.0]), np.array([-0.5, 0.0, 0.5]))
    params = solution.params._replace(
        interest_rate=0.0, human_capital_productivity=0.0, human_capital_depreciation=0.0
    )
    artificial_solution = replace(solution, params=params)

    def recover(_params, states, _continuation, _policy, _terminal):
        # Only the occupied node has a finite policy. Its earnings .5 and
        # consumption .5 imply identity; invalid inactive rows must not pollute
        # moments or the conservative scatter through zero-times-NaN.
        active = (states[:, 0] == 1.0) & (states[:, 1] == 0.0)
        finite_or_nan = jnp.where(active, 0.5, jnp.nan)
        return jnp.where(active, 0.0, jnp.nan), finite_or_nan, finite_or_nan, jnp.zeros_like(active)

    monkeypatch.setattr(distribution_module, "_cached_greedy_kernels",
                        lambda *args: (recover, None, jax.devices("cpu")[0]))
    distribution_module._cached_distribution_scan.cache_clear()
    mass = initialize_distribution(grid, 1.0, 1.0)
    result = simulate_distribution(
        artificial_solution, grid, mass, participation_hours_threshold=0.01
    )
    np.testing.assert_allclose(result.terminal_mass, mass)
    np.testing.assert_allclose(result.moments.consumption, 0.5)

    def exit_recover(_params, states, _continuation, _policy, _terminal):
        size = states.shape[0]
        return jnp.zeros(size), jnp.full(size, 0.1), jnp.full(size, 0.8), jnp.zeros(size)

    monkeypatch.setattr(distribution_module, "_cached_greedy_kernels",
                        lambda *args: (exit_recover, None, jax.devices("cpu")[0]))
    distribution_module._cached_distribution_scan.cache_clear()
    mass = initialize_distribution(grid, 2.0, 1.0)
    with pytest.raises(DistributionDomainError, match="positive population mass left") as error:
        simulate_distribution(artificial_solution, grid, mass, participation_hours_threshold=0.01)
    assert error.value.diagnostics["material_exit_mass"][0, 1] == 1
    assert error.value.diagnostics["maximum_exit_distance"][0, 1] > 0.1
    assert error.value.diagnostics["terminal_total_mass"] == 1
    assert error.value.diagnostics["completed_periods"] == 0
    distribution_module._cached_distribution_scan.cache_clear()


@pytest.mark.parametrize("mass", [[-0.1, 1.1], [0.2, 0.2], [np.nan, 1.0]])
def test_initial_weights_are_probabilities_without_hidden_normalization(grid, mass):
    with pytest.raises(ValueError, match="probability masses"):
        initialize_distribution(grid, [0.1, 0.2], [1.0, 1.0], mass)


def test_compiled_scan_enters_moves_along_and_leaves_floor(solution, monkeypatch):
    grid = DistributionGrid(
        solution.config.asset_minimum, np.array([0.001001, 0.1, 1.0, 3.0]),
        np.array([-1.0, 0.0, 0.5, 1.0]),
    )
    params = solution.params._replace(
        interest_rate=0.0, human_capital_productivity=0.0, human_capital_depreciation=0.2
    )
    artificial_solution = replace(solution, params=params)

    def recover(current_params, states, _continuation, _policy, terminal):
        hours = jnp.full(states.shape[0], 0.5)
        training = jnp.zeros_like(hours)
        capacity = distribution_module.maximum_feasible_consumption(
            states[:, 0], states[:, 1], hours, training, current_params,
            current_params.horizon / solution.config.periods, grid.asset_floor, 1,
        )
        consumption = jnp.where(terminal, 0.1, capacity)
        return jnp.zeros_like(hours), consumption, hours, training

    monkeypatch.setattr(distribution_module, "_cached_greedy_kernels",
                        lambda *args: (recover, None, jax.devices("cpu")[0]))
    distribution_module._cached_distribution_scan.cache_clear()
    initial = initialize_distribution(grid, 1.0, np.exp(0.5))
    result = simulate_distribution(
        artificial_solution, grid, initial, participation_hours_threshold=0.01,
        store_snapshots=True,
    )
    np.testing.assert_allclose(result.state_moments.asset_floor_mass, [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(result.diagnostics["entered_asset_floor_mass"], [1, 0], atol=1e-14)
    np.testing.assert_allclose(result.diagnostics["left_asset_floor_mass"], [0, 1], atol=1e-14)
    np.testing.assert_allclose(result.state_moments.log_human_capital, [0.5, 0.45, 0.4],
                               atol=1e-14)
    assert result.masses is not None
    np.testing.assert_allclose(result.masses.sum(axis=(1, 2)), 1, atol=1e-14)
    distribution_module._cached_distribution_scan.cache_clear()



def test_actual_binding_checkpoint_policy_uses_asset_roundoff_scale():
    params = benchmark_params()
    floors = jnp.geomspace(1e-4, 1e-2, 257)

    @jax.jit
    def contacts(current_params, current_floors):
        def one(floor):
            state = jnp.array([[floor, 0.0]])
            hours = jnp.zeros(1)
            capacity = maximum_feasible_consumption(
                state[:, 0], state[:, 1], hours, hours, current_params, 0.5, floor, 4
            )
            endpoint = maximum_feasible_consumption(
                state[:, 0], state[:, 1], hours, hours, current_params, 0.5, floor, 1
            )
            next_assets = constant_control_transition(
                state, jnp.stack((capacity, hours, hours), axis=-1), current_params, 0.5
            )[:, 0]
            old_contact = (next_assets <= floor) | (capacity >= endpoint)
            contact = endpoint_floor_contact(
                next_assets, floor, capacity, endpoint,
                0.5 * _exprel(current_params.interest_rate * 0.5),
            )
            return old_contact[0], contact[0]
        return jax.vmap(one)(current_floors)

    old, fixed = map(np.asarray, contacts(params, floors))
    # These are actual feasible controls maintaining A=floor: c=r*floor,
    # h=q=0. The old bit-for-bit comparison loses contacts solely to rounding.
    assert np.any(~old)
    np.testing.assert_array_equal(fixed, np.ones(floors.size, dtype=bool))


def test_slack_control_near_floor_is_still_interior():
    params = benchmark_params()
    floor = 1e-4
    state = jnp.array([[floor, 0.0]])
    zero = jnp.zeros(1)
    endpoint = maximum_feasible_consumption(
        state[:, 0], state[:, 1], zero, zero, params, 0.5, floor, 1
    )
    # Meaningful slack is tiny in economic units but many arithmetic ulps.
    consumption = endpoint - 1e-12
    destination = constant_control_transition(
        state, jnp.stack((consumption, zero, zero), axis=-1), params, 0.5
    )
    assert float(destination[0, 0]) > floor
    assert not bool(endpoint_floor_contact(
        destination[:, 0], floor, consumption, endpoint,
        0.5 * _exprel(params.interest_rate * 0.5),
    )[0])


def test_repeated_floor_policy_has_no_face_interior_alternation(solution, monkeypatch):
    from blinder_weiss.continuum import _cached_floor_contacts

    periods = 140
    params = solution.params._replace(horizon=70.0)
    config = replace(solution.config, periods=periods)
    policy_shape = (periods, *solution.consumption_policy.shape[1:])
    artificial = replace(
        solution, params=params, config=config, time=np.linspace(0, 70, periods + 1),
        values=np.zeros((periods + 1, *solution.values.shape[1:])),
        consumption_policy=np.zeros(policy_shape), hours_policy=np.zeros(policy_shape),
        training_time_policy=np.zeros(policy_shape),
    )
    grid = DistributionGrid(config.asset_minimum, np.array([0.001001, 0.5, 2.0, 5.0]),
                            np.linspace(-5, 1, 25))
    # The baseline Bellman box is narrow only in this small fixture; the
    # supplied policy is known analytically and the independent test domain
    # covers 70 years of human-capital depreciation.
    artificial = replace(artificial, log_human_capital_grid=np.linspace(-5, 1, 5))

    def recover(current_params, states, _continuation, _policy, _terminal):
        zero = jnp.zeros(states.shape[0])
        capacity = maximum_feasible_consumption(
            states[:, 0], states[:, 1], zero, zero, current_params, 0.5,
            config.asset_minimum, config.path_checkpoints,
        )
        return zero, capacity, zero, zero

    monkeypatch.setattr(distribution_module, "_cached_greedy_kernels",
                        lambda *args: (recover, None, jax.devices("cpu")[0]))
    distribution_module._cached_distribution_scan.cache_clear()
    mass = initialize_distribution(grid, config.asset_minimum, 1.0)
    result = simulate_distribution(
        artificial, grid, mass, participation_hours_threshold=0.01, store_snapshots=True
    )
    np.testing.assert_allclose(result.state_moments.asset_floor_mass, 1, atol=1e-13)
    np.testing.assert_allclose(result.diagnostics["stayed_asset_floor_mass"], 1, atol=1e-13)
    np.testing.assert_array_equal(result.diagnostics["left_asset_floor_mass"], 0)
    np.testing.assert_array_equal(result.diagnostics["lower_interior_remap_mass"], 0)

    @jax.jit
    def exact_path(current_params):
        def step(state, _):
            _, c, h, q = recover(current_params, state, None, None, None)
            controls = jnp.stack((c, h, q), axis=-1)
            destination = constant_control_transition(state, controls, current_params, 0.5)
            return destination, (state, controls)
        terminal, (states, controls) = jax.lax.scan(
            step, jnp.array([[config.asset_minimum, 0.0]]), None, length=periods
        )
        return jnp.concatenate((states, terminal[None, ...])), controls

    states, controls = exact_path(params)
    contact = np.asarray(_cached_floor_contacts(config)(params, states, controls))
    np.testing.assert_array_equal(contact, np.ones((periods + 1, 1), dtype=bool))
    distribution_module._cached_distribution_scan.cache_clear()
