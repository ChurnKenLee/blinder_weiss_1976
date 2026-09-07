from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, solve_bellman
from blinder_weiss import distribution as distribution_module
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
