from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, simulate_policy, solve_bellman
from blinder_weiss import bellman, population
from blinder_weiss.bellman import constant_control_transition
from blinder_weiss.model import effective_earnings_share
from blinder_weiss.population import cohort_moments, simulate_cohort


@pytest.fixture(scope="module")
def cohort_solution():
    return solve_bellman(
        benchmark_params(horizon=3.0),
        BellmanConfig(
            periods=3,
            asset_nodes=5,
            human_capital_nodes=5,
            asset_minimum=1e-3,
            asset_maximum=12.0,
            log_human_capital_minimum=-1.0,
            log_human_capital_maximum=1.0,
            hours_nodes=3,
            investment_nodes=3,
            consumption_nodes=5,
            refinement_steps=2,
            control_batch_size=16,
            neighbor_policy_sweeps=0,
            compute_platform="cpu",
        ),
    )


def test_cohort_matches_independent_greedy_lifecycles(cohort_solution) -> None:
    initial_assets = np.array([0.5, 5.0, 8.0])
    initial_human_capital = np.array([0.8, 1.0, 1.25])
    cohort = simulate_cohort(cohort_solution, initial_assets, initial_human_capital)
    assert cohort.states.shape == (4, 3, 2)
    assert cohort.controls.shape == (3, 3, 3)
    assert cohort.policy_values.shape == (3, 3)
    np.testing.assert_allclose(cohort.weights, np.full(3, 1.0 / 3.0))
    for person in range(3):
        separate = simulate_policy(
            cohort_solution,
            initial_assets=initial_assets[person],
            initial_human_capital=initial_human_capital[person],
        )
        np.testing.assert_allclose(cohort.assets[:, person], separate.assets, atol=1e-10)
        np.testing.assert_allclose(
            cohort.human_capital[:, person], separate.human_capital, atol=1e-10
        )
        np.testing.assert_allclose(cohort.consumption[:, person], separate.consumption, atol=1e-10)
        np.testing.assert_allclose(cohort.hours[:, person], separate.hours, atol=1e-10)
        np.testing.assert_allclose(
            cohort.training_time[:, person], separate.training_time, atol=1e-10
        )

    step = cohort_solution.params.horizon / cohort_solution.config.periods
    for duration in np.linspace(step / 12, step, 12):
        checkpoint_states = np.asarray(
            constant_control_transition(
                cohort.states[:-1], cohort.controls, cohort_solution.params, float(duration)
            )
        )
        assert np.min(checkpoint_states[..., 0]) >= cohort_solution.config.asset_minimum - 1e-8
    assert cohort.minimum_consumption_capacity_slack >= -1e-8


def test_cohort_broadcasts_scalar_states_and_normalizes_large_weights(cohort_solution) -> None:
    cohort = simulate_cohort(cohort_solution, [2.0, 5.0], 1.0, weights=[1e308, 1e308])
    np.testing.assert_allclose(cohort.human_capital[0], [1.0, 1.0])
    np.testing.assert_allclose(cohort.weights, [0.5, 0.5])
    single = simulate_cohort(cohort_solution, 5.0, 1.0)
    assert single.states.shape == (4, 1, 2)
    np.testing.assert_array_equal(single.weights, [1.0])


def test_fixed_cohort_size_reuses_rollout_for_changed_parameters(cohort_solution, monkeypatch):
    changed_solution = solve_bellman(
        cohort_solution.params._replace(leisure_weight=2.5), cohort_solution.config
    )
    original_optimizer = bellman._make_control_optimizer
    traces = []

    def count_optimizer_traces(*args, **kwargs):
        traces.append(True)
        return original_optimizer(*args, **kwargs)

    bellman._cached_greedy_kernels.cache_clear()
    monkeypatch.setattr(bellman, "_make_control_optimizer", count_optimizer_traces)
    initial = simulate_cohort(cohort_solution, [0.5, 5.0], 1.0)
    first_count = len(traces)
    assert first_count > 0
    changed = simulate_cohort(changed_solution, [0.75, 5.5], 1.0, weights=[1, 4])
    assert len(traces) == first_count
    assert not np.allclose(initial.controls, changed.controls)
    bellman._cached_greedy_kernels.cache_clear()
    fresh = simulate_cohort(changed_solution, [0.75, 5.5], 1.0, weights=[1, 4])
    assert len(traces) > first_count
    np.testing.assert_allclose(changed.states, fresh.states, atol=1e-10)
    np.testing.assert_allclose(changed.controls, fresh.controls, atol=1e-10)
    np.testing.assert_allclose(changed.policy_values, fresh.policy_values, atol=1e-10)


@pytest.mark.parametrize(
    ("assets", "human_capital", "weights", "message"),
    [
        ([], [], None, "nonempty"),
        ([[1.0]], [[1.0]], None, "one-dimensional"),
        ([np.nan], [1.0], None, "initial_assets must be finite"),
        ([np.inf], [1.0], None, "initial_assets must be finite"),
        ([1.0], [np.nan], None, "finite and positive"),
        ([1.0], [np.inf], None, "finite and positive"),
        ([1.0], [0.0], None, "finite and positive"),
        ([1.0], [-1.0], None, "finite and positive"),
        ([0.0], [1.0], None, "inside the Bellman domain"),
        ([13.0], [1.0], None, "inside the Bellman domain"),
        ([1.0], [3.0], None, "inside the Bellman domain"),
        ([1.0], [1.0], [1.0, 2.0], "one entry per person"),
        ([1.0], [1.0], [np.nan], "finite and nonnegative"),
        ([1.0], [1.0], [np.inf], "finite and nonnegative"),
        ([1.0], [1.0], [-1.0], "finite and nonnegative"),
        ([1.0], [1.0], [0.0], "positive total mass"),
    ],
)
def test_invalid_cohort_inputs_fail_before_rollout(
    cohort_solution, monkeypatch, assets, human_capital, weights, message
) -> None:
    def unexpected_kernel(*args):
        pytest.fail("invalid inputs should fail before requesting a rollout kernel")

    monkeypatch.setattr(population, "_cached_greedy_kernels", unexpected_kernel)
    with pytest.raises(ValueError, match=message):
        simulate_cohort(cohort_solution, assets, human_capital, weights=weights)


def test_weighted_profiles_and_period_start_earnings(cohort_solution) -> None:
    cohort = simulate_cohort(cohort_solution, [0.5, 5.0, 8.0], [0.8, 1.0, 1.25], weights=[1, 3, 0])
    threshold = float(cohort.hours[0, 0])
    moments = cohort_moments(cohort, participation_hours_threshold=threshold)
    np.testing.assert_allclose(cohort.weights, [0.25, 0.75, 0.0])
    for name in ("hours", "training_time", "consumption", "earnings"):
        values = getattr(cohort, name)
        expected = 0.25 * values[:, 0] + 0.75 * values[:, 1]
        np.testing.assert_allclose(getattr(moments, name), expected)
    expected_participation = (
        0.25 * (cohort.hours[:, 0] > threshold) + 0.75 * (cohort.hours[:, 1] > threshold)
    )
    np.testing.assert_allclose(moments.participation, expected_participation)
    assert moments.participation[0] in (0.0, 0.75)
    expected_earnings = (
        np.asarray(effective_earnings_share(cohort.hours, cohort.training_time))
        * cohort.human_capital[:-1]
    )
    np.testing.assert_allclose(cohort.earnings, expected_earnings)
    np.testing.assert_array_equal(moments.time, cohort_solution.time[:-1])
    assert moments.participation_hours_threshold == threshold


def test_training_reduces_measured_earnings_and_retirement_is_finite(cohort_solution) -> None:
    base = simulate_cohort(cohort_solution, [5.0, 5.0, 5.0], 1.0)
    # Verify the JAX measurement kernel independently at known economic endpoints.
    states = np.tile(np.array([[5.0, np.log(2.0)]] * 3), (4, 1, 1))
    controls = np.tile(np.array([[0.2, 0.0, 0.0], [0.2, 0.5, 0.0], [0.2, 0.5, 0.5]]), (3, 1, 1))
    earnings, *_ = population._cached_cohort_checks(cohort_solution.config)(
        cohort_solution.params, states, controls, np.zeros((3, 3))
    )
    np.testing.assert_allclose(np.asarray(earnings), [[0.0, 1.0, 0.0]] * 3, atol=1e-14)
    artificial = replace(base, controls=controls, earnings=np.asarray(earnings))
    moments = cohort_moments(artificial, participation_hours_threshold=0.0)
    np.testing.assert_allclose(moments.participation, np.full(3, 2 / 3))
    np.testing.assert_allclose(moments.training_time, np.full(3, 1 / 6))
    np.testing.assert_allclose(moments.earnings, np.full(3, 1 / 3))


@pytest.mark.parametrize("threshold", [-0.1, 1.0, np.nan, np.inf])
def test_invalid_participation_threshold(cohort_solution, threshold) -> None:
    cohort = simulate_cohort(cohort_solution, 5.0, 1.0)
    with pytest.raises(ValueError, match="participation_hours_threshold"):
        cohort_moments(cohort, participation_hours_threshold=threshold)


@pytest.mark.parametrize(
    ("failure", "message"),
    [("nonfinite", "nonfinite"), ("domain", "left the Bellman domain"), ("control", "constraint")],
)
def test_bad_rollouts_cannot_silently_produce_calibration_moments(
    cohort_solution, monkeypatch, failure, message
) -> None:
    cohort = simulate_cohort(cohort_solution, 5.0, 1.0)
    states = cohort.states.copy()
    controls = cohort.controls.copy()
    values = cohort.policy_values.copy()
    if failure == "nonfinite":
        values[0, 0] = np.nan
    elif failure == "domain":
        states[-1, 0, 0] = cohort_solution.config.asset_maximum + 1.0
    else:
        controls[0, 0, 2] = controls[0, 0, 1] + 0.1

    def fake_rollout(*args):
        return states, controls, values

    monkeypatch.setattr(
        population,
        "_cached_greedy_kernels",
        lambda *args: (None, fake_rollout, jax.devices("cpu")[0]),
    )
    with pytest.raises(RuntimeError, match=message):
        simulate_cohort(cohort_solution, 5.0, 1.0)
