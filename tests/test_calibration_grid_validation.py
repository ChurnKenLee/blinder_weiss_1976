"""Sampling and aggregation must not manufacture grid-convergence evidence."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, solve_bellman
from blinder_weiss.bellman import constant_control_transition
from blinder_weiss.model import bequest_utility, flow_utility

_SCRIPT = Path(__file__).resolve().parents[1] / "tools/validate_calibration_grid.py"
_SPEC = importlib.util.spec_from_file_location("calibration_grid_validation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


def _trajectory(params):
    time = np.asarray([0.0, 0.7, 2.0])
    controls = np.asarray(
        [[[0.2, 0.3, 0.1], [0.3, 0.5, 0.2]], [[0.25, 0.2, 0.05], [0.1, 0.4, 0.1]]]
    )
    states = [np.asarray([[2.0, np.log(0.8)], [8.0, np.log(1.2)]])]
    for index, duration in enumerate(np.diff(time)):
        states.append(
            np.asarray(
                constant_control_transition(
                    jnp.asarray(states[-1]), jnp.asarray(controls[index]), params, float(duration)
                )
            )
        )
    return time, np.asarray(states), controls


def test_fixed_cohort_matches_calibration_random_draws() -> None:
    first = validation.fixed_cohort()
    second = validation.fixed_cohort()
    random = np.random.default_rng(125)
    expected_assets = random.uniform(2.0, 8.0, 256)
    expected_capital = np.exp(random.uniform(-0.25, 0.25, 256))
    expected_weights = random.uniform(0.5, 1.5, 256)
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(first[0], expected_assets)
    np.testing.assert_array_equal(first[1], expected_capital)
    np.testing.assert_allclose(first[2], expected_weights / expected_weights.sum())


@pytest.mark.parametrize("interest_rate", [0.0, 0.05])
def test_state_sampling_is_exact_and_controls_remain_steps(interest_rate) -> None:
    params = benchmark_params(horizon=2.0, interest_rate=interest_rate)
    time, states, controls = _trajectory(params)
    sample_times = np.asarray([0.0, 0.2, 0.7, 1.3, 2.0])
    sampled_states, sampled_controls = validation.sample_trajectory(
        time, states, controls, params, sample_times
    )
    for index, age in enumerate(sample_times):
        period = 0 if age < 0.7 else 1
        expected = constant_control_transition(
            jnp.asarray(states[period]),
            jnp.asarray(controls[period]),
            params,
            float(age - time[period]),
        )
        np.testing.assert_allclose(sampled_states[index], expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(sampled_controls[index], controls[period])
    np.testing.assert_allclose(sampled_states[-1], states[-1], rtol=1e-13)


@pytest.mark.parametrize("rho", [0.0, 0.03])
def test_lifetime_utility_matches_discounted_constant_control_integrals(rho) -> None:
    params = benchmark_params(horizon=2.0, rho=rho)
    time, states, controls = _trajectory(params)
    expected = np.zeros(2)
    for index, duration in enumerate(np.diff(time)):
        factor = duration if rho == 0.0 else -np.expm1(-rho * duration) / rho
        flow = np.asarray(
            flow_utility(
                jnp.asarray(controls[index, :, 0]), jnp.asarray(controls[index, :, 1]), params
            )
        )
        expected += np.exp(-rho * time[index]) * factor * flow
    expected += np.exp(-rho * time[-1]) * np.asarray(
        bequest_utility(jnp.asarray(states[-1, :, 0]), params)
    )
    np.testing.assert_allclose(
        validation.lifetime_utilities(time, states, controls, params), expected, rtol=1e-13
    )


def test_weighted_errors_retain_signed_and_worst_case_differences() -> None:
    actual = np.asarray([[0.0, 4.0], [-2.0, 0.0]])
    result = validation.error_metrics(actual, np.zeros_like(actual), np.asarray([1.0, 3.0]))
    assert result["rmse"] == pytest.approx(np.sqrt(6.5))
    assert result["mean_absolute"] == pytest.approx(1.75)
    assert result["mean_signed"] == pytest.approx(1.25)
    assert result["maximum_absolute"] == 4.0
    assert result["maximum_index"] == [0, 1]


def test_reference_states_use_independent_integration(tmp_path) -> None:
    params = benchmark_params(horizon=2.0, initial_assets=2.0, initial_human_capital=0.8)
    time = np.asarray([0.0, 0.5, 1.0, 2.0])
    control = jnp.asarray([0.3, 0.4, 0.1])
    initial = jnp.asarray([2.0, np.log(0.8)])
    states = np.asarray(
        [constant_control_transition(initial, control, params, float(age)) for age in time]
    )
    metadata = {
        "params": params._asdict(),
        "reference_accepted": True,
        "diagnostics": {"accepted_success": True},
        "lifetime_utility": -10.0,
    }
    (tmp_path / "reference.json").write_text(json.dumps(metadata))
    np.savez(
        tmp_path / "reference.npz",
        time=time,
        A=states[:, 0],
        K=np.exp(states[:, 1]),
        c=np.full(4, 0.3),
        h=np.full(4, 0.4),
        q=np.full(4, 0.1),
    )
    comparison = validation._reference_comparison(
        tmp_path,
        "reference",
        SimpleNamespace(params=params),
        time,
        states,
        np.tile(control, (3, 1)),
        -10.1,
    )
    assert comparison["status"] == "compared"
    assert comparison["path_errors"]["A"]["maximum_absolute"] < 1e-8
    assert comparison["path_errors"]["K"]["maximum_absolute"] < 1e-8
    assert comparison["reference_minus_realized_utility"] == pytest.approx(0.1)
    metadata["reference_accepted"] = False
    (tmp_path / "reference.json").write_text(json.dumps(metadata))
    rejected = validation._reference_comparison(
        tmp_path,
        "reference",
        SimpleNamespace(params=params),
        time,
        states,
        np.tile(control, (3, 1)),
        -10.1,
    )
    assert rejected["status"] == "reference_not_accepted"
    assert "path_errors" not in rejected


def test_saved_grid_smoke_comparison_reports_zero_for_identical_grids(tmp_path) -> None:
    solution = solve_bellman(
        benchmark_params(horizon=1.0),
        BellmanConfig(
            periods=2,
            asset_nodes=3,
            human_capital_nodes=3,
            asset_minimum=1e-3,
            asset_maximum=12.0,
            log_human_capital_minimum=-1.0,
            log_human_capital_maximum=1.0,
            hours_nodes=3,
            investment_nodes=3,
            consumption_nodes=3,
            refinement_steps=0,
            neighbor_policy_sweeps=0,
            compute_platform="cpu",
        ),
    )
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "report.json").write_text(
        json.dumps(
            {
                "params": solution.params._asdict(),
                "config": asdict(solution.config),
                "results": [{"solve_seconds": solution.solve_seconds}],
            }
        )
    )
    np.savez(
        folder / "policies.npz",
        time=solution.time,
        A=solution.asset_grid,
        y=solution.log_human_capital_grid,
        values=solution.values,
        c=solution.consumption_policy,
        h=solution.hours_policy,
        q=solution.training_time_policy,
    )
    output = tmp_path / "comparison.json"
    result = validation.validate_folders([folder, folder], tmp_path, output, "cpu", 0.02)
    assert [run["status"] for run in result["runs"]] == ["evaluated", "evaluated"]
    for metrics in result["successive_grid_comparisons"][0]["cohort_path_errors"].values():
        assert metrics["maximum_absolute"] == 0.0
    assert output.exists() and output.with_suffix(".npz").exists()
    assert "converged" not in result
    audit = validation.audit_saved_recovery(
        output,
        tmp_path / "recovery_audit.json",
        "cpu",
        top_types=1,
        states_per_type=1,
        refinement_starts=2,
        refinement_steps=4,
    )
    assert audit["maximum_saved_policy_value_recomputation_error"] < 1e-10
    assert audit["maximum_telescoping_error"] < 1e-10
    assert "converged" not in audit
    assert len(audit["types"]) >= 1
    for person in audit["types"]:
        assert len(person["worst_defect_cell_corners"]) == 4
        for state in person["states"]:
            assert state["observed_policy_regret_lower_bound"] == max(
                0.0, state["stronger_minus_saved_objective"]
            )
    assert (
        result["runs"][0]["direct_references"]["direct_reference"]["status"] == "missing_reference"
    )


def test_recovery_gap_decomposition_retains_sign_and_detects_stale_values() -> None:
    params = benchmark_params(horizon=2.0)
    time, states, controls = _trajectory(params)
    solution = SimpleNamespace(
        params=params,
        config=BellmanConfig(periods=2, asset_nodes=3, human_capital_nodes=3),
        time=time,
        asset_grid=np.asarray([0.0, 5.0, 10.0]),
        log_human_capital_grid=np.asarray([-1.0, 0.0, 1.0]),
        values=np.asarray(
            [
                [[-20.0, -19.0, -18.0], [-12.0, -11.0, -10.0], [-8.0, -7.0, -6.0]],
                [[-15.0, -14.0, -13.0], [-10.0, -9.0, -8.0], [-7.0, -6.0, -5.0]],
                [[-1.0, -1.0, -1.0], [-0.2, -0.2, -0.2], [-0.1, -0.1, -0.1]],
            ]
        ),
    )
    evaluated = validation.recovery_gap_decomposition(
        solution, time, states, controls, np.zeros((2, 2))
    )
    policy_values = evaluated["reevaluated_policy_values"]
    expected_gap = policy_values[0] - validation.lifetime_utilities(time, states, controls, params)
    np.testing.assert_allclose(evaluated["initial_value_gap"], expected_gap, atol=1e-13)
    np.testing.assert_allclose(evaluated["telescoping_error"], 0.0, atol=1e-13)
    signed_defects = (evaluated["represented_values"][1] - policy_values[1]) * np.exp(
        -params.rho * time[1]
    )
    np.testing.assert_allclose(expected_gap, signed_defects, atol=1e-13)
    stale = policy_values.copy()
    stale[1, 0] += 0.25
    checked = validation.recovery_gap_decomposition(solution, time, states, controls, stale)
    assert checked["saved_policy_value_error"][1, 0] == pytest.approx(0.25)
    np.testing.assert_allclose(checked["initial_value_gap"], expected_gap, atol=1e-13)
