"""Independent references for the validation tool, not the production bound."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from blinder_weiss.model import benchmark_params
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import exprel

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
_SPEC = importlib.util.spec_from_file_location(
    "boundary_validation", _TOOLS / "validate_boundary_refinement.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    "rate,training", [(0.05, 0.4), (0.0, 0.4), (-0.03, 0.4), (0.038, 0.4), (0.0, 0.05 / 0.22)]
)
def test_analytic_minima_match_independent_adaptive_ode_and_scalar_search(rate, training):
    params = benchmark_params(interest_rate=rate)
    generator = np.random.default_rng(715)
    states = np.column_stack((generator.uniform(0.02, 2.0, 10), generator.uniform(-0.4, 0.4, 10)))
    controls = np.column_stack(
        (generator.uniform(0.1, 1.5, 10), np.full(10, 0.7), np.full(10, training))
    )
    minimum, when, endpoint = _MODULE.within_period_asset_minima(states, controls, params, 0.7)
    for index in range(len(states)):
        c, h, q = controls[index]
        a0, log_k = states[index]
        slope = np.sqrt(1.25) - 0.5
        income = h * (1 - slope * q / h - slope**2 * (q / h) ** 2) * np.exp(log_k)
        growth = params.human_capital_productivity * q - params.human_capital_depreciation
        solution = solve_ivp(
            lambda t, a, income=income, growth=growth, c=c: (
                rate * a + income * np.exp(growth * t) - c
            ),
            (0, 0.7),
            [a0],
            rtol=2e-12,
            atol=2e-13,
            dense_output=True,
        )
        assert solution.success
        optimized: Any = minimize_scalar(
            lambda t, solution=solution: float(solution.sol(t)[0]),
            bounds=(0, 0.7),
            method="bounded",
            options={"xatol": 1e-13},
        )
        expected = min(a0, float(solution.y[0, -1]), optimized.fun)
        assert minimum[index] == pytest.approx(expected, abs=4e-12, rel=2e-11)
        assert endpoint[index] == pytest.approx(solution.y[0, -1], abs=4e-12, rel=2e-11)
        assert minimum[index] == pytest.approx(solution.sol(when[index])[0], abs=4e-12, rel=2e-11)


def test_checkpoint_contact_can_hide_a_genuine_interior_violation():
    params = benchmark_params()
    floor, horizon = 1e-4, 0.5
    state = np.array([floor, 0.0])
    h, q = 0.6, 0.4
    growth = params.human_capital_productivity * q - params.human_capital_depreciation
    income = float(_MODULE.earnings_share(np.array(h), np.array(q)))
    checkpoint = horizon / 4
    consumption = (
        floor * np.exp(params.interest_rate * checkpoint)
        + income
        * np.exp(params.interest_rate * checkpoint)
        * checkpoint
        * exprel((growth - params.interest_rate) * checkpoint)
        - floor
    ) / (checkpoint * exprel(params.interest_rate * checkpoint))
    control = np.array([consumption, h, q])
    minimum, when, endpoint = _MODULE.within_period_asset_minima(state, control, params, horizon)
    assert 0 < float(when) < checkpoint
    assert float(minimum) < floor - 1e-5
    for t in np.linspace(checkpoint, horizon, 4):
        _, _, at_t = _MODULE.within_period_asset_minima(state, control, params, t)
        assert float(at_t) >= floor - 1e-13
    audit, _, _ = _MODULE.path_audit(
        np.array([0, horizon]),
        np.array([[state], [[endpoint, growth * horizon]]]),
        control.reshape(1, 1, 3),
        params,
        floor,
        np.array([1.0]),
        1e-10,
    )
    assert audit["violating_interval_count"] == 1
    assert audit["weight_of_paths_with_any_numerical_floor_violation"] == 1
    assert audit["maximum_transition_reproduction_error"] < 1e-14


def test_common_physical_sampling_does_not_confuse_time_step_and_policy_changes():
    params = benchmark_params()
    initial = np.array([[3.0, 0.1], [2.0, -0.2]])
    control = np.array([[0.4, 0.5, 0.2], [0.3, 0.4, 0.1]])
    runs = []
    for periods in (4, 8):
        time = np.linspace(0, 2, periods + 1)
        states = [initial]
        for _ in range(periods):
            _, _, assets = _MODULE.within_period_asset_minima(
                states[-1], control, params, 2 / periods
            )
            capital = (
                states[-1][:, 1]
                + (
                    params.human_capital_productivity * control[:, 2]
                    - params.human_capital_depreciation
                )
                * 2
                / periods
            )
            states.append(np.column_stack((assets, capital)))
        runs.append((time, np.asarray(states), np.broadcast_to(control, (periods, 2, 3))))
    state_time = np.linspace(0, 2, 17)
    moment_time = (state_time[:-1] + state_time[1:]) / 2
    first = _MODULE.common_profiles(
        *runs[0], params, np.array([0.3, 0.7]), state_time, moment_time, 0.02
    )
    second = _MODULE.common_profiles(
        *runs[1], params, np.array([0.3, 0.7]), state_time, moment_time, 0.02
    )
    for key in first[0]:
        np.testing.assert_allclose(first[0][key], second[0][key], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(first[1], second[1], rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(first[2], second[2])


def test_saved_population_measurements_preserve_weights_and_reject_mismatched_states():
    from types import SimpleNamespace

    time = np.array([0.0, 0.5, 1.0])
    states = np.array(
        [[[3.0, 0.1], [4.0, -0.2]], [[2.9, 0.08], [3.8, -0.22]], [[2.8, 0.06], [3.6, -0.24]]]
    )
    controls = np.broadcast_to(np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.0]]), (2, 2, 3))
    weights = np.array([0.3, 0.7])
    nodes = SimpleNamespace(
        assets=states[0, :, 0], human_capital=np.exp(states[0, :, 1]), weights=weights
    )
    solution = SimpleNamespace(time=time)
    arrays = {
        "run_0_time": time,
        "run_0_states": states,
        "run_0_controls": controls,
        "run_0_policy_values": np.zeros((2, 2)),
        "run_0_native_floor_mass": np.zeros(3),
    }
    recovered = _MODULE.population_from_saved(
        solution, nodes, {"array_prefix": "run_0"}, arrays, Path("validated_report.json")
    )
    np.testing.assert_array_equal(recovered.simulation.states, states)
    np.testing.assert_allclose(recovered.moments.consumption, controls[..., 0] @ weights)
    np.testing.assert_allclose(recovered.state_moments.assets, states[..., 0] @ weights)
    mismatched = SimpleNamespace(
        assets=nodes.assets + 1e-3, human_capital=nodes.human_capital, weights=weights
    )
    with pytest.raises(ValueError, match="initial states"):
        _MODULE.population_from_saved(
            solution, mismatched, {"array_prefix": "run_0"}, arrays, Path("validated_report.json")
        )
    with pytest.raises(ValueError, match="time grid"):
        _MODULE.population_from_saved(
            SimpleNamespace(time=time + 0.01),
            nodes,
            {"array_prefix": "run_0"},
            arrays,
            Path("validated_report.json"),
        )
