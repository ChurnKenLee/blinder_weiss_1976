"""Independent references for the validation tool, not the production bound."""

import importlib.util
import sys
from pathlib import Path

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
            lambda t, a: rate * a + income * np.exp(growth * t) - c,
            (0, 0.7),
            [a0],
            rtol=2e-12,
            atol=2e-13,
            dense_output=True,
        )
        assert solution.success
        optimized = minimize_scalar(
            lambda t: float(solution.sol(t)[0]),
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
