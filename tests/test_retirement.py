"""Independent numerical checks for the restricted discrete retirement optimum."""

import numpy as np
import pytest
from blinder_weiss.model import benchmark_params
from blinder_weiss.retirement import retirement_reference
from scipy.integrate import quad, solve_ivp
from scipy.optimize import Bounds, LinearConstraint, minimize


@pytest.mark.parametrize(
    ("power", "interest", "rho", "step", "periods"),
    [(-1.0, 0.05, 0.03, 0.5, 5), (-0.3, -0.02, 0.01, 1.2, 4), (0.4, 0.0, 0.0, 0.7, 6)],
)
def test_matches_independent_scipy_resource_allocation(power, interest, rho, step, periods):
    params = benchmark_params(
        consumption_power=power,
        bequest_power=power,
        consumption_weight=1.3,
        bequest_weight=0.8,
        interest_rate=interest,
        rho=rho,
    )
    wealth = 3.7
    reference = retirement_reference(params, periods=periods, step=step)
    path = reference.path(wealth)
    # Build the finite-dimensional budget independently, using quadrature for
    # both within-period integrals. SLSQP optimizes every c[n] and terminal A.
    flow_integral = quad(lambda s: np.exp(-rho * s), 0.0, step)[0]
    asset_integral = quad(lambda s: np.exp(interest * s), 0.0, step)[0]
    times = step * np.arange(periods)
    weights = np.r_[
        params.consumption_weight * flow_integral * np.exp(-rho * times),
        params.bequest_weight * np.exp(-rho * periods * step),
    ]
    prices = np.r_[
        asset_integral * np.exp(-interest * (times + step)),
        np.exp(-interest * periods * step),
    ]
    initial = np.full(periods + 1, wealth / prices.sum())
    result = minimize(
        lambda allocation: -np.dot(weights, allocation**power / power),
        initial,
        jac=lambda allocation: -weights * allocation ** (power - 1.0),
        bounds=Bounds(1e-10, np.inf),
        constraints=LinearConstraint(prices, wealth, wealth),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    assert result.success, result.message
    expected = np.r_[path.consumption, path.assets[-1]]
    np.testing.assert_allclose(expected, result.x, rtol=2e-6, atol=2e-7)
    leisure_value = (
        params.leisure_weight
        / params.leisure_power
        * quad(lambda t: np.exp(-rho * t), 0.0, periods * step)[0]
    )
    assert path.value == pytest.approx(-result.fun + leisure_value, rel=2e-12, abs=2e-12)
    multipliers = weights * expected ** (power - 1.0) / prices
    np.testing.assert_allclose(multipliers, reference.marginal_asset_value(wealth), rtol=2e-13)
    assert np.dot(prices, expected) == pytest.approx(wealth, rel=2e-14)


@pytest.mark.parametrize("interest", [-0.03, 0.0, 0.08])
def test_exact_retired_ode_and_unconstrained_path_feasibility(interest):
    params = benchmark_params(interest_rate=interest, initial_assets=5.0)
    reference = retirement_reference(params, periods=6, step=0.7)
    path = reference.path(
        asset_minimum=0.0,
        asset_maximum=6.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
    )
    for n, consumption in enumerate(path.consumption):
        result = solve_ivp(
            lambda _t, assets, consumption=consumption: interest * assets - consumption,
            (0.0, reference.step),
            [path.assets[n]],
            rtol=2e-12,
            atol=2e-13,
            dense_output=True,
        )
        assert result.success
        assert result.y[0, -1] == pytest.approx(path.assets[n + 1], rel=2e-11, abs=2e-12)
        sampled = result.sol(np.linspace(0.0, reference.step, 51))[0]
        assert np.min(sampled) >= min(path.assets[n : n + 2]) - 2e-11
        assert np.max(sampled) <= max(path.assets[n : n + 2]) + 2e-11
    sigma = 1.0 - params.consumption_power
    np.testing.assert_allclose(
        path.consumption[1:] / path.consumption[:-1],
        np.exp((interest - params.rho) * reference.step / sigma),
        rtol=2e-14,
    )
    flow_integral = quad(lambda s: np.exp(-params.rho * s), 0.0, reference.step)[0]
    asset_integral = quad(lambda s: np.exp(interest * s), 0.0, reference.step)[0]
    terminal_ratio = (
        np.exp(-params.rho * reference.step)
        * asset_integral
        * params.bequest_weight
        / (flow_integral * params.consumption_weight)
    ) ** (1.0 / sigma)
    assert path.assets[-1] / path.consumption[-1] == pytest.approx(terminal_ratio, rel=2e-14)
    assert path.minimum_asset_slack > 0.0
    assert path.maximum_asset_slack is not None
    assert path.maximum_asset_slack > 0.0


@pytest.mark.parametrize(
    "power,interest,rho", [(-1.0, 0.05, 0.03), (0.4, 0.06, 0.02), (-1.0, 0.0, 0.03)]
)
def test_converges_to_independent_continuous_retirement_limit(power, interest, rho):
    horizon, wealth = 8.0, 4.0
    params = benchmark_params(
        consumption_power=power,
        bequest_power=power,
        interest_rate=interest,
        rho=rho,
        consumption_weight=1.3,
        bequest_weight=0.8,
    )
    sigma = 1.0 - power
    growth = (interest - rho) / sigma
    terminal_ratio = (params.bequest_weight / params.consumption_weight) ** (1.0 / sigma)
    resource_integral = quad(lambda t: np.exp((growth - interest) * t), 0.0, horizon)[0]
    c0 = wealth / (resource_integral + terminal_ratio * np.exp((growth - interest) * horizon))
    terminal_assets = terminal_ratio * c0 * np.exp(growth * horizon)
    continuous_value = (
        quad(
            lambda t: (
                np.exp(-rho * t)
                * (
                    params.consumption_weight * (c0 * np.exp(growth * t)) ** power / power
                    + params.leisure_weight / params.leisure_power
                )
            ),
            0.0,
            horizon,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        + np.exp(-rho * horizon) * params.bequest_weight * terminal_assets**power / power
    )
    consumption_errors, value_errors = [], []
    for periods in (20, 40, 80, 160):
        path = retirement_reference(params, periods=periods, step=horizon / periods).path(wealth)
        consumption_errors.append(abs(path.consumption[0] - c0))
        value_errors.append(abs(path.value - continuous_value))
        # Piecewise constant controls restrict the continuous feasible set.
        assert path.value <= continuous_value + 2e-12
    assert np.max(np.array(consumption_errors[1:]) / consumption_errors[:-1]) < 0.55
    assert np.max(np.array(value_errors[1:]) / value_errors[:-1]) < 0.3
    assert value_errors[-1] / max(1.0, abs(continuous_value)) < 5e-6


def test_value_derivatives_scaling_and_terminal_only_case():
    params = benchmark_params()
    reference = retirement_reference(params, periods=40, step=0.5)
    assets = np.array([1.0, 3.0, 8.0])
    perturbation = 1e-4
    numerical_first = (
        reference.value(assets + perturbation) - reference.value(assets - perturbation)
    ) / (2.0 * perturbation)
    numerical_second = (
        reference.marginal_asset_value(assets + perturbation)
        - reference.marginal_asset_value(assets - perturbation)
    ) / (2.0 * perturbation)
    np.testing.assert_allclose(reference.marginal_asset_value(assets), numerical_first, rtol=2e-8)
    np.testing.assert_allclose(reference.asset_value_curvature(assets), numerical_second, rtol=3e-8)
    assert np.all(reference.marginal_asset_value(assets) > 0.0)
    assert np.all(reference.asset_value_curvature(assets) < 0.0)
    np.testing.assert_allclose(
        reference.path(6.0).consumption, 2.0 * reference.path(3.0).consumption
    )
    terminal = retirement_reference(params, periods=0, step=0.5)
    terminal_path = terminal.path(3.0)
    assert terminal_path.consumption.size == 0
    np.testing.assert_equal(terminal_path.assets, [3.0])
    assert terminal_path.value == pytest.approx(
        params.bequest_weight * 3.0**params.bequest_power / params.bequest_power
    )
    assert terminal.leisure_value == 0.0


def test_rejects_binding_floors_and_computational_domains():
    reference = retirement_reference(benchmark_params(), periods=20, step=0.5)
    path = reference.path()
    with pytest.raises(ValueError, match="asset floor"):
        reference.path(asset_minimum=float(np.min(path.assets)))
    with pytest.raises(ValueError, match="asset ceiling"):
        reference.path(asset_maximum=float(np.max(path.assets)))
    with pytest.raises(ValueError, match="consumption floor"):
        reference.path(consumption_floor=float(np.min(path.consumption)))
    with pytest.raises(ValueError, match="lower bound"):
        reference.path(log_human_capital_minimum=float(np.min(path.log_human_capital)))
    with pytest.raises(ValueError, match="upper bound"):
        reference.path(log_human_capital_maximum=float(np.max(path.log_human_capital)))
    economic_floor = retirement_reference(benchmark_params(asset_floor=10.0), periods=20, step=0.5)
    with pytest.raises(ValueError, match="asset floor"):
        economic_floor.path(asset_minimum=-1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"consumption_power": 0.0, "bequest_power": 0.0},
        {"consumption_power": 1.1, "bequest_power": 1.1},
        {"bequest_power": -2.0},
        {"consumption_weight": 0.0},
        {"bequest_weight": -1.0},
        {"leisure_power": 0.0},
    ],
)
def test_rejects_unsupported_preferences(overrides):
    with pytest.raises(ValueError):
        retirement_reference(benchmark_params(**overrides), periods=2, step=0.5)
