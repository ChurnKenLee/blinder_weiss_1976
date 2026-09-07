"""Keep the experimental continuation representation consistent end to end."""

from dataclasses import replace

import numpy as np
import pytest
from blinder_weiss import BellmanConfig, benchmark_params, diagnose_bellman, solve_bellman, value_at
from scipy.interpolate import PchipInterpolator


def test_pchip_recursion_queries_and_diagnostics_use_same_value_representation():
    cfg = BellmanConfig(
        periods=3,
        asset_nodes=7,
        human_capital_nodes=5,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=5,
        refinement_steps=4,
        value_interpolation="pchip",
        compute_platform="cpu",
    )
    s = solve_bellman(benchmark_params(horizon=3.0), cfg)
    d = diagnose_bellman(s)
    assert d.accepted_node_solution
    assert d.maximum_node_bellman_residual < 1e-9
    assert d.simulation_stayed_in_domain
    assets, log_k = 2.345, 0.134
    along_a = PchipInterpolator(s.asset_grid, s.values[0], axis=0)(assets)
    reference = PchipInterpolator(s.log_human_capital_grid, along_a)(log_k)
    assert float(value_at(s, 0, assets, np.exp(log_k))) == pytest.approx(
        float(reference), rel=1e-12
    )
    assert s.initial_value == pytest.approx(
        float(value_at(s, 0, s.params.initial_assets, s.params.initial_human_capital)), rel=1e-12
    )


@pytest.mark.parametrize("mode", ["pchip", "monotone_bicubic"])
def test_terminal_analytic_bellman_step_is_independent_of_interpolation(mode):
    cfg = BellmanConfig(
        periods=1,
        asset_nodes=5,
        human_capital_nodes=4,
        hours_nodes=3,
        investment_nodes=3,
        consumption_nodes=3,
        refinement_steps=3,
        compute_platform="cpu",
    )
    p = benchmark_params(horizon=1.0)
    linear = solve_bellman(p, cfg)
    cubic = solve_bellman(p, replace(cfg, value_interpolation=mode))
    np.testing.assert_allclose(cubic.values, linear.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        cubic.consumption_policy, linear.consumption_policy, rtol=1e-12, atol=1e-12
    )


def test_exact_bilinear_consumption_polish_cannot_silently_run_against_pchip():
    with pytest.raises(ValueError, match="requires bilinear"):
        solve_bellman(config=BellmanConfig(value_interpolation="pchip", consumption_polish=True))


def test_monotone_bicubic_full_recursion_keeps_values_increasing():
    cfg = BellmanConfig(
        periods=3,
        asset_nodes=7,
        human_capital_nodes=5,
        hours_nodes=5,
        investment_nodes=5,
        consumption_nodes=5,
        refinement_steps=4,
        value_interpolation="monotone_bicubic",
        compute_platform="cpu",
    )
    solution = solve_bellman(benchmark_params(horizon=3.0), cfg)
    diagnostics = diagnose_bellman(solution)
    assert diagnostics.accepted_node_solution
    assert diagnostics.maximum_node_bellman_residual < 1e-9
    assert diagnostics.maximum_value_monotonicity_violation < 1e-8
    assert diagnostics.simulation_stayed_in_domain
