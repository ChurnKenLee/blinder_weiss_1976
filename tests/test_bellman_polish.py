"""Full recursion checks for continuous consumption recovery."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from blinder_weiss import BellmanConfig, benchmark_params, diagnose_bellman, solve_bellman
from blinder_weiss.bellman import maximum_feasible_consumption


def test_local_refinement_can_go_below_consumption_seed_minimum():
    params = benchmark_params(horizon=0.005, initial_assets=35.0)
    cfg = BellmanConfig(periods=1, asset_nodes=4, human_capital_nodes=3,
                        asset_maximum=40.0, hours_nodes=3, investment_nodes=3,
                        consumption_nodes=3, neighbor_policy_sweeps=0,
                        refinement_steps=0, compute_platform='cpu')
    coarse = solve_bellman(params, cfg)
    refined = solve_bellman(params, replace(cfg, refinement_steps=32))
    assets, logs = jnp.meshgrid(jnp.asarray(refined.asset_grid),
                               jnp.asarray(refined.log_human_capital_grid), indexing='ij')
    capacity = np.asarray(maximum_feasible_consumption(
        assets, logs, jnp.asarray(refined.hours_policy[0]),
        jnp.asarray(refined.training_time_policy[0]), params, params.horizon,
        cfg.asset_minimum, cfg.path_checkpoints))
    assert np.min(refined.consumption_policy[0, -1] / capacity[-1]) < 0.01
    assert np.max(refined.values[0, -1] - coarse.values[0, -1]) > 1e-4


def test_consumption_polish_preserves_bellman_identity_over_recursion():
    cfg = BellmanConfig(periods=3, asset_nodes=7, human_capital_nodes=5,
                        hours_nodes=5, investment_nodes=5, consumption_nodes=5,
                        refinement_steps=4, consumption_polish=True, compute_platform='cpu')
    solution = solve_bellman(benchmark_params(horizon=3.0), cfg)
    diagnostics = diagnose_bellman(solution)
    assert diagnostics.accepted_node_solution
    assert diagnostics.maximum_node_bellman_residual < 1e-9
    assert diagnostics.minimum_consumption_capacity_slack >= -1e-10
    assert diagnostics.simulation_stayed_in_domain
