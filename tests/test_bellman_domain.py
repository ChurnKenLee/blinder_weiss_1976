"""Simulation must distinguish boundary roundoff from state-domain exits."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, bellman, benchmark_params, solve_bellman


@pytest.fixture(scope="module")
def domain_solution():
    return solve_bellman(
        benchmark_params(horizon=1.0),
        BellmanConfig(
            periods=1,
            asset_nodes=3,
            human_capital_nodes=3,
            asset_minimum=1e-4,
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


@pytest.mark.parametrize("policy_method", ["greedy", "interpolate"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("roundoff_only", [True, False])
def test_simulation_domain_flag_distinguishes_roundoff_from_real_exit(
    domain_solution,
    monkeypatch: pytest.MonkeyPatch,
    axis: int,
    side: str,
    roundoff_only: bool,
    policy_method: str,
) -> None:
    solution = domain_solution
    grids = (solution.asset_grid, solution.log_human_capital_grid)
    boundary = grids[axis][0 if side == "lower" else -1]
    direction = -1.0 if side == "lower" else 1.0
    endpoint = (
        np.nextafter(boundary, direction * np.inf)
        if roundoff_only
        else boundary + direction * 1e-6
    )
    initial_state = [solution.params.initial_assets, np.log(solution.params.initial_human_capital)]
    states = np.asarray([initial_state, initial_state])
    states[-1, axis] = endpoint

    def rollout(*_):
        return states, np.asarray([[1.0, 0.5, 0.0]]), np.asarray([-1.0])

    monkeypatch.setattr(
        bellman,
        "_cached_greedy_kernels",
        lambda *_: (None, rollout, jax.devices("cpu")[0]),
    )
    monkeypatch.setattr(bellman, "constant_control_transition", lambda *_: states[-1])
    simulation = bellman.simulate_policy(solution, policy_method=policy_method)
    assert simulation.stayed_in_domain is roundoff_only
