from __future__ import annotations

import numpy as np
import pytest
from blinder_weiss import SolverConfig, diagnose_path, solve_lifecycle


@pytest.fixture(scope="module")
def coarse_solution():
    return solve_lifecycle(
        config=SolverConfig(
            intervals=8,
            scheme="hermite-simpson",
            max_iterations=1_000,
        )
    )


def test_coarse_solver_is_feasible_and_respects_bounds(coarse_solution) -> None:
    result = coarse_solution
    assert result.success, result.message
    assert result.max_constraint_violation < 1e-7
    assert np.min(result.assets) >= result.params.asset_floor - 1e-9
    assert np.min(result.consumption) > 0.0
    assert np.all(result.training_time >= -1e-9)
    assert np.all(result.training_time <= result.hours + 1e-8)


def test_coarse_solver_generates_the_four_lifecycle_regimes(coarse_solution) -> None:
    result = coarse_solution
    diagnostics = diagnose_path(result, independent_integration=False)
    assert diagnostics.accepted_success
    assert diagnostics.projected_kkt_residual < 1e-4
    assert diagnostics.minimum_collocation_path_assets >= result.params.asset_floor - 1e-8
    assert diagnostics.normal_regime_order
    assert diagnostics.regime_sequence == (
        "schooling → on-the-job training → pure work → retirement"
    )
    assert result.investment_share[0] > 1.0 - 1e-5
    assert result.hours[-1] < 1e-4
