"""Stationarity depends on admissible duals at the returned primal iterate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from blinder_weiss import (
    SolverConfig,
    benchmark_params,
    diagnose_path,
    diagnostics,
    solve_lifecycle,
)
from scipy.optimize import OptimizeResult


@pytest.fixture(scope="module")
def coarse_primal():
    # This exact solve produced stale reported SLSQP multipliers after the
    # earnings extension: raw residual 2.86e-4, reconstructed residual <1e-6.
    return solve_lifecycle(config=SolverConfig(intervals=8, max_iterations=1000))


def _with_multipliers(result, value):
    optimizer_result = OptimizeResult(result.optimizer_result)
    optimizer_result.multipliers = np.full_like(optimizer_result.multipliers, value)
    return replace(result, optimizer_result=optimizer_result)


def test_coarse_primal_passes_with_reconstructed_stationarity(coarse_primal) -> None:
    before = coarse_primal.decision.copy()
    diagnostic = diagnose_path(coarse_primal, independent_integration=False)
    assert diagnostic.accepted_success
    assert diagnostic.projected_kkt_residual < 1e-4
    np.testing.assert_array_equal(coarse_primal.decision, before)


@pytest.mark.parametrize("bad_multiplier", [0.0, 1e6, -1e6, np.nan])
def test_corrupted_duals_do_not_reject_an_unchanged_good_primal(
    coarse_primal, monkeypatch: pytest.MonkeyPatch, bad_multiplier: float
) -> None:
    result = _with_multipliers(coarse_primal, bad_multiplier)
    original_lsq = diagnostics.lsq_linear
    reconstructions = []

    def counted_reconstruction(*args, **kwargs):
        reconstructions.append(True)
        return original_lsq(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "lsq_linear", counted_reconstruction)
    diagnostic = diagnose_path(result, independent_integration=False)
    assert len(reconstructions) == 1
    assert diagnostic.accepted_success
    assert diagnostic.projected_kkt_residual < 1e-4
    np.testing.assert_array_equal(result.decision, coarse_primal.decision)


def test_reconstruction_rejects_a_feasible_but_nonstationary_primal(coarse_primal) -> None:
    # Utility weights do not enter dynamics or constraints. The same feasible
    # path is suboptimal for this objective, which valid duals cannot disguise.
    result = replace(
        _with_multipliers(coarse_primal, 0.0),
        params=coarse_primal.params._replace(consumption_weight=5.0),
    )
    assert result.success
    diagnostic = diagnose_path(result, independent_integration=False)
    assert diagnostic.projected_kkt_residual > 1e-4
    assert not diagnostic.accepted_success


def test_reconstruction_does_not_accept_a_failed_infeasible_solve() -> None:
    failed = solve_lifecycle(
        benchmark_params(), SolverConfig(intervals=8, max_iterations=1)
    )
    assert not failed.success
    assert failed.max_constraint_violation > failed.config.constraint_tolerance
    diagnostic = diagnose_path(_with_multipliers(failed, 0.0), independent_integration=False)
    assert not diagnostic.accepted_success
