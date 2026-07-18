from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from blinder_weiss.model import benchmark_params
from blinder_weiss.transcription import (
    DecisionLayout,
    asset_path_slack,
    equality_constraints,
    make_initial_guess,
    scaled_negative_objective,
    training_time_slack,
)


def test_initial_guess_is_exactly_trapezoid_feasible() -> None:
    params = benchmark_params()
    intervals = 12
    decision = make_initial_guess(params, intervals)
    residual = equality_constraints(jnp.asarray(decision), params, intervals, "trapezoid")
    assert decision.shape == (DecisionLayout(intervals).size,)
    assert float(jnp.max(jnp.abs(residual))) < 1e-12
    assert float(jnp.min(training_time_slack(jnp.asarray(decision), intervals))) >= 0.0
    assert (
        float(jnp.min(asset_path_slack(jnp.asarray(decision), params, intervals, "trapezoid")))
        >= 0.0
    )


def test_autodiff_objective_and_constraint_jacobian_are_float64_and_finite() -> None:
    params = benchmark_params()
    intervals = 4
    decision = jnp.asarray(make_initial_guess(params, intervals))

    def objective(vector):
        return scaled_negative_objective(vector, params, intervals, "hermite-simpson")

    def constraints(vector):
        return equality_constraints(vector, params, intervals, "hermite-simpson")

    gradient = jax.grad(objective)(decision)
    jacobian = jax.jacrev(constraints)(decision)
    assert gradient.dtype == jnp.float64
    assert jacobian.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.isfinite(np.asarray(jacobian)))
