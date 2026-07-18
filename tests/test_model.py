from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss.model import (
    benchmark_params,
    dynamics,
    earnings_tradeoff,
    earnings_tradeoff_prime,
    effective_earnings_share,
)


def test_benchmark_has_normal_lifecycle_discount_condition() -> None:
    params = benchmark_params()
    assert params.rho < params.interest_rate + params.human_capital_depreciation
    assert params.initial_assets > params.asset_floor
    assert params.human_capital_productivity == pytest.approx(0.22)


def test_earnings_tradeoff_has_required_endpoints_and_concavity() -> None:
    assert float(earnings_tradeoff(jnp.asarray(0.0))) == pytest.approx(1.0)
    assert float(earnings_tradeoff(jnp.asarray(1.0))) == pytest.approx(0.0, abs=1e-14)
    assert float(earnings_tradeoff_prime(jnp.asarray(0.0))) < 0.0
    assert np.isfinite(float(earnings_tradeoff_prime(jnp.asarray(1.0))))
    second_derivative = jax.grad(jax.grad(earnings_tradeoff))(jnp.asarray(0.5))
    assert float(second_derivative) < 0.0


def test_training_time_perspective_matches_original_controls() -> None:
    hours = jnp.asarray([0.2, 0.5, 0.9])
    investment_share = jnp.asarray([0.0, 0.4, 1.0])
    training_time = hours * investment_share
    expected = hours * earnings_tradeoff(investment_share)
    np.testing.assert_allclose(
        effective_earnings_share(hours, training_time), expected, rtol=1e-13, atol=1e-13
    )
    assert float(effective_earnings_share(jnp.asarray(0.0), jnp.asarray(0.0))) == 0.0


def test_log_human_capital_dynamics_use_training_time() -> None:
    params = benchmark_params()
    state = jnp.asarray([params.initial_assets, jnp.log(params.initial_human_capital)])
    control = jnp.asarray([1.0, 0.6, 0.3])
    state_growth = dynamics(state, control, params)
    expected_log_growth = (
        params.human_capital_productivity * control[2] - params.human_capital_depreciation
    )
    assert float(state_growth[1]) == pytest.approx(float(expected_log_growth))
