"""Economic primitives for the Blinder--Weiss lifecycle problem.

The implementation uses labor hours ``h`` throughout. Leisure is ``1 - h``.
The terminal bequest is discounted by ``exp(-rho * T)`` in the objective.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class ModelParams(NamedTuple):
    """Parameters and initial conditions for one deterministic lifecycle."""

    horizon: float = 70.0
    rho: float = 0.03
    interest_rate: float = 0.05
    human_capital_depreciation: float = 0.05
    human_capital_productivity: float = 0.22
    consumption_power: float = -1.0
    consumption_weight: float = 1.0
    leisure_power: float = -1.0
    leisure_weight: float = 1.0
    bequest_power: float = -1.0
    bequest_weight: float = 1.0
    initial_assets: float = 5.0
    initial_human_capital: float = 1.0
    asset_floor: float = 0.0


def benchmark_params(**overrides: float) -> ModelParams:
    """Return the documented benchmark, optionally replacing named fields."""

    params = ModelParams()
    unknown = set(overrides).difference(params._fields)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown model parameter(s): {names}")
    return params._replace(**overrides)


def power_utility(quantity: Array, power: float) -> Array:
    """Power utility in the parameterization used by the legacy notebooks."""

    return jnp.power(quantity, power) / power


def marginal_power_utility(quantity: Array, power: float) -> Array:
    """Derivative of :func:`power_utility` with respect to quantity."""

    return jnp.power(quantity, power - 1.0)


def earnings_tradeoff(
    investment_share: Array,
) -> Array:
    """Concave job-earnings tradeoff ``g(x)`` used in the legacy work.

    It satisfies ``g(0) = 1``, ``g(1) = 0``, and has finite negative
    derivatives at both endpoints. Bounds on ``x`` are imposed by the NLP;
    clipping here would introduce unnecessary nondifferentiabilities.
    """

    slope = jnp.sqrt(1.25) - 0.5
    return 1.25 - jnp.square(slope * investment_share + 0.5)


def earnings_tradeoff_prime(
    investment_share: Array,
) -> Array:
    """Analytic derivative of :func:`earnings_tradeoff`."""

    slope = jnp.sqrt(1.25) - 0.5
    return -2.0 * slope * (slope * investment_share + 0.5)


def effective_earnings_share(hours: Array, training_time: Array) -> Array:
    """Return ``h * g(q / h)`` without dividing by hours.

    ``q = h * x`` is time spent accumulating human capital. The quadratic
    benchmark tradeoff has ``g(x) = 1 - b*x - b**2*x**2``, so its perspective
    is continuous at retirement ``(h, q) = (0, 0)``. The NLP imposes
    ``0 <= q <= h``.
    """

    slope = jnp.sqrt(1.25) - 0.5
    safe_hours = jnp.where(hours > 0.0, hours, 1.0)
    return (
        hours - slope * training_time - jnp.square(slope) * jnp.square(training_time) / safe_hours
    )


def flow_utility(
    consumption: Array,
    hours: Array,
    params: ModelParams,
) -> Array:
    """Instantaneous utility from consumption and leisure."""

    leisure = 1.0 - hours
    return params.consumption_weight * power_utility(
        consumption, params.consumption_power
    ) + params.leisure_weight * power_utility(leisure, params.leisure_power)


def bequest_utility(terminal_assets: Array, params: ModelParams) -> Array:
    """Undiscounted terminal bequest utility."""

    return params.bequest_weight * power_utility(terminal_assets, params.bequest_power)


def dynamics(
    state: Array,
    control: Array,
    params: ModelParams,
) -> Array:
    """Dynamics in assets and log human capital.

    ``state = [A, log(K)]`` and ``control = [c, h, q]``, where ``q = h*x`` is
    training time. Using ``log(K)`` enforces positive human capital and removes
    multiplicative growth from the second state equation.
    """

    assets, log_human_capital = state
    consumption, hours, training_time = control
    human_capital = jnp.exp(log_human_capital)
    earnings = effective_earnings_share(hours, training_time) * human_capital
    asset_growth = params.interest_rate * assets + earnings - consumption
    log_human_capital_growth = (
        params.human_capital_productivity * training_time - params.human_capital_depreciation
    )
    return jnp.stack((asset_growth, log_human_capital_growth))


batched_dynamics = jax.vmap(dynamics, in_axes=(0, 0, None))
