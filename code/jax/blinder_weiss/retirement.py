"""Exact all-retired reference for the constant-control Bellman time step.

This solves the restricted problem ``h = q = 0`` at every remaining date.
It does not establish that retirement beats working in the full model. The
unconstrained formulas require a common, nonzero consumption/bequest power
``p < 1`` and positive utility weights. ``RetirementReference.path`` checks
that economic and optional computational-domain constraints remain slack
before the result is treated as a feasible-policy lower bound.

For step ``d``, write ``E = exp(r*d)``, ``F = integral_0^d exp(r*s) ds``,
``beta = exp(-rho*d)`` and ``D = integral_0^d exp(-rho*s) ds``. Retirement
has the exact transition ``A[n+1] = E*A[n] - F*c[n]``. Its discounted resource
budget is ``sum(P[n]*c[n]) + P[T]*A[T] = A[0]``, with prices
``P[n] = F/E**(n+1)`` and ``P[T] = E**(-N)``. Objective weights are
``w[n] = D*consumption_weight*beta**n`` and ``w[T] = bequest_weight*beta**N``.

Set ``sigma = 1-p`` and ``S = sum(w**(1/sigma)*P**(-p/sigma))``, including
the terminal asset as the last allocation. The unique restricted optimum is
``allocation[i] = A[0]*(w[i]/P[i])**(1/sigma)/S``. Consequently,
``c[n+1]/c[n] = exp((r-rho)*d/sigma)`` and
``A[T]/c[N-1] = (beta*F*bequest_weight/(D*consumption_weight))**(1/sigma)``.
The exact value is ``S**sigma * A[0]**p/p`` plus the discounted utility of
full leisure. Both asset derivatives follow analytically. These values are
discounted relative to the start of the remaining horizon, as in Bellman V.

The normalized asset path uses the unspent discounted budget in reverse
log-sum-exp order. This avoids subtracting nearly equal cumulative spending
from initial wealth near the terminal date. No JAX compilation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ModelParams


def _positive_assets(assets: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(assets, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("assets must be finite and strictly positive")
    return result


def _exponential_integral(rate: float, duration: float) -> float:
    """Integral of exp(rate*t), including exactly zero rate/duration."""
    if rate == 0.0:
        return duration
    return float(np.expm1(rate * duration) / rate)


@dataclass(frozen=True)
class RetirementPath:
    """An exact restricted optimum with checked, strictly slack state bounds.

    Controls are constant within each interval. Since retired assets satisfy
    ``A_dot(t) = exp(r*t)*(r*A_start-c)`` within an interval, their extrema
    occur at its endpoints; checking these nodes checks the entire path.
    Log human capital is linear in time and is checked the same way.
    """

    reference: RetirementReference
    assets: NDArray[np.float64]
    consumption: NDArray[np.float64]
    log_human_capital: NDArray[np.float64]
    value: float
    minimum_asset_slack: float
    maximum_asset_slack: float | None

    @property
    def time(self) -> NDArray[np.float64]:
        return self.reference.time

    @property
    def human_capital(self) -> NDArray[np.float64]:
        return np.exp(self.log_human_capital)


@dataclass(frozen=True)
class RetirementReference:
    """Scale-free exact solution for one remaining horizon and parameter set.

    ``consumption_per_initial_asset`` has N entries; ``assets_per_initial_asset``
    has N+1. N=0 gives the terminal bequest without any consumption intervals.
    Value methods describe the unconstrained restricted problem. Call ``path``
    with the relevant state bounds before using them as feasible lower bounds
    for a numerically bounded Bellman problem.
    """

    params: ModelParams
    periods: int
    step: float
    consumption_growth_factor: float
    consumption_per_initial_asset: NDArray[np.float64]
    assets_per_initial_asset: NDArray[np.float64]
    value_scale: float
    leisure_value: float

    @property
    def time(self) -> NDArray[np.float64]:
        return self.step * np.arange(self.periods + 1, dtype=np.float64)

    def value(self, assets: ArrayLike) -> NDArray[np.float64]:
        """Exact restricted value, including full-leisure utility and bequest."""
        power = self.params.consumption_power
        return self.value_scale * _positive_assets(assets) ** power / power + self.leisure_value

    def marginal_asset_value(self, assets: ArrayLike) -> NDArray[np.float64]:
        """Exact first asset derivative of the restricted value."""
        return self.value_scale * _positive_assets(assets) ** (self.params.consumption_power - 1.0)

    def asset_value_curvature(self, assets: ArrayLike) -> NDArray[np.float64]:
        """Exact second asset derivative; strictly negative on positive assets."""
        power = self.params.consumption_power
        return (power - 1.0) * self.value_scale * _positive_assets(assets) ** (power - 2.0)

    def path(
        self,
        initial_assets: float | None = None,
        initial_human_capital: float | None = None,
        *,
        asset_minimum: float | None = None,
        asset_maximum: float | None = None,
        log_human_capital_minimum: float | None = None,
        log_human_capital_maximum: float | None = None,
        consumption_floor: float = 0.0,
    ) -> RetirementPath:
        """Scale the solution and reject binding or violated path/domain bounds.

        The economic ``params.asset_floor`` always applies. An optional stricter
        numerical asset floor, asset ceiling, log-capital domain, or consumption
        floor may also be supplied. Domain bounds are optional because the
        restricted economic problem itself has no finite computational domain.
        No clipped or constrained approximation is returned when a bound binds.
        """
        params = self.params
        initial_assets = params.initial_assets if initial_assets is None else float(initial_assets)
        initial_human_capital = (
            params.initial_human_capital
            if initial_human_capital is None
            else float(initial_human_capital)
        )
        _positive_assets(initial_assets)
        if not np.isfinite(initial_human_capital) or initial_human_capital <= 0.0:
            raise ValueError("initial_human_capital must be finite and strictly positive")
        if not np.isfinite(consumption_floor) or consumption_floor < 0.0:
            raise ValueError("consumption_floor must be finite and nonnegative")
        bounds = (
            asset_minimum,
            asset_maximum,
            log_human_capital_minimum,
            log_human_capital_maximum,
        )
        if any(bound is not None and not np.isfinite(bound) for bound in bounds):
            raise ValueError("supplied state bounds must be finite")
        floor = max(params.asset_floor, params.asset_floor if asset_minimum is None else asset_minimum)
        assets = initial_assets * self.assets_per_initial_asset
        consumption = initial_assets * self.consumption_per_initial_asset
        log_capital = (
            np.log(initial_human_capital) - params.human_capital_depreciation * self.time
        )
        if not all(np.all(np.isfinite(array)) for array in (assets, consumption, log_capital)):
            raise ValueError("retirement path is not representable in float64")
        minimum_slack = float(np.min(assets) - floor)
        if minimum_slack <= 0.0:
            raise ValueError("retirement asset floor binds or is violated")
        maximum_slack = None if asset_maximum is None else float(asset_maximum - np.max(assets))
        if maximum_slack is not None and maximum_slack <= 0.0:
            raise ValueError("retirement asset ceiling binds or is violated")
        if np.any(consumption <= consumption_floor):
            raise ValueError("retirement consumption floor binds or is violated")
        if log_human_capital_minimum is not None and np.min(log_capital) <= log_human_capital_minimum:
            raise ValueError("retirement log-human-capital lower bound binds or is violated")
        if log_human_capital_maximum is not None and np.max(log_capital) >= log_human_capital_maximum:
            raise ValueError("retirement log-human-capital upper bound binds or is violated")
        return RetirementPath(
            reference=self,
            assets=assets,
            consumption=consumption,
            log_human_capital=log_capital,
            value=float(self.value(initial_assets)),
            minimum_asset_slack=minimum_slack,
            maximum_asset_slack=maximum_slack,
        )


def retirement_reference(params: ModelParams, *, periods: int, step: float) -> RetirementReference:
    """Construct the exact discrete all-retired reference for a remaining horizon.

    Supports any finite r and rho, a common nonzero consumption/bequest power
    p<1, and positive consumption/bequest weights. The logarithmic (p=0),
    differing-power and constrained retirement problems are explicitly rejected.
    ``periods * step`` is the remaining horizon and need not equal the full
    model horizon. Zero periods returns the terminal bequest reference.
    """
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 0:
        raise ValueError("periods must be a nonnegative integer")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and strictly positive")
    if not all(np.isfinite(value) for value in params):
        raise ValueError("model parameters must be finite")
    power = params.consumption_power
    if power == 0.0 or power >= 1.0 or params.bequest_power != power:
        raise ValueError("consumption and bequest require a common nonzero power below one")
    if params.consumption_weight <= 0.0 or params.bequest_weight <= 0.0:
        raise ValueError("consumption and bequest weights must be strictly positive")
    if params.leisure_power == 0.0:
        raise ValueError("zero leisure power is unsupported by the model's utility convention")
    periods = int(periods)
    step = float(step)
    sigma = 1.0 - power
    duration = periods * step
    if not np.isfinite(duration):
        raise ValueError("remaining horizon must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        flow_discount = _exponential_integral(-params.rho, step)
        consumption_factor = _exponential_integral(params.interest_rate, step)
        n = np.arange(periods, dtype=np.float64)
        log_weights = np.concatenate(
            (
                np.log(flow_discount * params.consumption_weight) - params.rho * step * n,
                [np.log(params.bequest_weight) - params.rho * duration],
            )
        )
        log_prices = np.concatenate(
            (
                np.log(consumption_factor) - params.interest_rate * step * (n + 1.0),
                [-params.interest_rate * duration],
            )
        )
        log_terms = (log_weights - power * log_prices) / sigma
        log_remaining_budget = np.logaddexp.accumulate(log_terms[::-1])[::-1]
        log_s = log_remaining_budget[0]
        consumption_scale = np.exp((log_weights[:-1] - log_prices[:-1]) / sigma - log_s)
        asset_scale = np.exp(
            log_remaining_budget - log_s + params.interest_rate * step * np.arange(periods + 1)
        )
        value_scale = float(np.exp(sigma * log_s))
        leisure_value = (
            params.leisure_weight
            / params.leisure_power
            * _exponential_integral(-params.rho, duration)
        )
        growth_factor = float(np.exp((params.interest_rate - params.rho) * step / sigma))
    quantities = [consumption_scale, asset_scale, value_scale, leisure_value, growth_factor]
    if not all(np.all(np.isfinite(quantity)) for quantity in quantities):
        raise ValueError("retirement coefficients are not representable in float64")
    if (
        np.any(consumption_scale <= 0.0)
        or np.any(asset_scale <= 0.0)
        or value_scale <= 0.0
        or growth_factor <= 0.0
    ):
        raise ValueError("retirement coefficients underflowed float64")
    consumption_scale.setflags(write=False)
    asset_scale.setflags(write=False)
    return RetirementReference(
        params=params,
        periods=periods,
        step=step,
        consumption_growth_factor=growth_factor,
        consumption_per_initial_asset=consumption_scale,
        assets_per_initial_asset=asset_scale,
        value_scale=value_scale,
        leisure_value=float(leisure_value),
    )
