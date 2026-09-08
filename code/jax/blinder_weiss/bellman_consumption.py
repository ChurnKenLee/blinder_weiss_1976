"""Exact conditional consumption recovery for the bilinear Bellman equation.

With hours and training fixed, next assets are affine in consumption. Each
asset interpolation interval therefore has a scalar power-utility objective
whose maximum is an endpoint or its analytic stationary point. Scanning the
intervals avoids assuming that the stored continuation values are concave.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .model import ModelParams, bequest_utility, effective_earnings_share, flow_utility


def optimize_conditional_consumption(
    states: Array,
    hours: Array,
    training_time: Array,
    continuation_values: Array,
    asset_grid: Array,
    log_human_capital_grid: Array,
    params: ModelParams,
    *,
    step: ArrayLike,
    asset_minimum: ArrayLike,
    consumption_floor: float,
    path_checkpoints: int,
    asset_feasibility: Literal["continuous", "checkpoints"] = "continuous",
    continuation_is_terminal: ArrayLike = False,
    incumbent_consumption: Array | None = None,
    terminal_iterations: int = 48,
) -> tuple[Array, Array]:
    """Return maximizing absolute consumption and its conditional Bellman value.

    All leading state dimensions are batched. Model parameters may be dynamic
    JAX inputs; checkpoint and iteration counts must be static under ``jit``.
    The control set uses the same selected full-period/legacy-checkpoint and
    next-state domain restriction
    as the main Bellman solver. No artificial minimum consumption *fraction*
    is imposed. A supplied feasible incumbent is retained whenever better.

    The nonterminal solve is global over consumption even for nonconcave
    continuation tables. For terminal utility, bracketed marginal-utility
    bisection assumes positive utility weights and concave power utilities
    (powers below one), as in the model's documented calibration.
    """

    # Import after module initialization so bellman can use this optional
    # optimizer without a circular import at module load time.
    from .bellman import _exprel, _interpolate_jax, maximum_feasible_consumption

    states = jnp.asarray(states)
    hours, training_time = jnp.broadcast_arrays(hours, training_time)
    assets = states[..., 0]
    log_human_capital = states[..., 1]
    duration = jnp.asarray(step)
    beta = jnp.exp(-params.rho * duration)
    flow_discount = duration * _exprel(-params.rho * duration)
    growth = params.human_capital_productivity * training_time - params.human_capital_depreciation
    asset_factor = jnp.exp(params.interest_rate * duration)
    consumption_factor = duration * _exprel(params.interest_rate * duration)
    income_factor = asset_factor * duration * _exprel((growth - params.interest_rate) * duration)
    assets_without_consumption = assets * asset_factor + income_factor * (
        effective_earnings_share(hours, training_time) * jnp.exp(log_human_capital)
    )
    next_log_human_capital = log_human_capital + growth * duration
    capacity = maximum_feasible_consumption(
        assets,
        log_human_capital,
        hours,
        training_time,
        params,
        duration,
        asset_minimum,
        path_checkpoints,
        method=asset_feasibility,
    )
    lower = jnp.maximum(
        consumption_floor,
        (assets_without_consumption - asset_grid[-1]) / consumption_factor,
    )
    upper = jnp.minimum(
        capacity,
        (assets_without_consumption - asset_grid[0]) / consumption_factor,
    )
    state_feasible = (
        (upper >= lower)
        & (next_log_human_capital >= log_human_capital_grid[0] - 1e-10)
        & (next_log_human_capital <= log_human_capital_grid[-1] + 1e-10)
    )
    is_terminal = jnp.asarray(continuation_is_terminal)

    def value_at_consumption(consumption: Array) -> Array:
        next_assets = assets_without_consumption - consumption_factor * consumption
        continuation = jax.lax.cond(
            is_terminal,
            lambda _: bequest_utility(jnp.maximum(next_assets, asset_grid[0]), params),
            lambda _: _interpolate_jax(
                continuation_values,
                asset_grid,
                log_human_capital_grid,
                next_assets,
                next_log_human_capital,
            ),
            operand=None,
        )
        value = flow_discount * flow_utility(consumption, hours, params) + beta * continuation
        feasible = (
            state_feasible
            & (consumption >= consumption_floor)
            & (consumption <= capacity + 1e-10)
            & (next_assets >= asset_grid[0] - 1e-10)
            & (next_assets <= asset_grid[-1] + 1e-10)
        )
        return jnp.where(feasible & jnp.isfinite(value), value, -jnp.inf)

    initial_consumption = (
        jnp.maximum(lower, consumption_floor)
        if incumbent_consumption is None
        else jnp.broadcast_to(incumbent_consumption, assets.shape)
    )
    initial = (initial_consumption, value_at_consumption(initial_consumption))

    def keep_better(
        carry: tuple[Array, Array], consumption: Array, value: Array
    ) -> tuple[Array, Array]:
        best_consumption, best_value = carry
        better = value > best_value
        return (
            jnp.where(better, consumption, best_consumption),
            jnp.where(better, value, best_value),
        )

    def terminal_solve(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        safe_upper = jnp.maximum(upper, lower)

        def bisect(_, bracket):
            left, right = bracket
            midpoint = 0.5 * (left + right)
            next_assets = jnp.maximum(
                assets_without_consumption - consumption_factor * midpoint,
                asset_grid[0],
            )
            marginal = flow_discount * params.consumption_weight * midpoint ** (
                params.consumption_power - 1.0
            ) - beta * consumption_factor * params.bequest_weight * next_assets ** (
                params.bequest_power - 1.0
            )
            return (
                jnp.where(marginal > 0.0, midpoint, left),
                jnp.where(marginal > 0.0, right, midpoint),
            )

        left, right = jax.lax.fori_loop(0, terminal_iterations, bisect, (lower, safe_upper))
        candidate = 0.5 * (left + right)
        for consumption in (lower, safe_upper, candidate):
            carry = keep_better(carry, consumption, value_at_consumption(consumption))
        return carry

    def piecewise_solve(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        bounded_log_human_capital = jnp.clip(
            next_log_human_capital,
            log_human_capital_grid[0],
            log_human_capital_grid[-1],
        )
        log_index = jnp.clip(
            jnp.searchsorted(log_human_capital_grid, bounded_log_human_capital, side="right") - 1,
            0,
            log_human_capital_grid.size - 2,
        )
        log_weight = (bounded_log_human_capital - log_human_capital_grid[log_index]) / (
            log_human_capital_grid[log_index + 1] - log_human_capital_grid[log_index]
        )

        def scan_interval(index, interval_carry):
            asset_left, asset_right = asset_grid[index], asset_grid[index + 1]
            value_left = (
                continuation_values[index, log_index] * (1.0 - log_weight)
                + continuation_values[index, log_index + 1] * log_weight
            )
            value_right = (
                continuation_values[index + 1, log_index] * (1.0 - log_weight)
                + continuation_values[index + 1, log_index + 1] * log_weight
            )
            slope = (value_right - value_left) / (asset_right - asset_left)
            interval_lower = jnp.maximum(
                lower, (assets_without_consumption - asset_right) / consumption_factor
            )
            interval_upper = jnp.minimum(
                upper, (assets_without_consumption - asset_left) / consumption_factor
            )
            interval_feasible = state_feasible & (interval_lower <= interval_upper)
            safe_upper = jnp.maximum(interval_upper, interval_lower)
            positive_slope = jnp.where(slope > 0.0, slope, 1.0)
            stationary = (
                flow_discount
                * params.consumption_weight
                / (beta * consumption_factor * positive_slope)
            ) ** (1.0 / (1.0 - params.consumption_power))
            stationary = jnp.where(slope > 0.0, stationary, safe_upper)
            stationary = jnp.clip(stationary, interval_lower, safe_upper)
            for consumption in (interval_lower, safe_upper, stationary):
                next_assets = assets_without_consumption - consumption_factor * consumption
                continuation = value_left + slope * (next_assets - asset_left)
                value = (
                    flow_discount * flow_utility(consumption, hours, params) + beta * continuation
                )
                value = jnp.where(interval_feasible & jnp.isfinite(value), value, -jnp.inf)
                interval_carry = keep_better(interval_carry, consumption, value)
            return interval_carry

        return jax.lax.fori_loop(0, asset_grid.size - 1, scan_interval, carry)

    consumption, _ = jax.lax.cond(is_terminal, terminal_solve, piecewise_solve, initial)
    # Reevaluate through the solver's actual interpolation at cell boundaries;
    # this also guarantees that a supplied feasible incumbent cannot worsen.
    return keep_better(initial, consumption, value_at_consumption(consumption))
