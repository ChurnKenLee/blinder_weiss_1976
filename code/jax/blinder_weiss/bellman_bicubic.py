"""Coordinate-monotone bicubic Hermite continuation reconstruction.

Derivative constraints follow Carlson and Fritsch (1989), DOI
10.1137/0726013, and the implementation accompanying Homogenized Yarn-Level
Cloth. Only derivatives are limited: nodal values and policies are untouched.
Coordinate monotonicity on monotone cells does not imply that the nonlinear
reconstruction preserves ordering between arbitrary value tables.

Adapted from https://git.ista.ac.at/gsperl/HYLC, with the mixed derivative
bounds from both axes intersected simultaneously.

MIT License

Copyright (c) 2020 Georg Sperl, Rahul Narain, Chris Wojtan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .bellman_interpolation import _hermite, _left_index, pchip_slopes


def _constraint_differences(values: Array) -> Array:
    """Classify roundoff-sized edge changes as flat only inside the limiter.

    An exact sign test can interpret opposite one-ULP perturbations of a flat
    edge as a real change of monotonic sense and disable a cell's mixed
    derivative constraints. Large initial twists may then survive despite
    machine-scale differences in the input values. Use an endpoint-local
    floating-point scale consistently for every derivative constraint. There
    is no absolute tolerance floor: small-magnitude value tables retain their
    relative shape. The actual nodal values are never modified.
    """

    differences = jnp.diff(values, axis=0)
    scale = jnp.maximum(jnp.abs(values[:-1]), jnp.abs(values[1:]))
    tolerance = 32.0 * jnp.finfo(values.dtype).eps * scale
    return jnp.where(jnp.abs(differences) <= tolerance, 0.0, differences)


def _limit_edge_slopes(grid: Array, values: Array, slopes: Array) -> Array:
    secants = _constraint_differences(values) / jnp.diff(grid)[:, None]
    edge_lower = jnp.minimum(0.0, 3.0 * secants)
    edge_upper = jnp.maximum(0.0, 3.0 * secants)
    lower = jnp.maximum(
        jnp.pad(edge_lower, ((0, 1), (0, 0)), constant_values=-jnp.inf),
        jnp.pad(edge_lower, ((1, 0), (0, 0)), constant_values=-jnp.inf),
    )
    upper = jnp.minimum(
        jnp.pad(edge_upper, ((0, 1), (0, 0)), constant_values=jnp.inf),
        jnp.pad(edge_upper, ((1, 0), (0, 0)), constant_values=jnp.inf),
    )
    return jnp.clip(slopes, lower, upper)


def _cross_difference_bounds(
    axis_grid: Array, other_grid: Array, values: Array, axis_slopes: Array
) -> tuple[Array, Array]:
    """Bounds on neighboring derivatives in the orthogonal direction."""

    differences = _constraint_differences(values)
    farther_slope = jnp.where(
        jnp.abs(axis_slopes[:-1]) > jnp.abs(axis_slopes[1:]),
        axis_slopes[:-1],
        axis_slopes[1:],
    )
    allowance = 3.0 * differences - jnp.diff(axis_grid)[:, None] * farther_slope
    other_widths = jnp.diff(other_grid)
    previous_width = jnp.concatenate((other_widths[:1], other_widths))
    next_width = jnp.concatenate((other_widths, other_widths[-1:]))
    signed_infinity = jnp.where(differences < 0.0, -jnp.inf, jnp.inf)
    raw_lower = jnp.where(
        jnp.arange(other_grid.size)[None, :] < other_grid.size - 1,
        -allowance / next_width[None, :],
        -signed_infinity,
    )
    raw_upper = jnp.where(
        jnp.arange(other_grid.size)[None, :] > 0,
        allowance / previous_width[None, :],
        signed_infinity,
    )
    return jnp.minimum(raw_lower, raw_upper), jnp.maximum(raw_lower, raw_upper)


def _project_pair(lower: Array, upper: Array, first: Array, second: Array) -> tuple[Array, Array]:
    """Project a difference into its interval without increasing magnitudes."""

    above = second - first > upper
    total = first + second
    midpoint = total / 2.0
    new_first = jnp.where(
        total > upper,
        jnp.maximum(first, 0.0),
        jnp.where(total < -upper, jnp.minimum(second, 0.0) - upper, midpoint - upper / 2.0),
    )
    new_second = jnp.where(
        total > upper,
        jnp.maximum(first, 0.0) + upper,
        jnp.where(total < -upper, jnp.minimum(second, 0.0), midpoint + upper / 2.0),
    )
    first = jnp.where(above, new_first, first)
    second = jnp.where(above, new_second, second)
    below = second - first < lower
    total = first + second
    midpoint = total / 2.0
    new_first = jnp.where(
        total > -lower,
        jnp.maximum(second, 0.0) - lower,
        jnp.where(total < lower, jnp.minimum(first, 0.0), midpoint - lower / 2.0),
    )
    new_second = jnp.where(
        total > -lower,
        jnp.maximum(second, 0.0),
        jnp.where(total < lower, jnp.minimum(first, 0.0) + lower, midpoint + lower / 2.0),
    )
    return jnp.where(below, new_first, first), jnp.where(below, new_second, second)


def _limit_cross_differences(lower: Array, upper: Array, slopes: Array) -> Array:
    edge_count = slopes.shape[0] - 1

    def forward(index, current):
        first, second = _project_pair(
            lower[index], upper[index], current[index], current[index + 1]
        )
        return current.at[index].set(first).at[index + 1].set(second)

    def backward(index, current):
        return forward(edge_count - index - 1, current)

    result = jax.lax.fori_loop(0, edge_count, forward, slopes)
    return jax.lax.fori_loop(0, edge_count, backward, result)


def _cell_twist_bounds(
    axis_grid: Array,
    other_grid: Array,
    values: Array,
    axis_slopes: Array,
    other_slopes: Array,
) -> tuple[Array, Array]:
    """Mixed derivative intervals, ordered as corners 00, 10, 01, 11.

    On each edge parallel to the other coordinate, the axis derivative is
    itself a cubic Hermite curve. These bounds make the Bernstein coefficients
    of that curve and of ``3 * axis_secant - axis_derivative`` nonnegative.
    Consequently, at every point along the edge, the two endpoint derivatives
    in the axis direction lie between zero and three times the local secant.
    The resulting axis-direction cubic is monotone throughout the cell. For
    decreasing cells the same argument applies after reversing signs. Taking
    the intersection for both coordinate directions establishes both shape
    conditions while retaining shared knot derivatives and C1 continuity.
    """

    width = jnp.diff(axis_grid)[:, None]
    other_width = jnp.diff(other_grid)[None, :]
    differences = _constraint_differences(values)
    low_difference = differences[:, :-1]
    high_difference = differences[:, 1:]
    low_cross = (other_slopes[1:, :-1] - other_slopes[:-1, :-1]) / width
    high_cross = (other_slopes[1:, 1:] - other_slopes[:-1, 1:]) / width
    low_term = low_cross + 3.0 * low_difference / (width * other_width)
    high_term = high_cross - 3.0 * high_difference / (width * other_width)
    left_low = axis_slopes[:-1, :-1] / other_width
    right_low = axis_slopes[1:, :-1] / other_width
    left_high = axis_slopes[:-1, 1:] / other_width
    right_high = axis_slopes[1:, 1:] / other_width
    first = 3.0 * jnp.stack((-left_low, -right_low, high_term + left_high, high_term + right_high))
    second = 3.0 * jnp.stack((low_term - left_low, low_term - right_low, left_high, right_high))
    monotone_cell = ((low_difference >= 0.0) & (high_difference >= 0.0)) | (
        (low_difference <= 0.0) & (high_difference <= 0.0)
    )
    return (
        jnp.where(monotone_cell[None, ...], jnp.minimum(first, second), -jnp.inf),
        jnp.where(monotone_cell[None, ...], jnp.maximum(first, second), jnp.inf),
    )


def _twist_bounds(
    values: Array, asset_grid: Array, log_grid: Array, asset_slopes: Array, log_slopes: Array
) -> tuple[Array, Array]:
    x_lower, x_upper = _cell_twist_bounds(asset_grid, log_grid, values, asset_slopes, log_slopes)
    y_lower, y_upper = _cell_twist_bounds(
        log_grid, asset_grid, values.T, log_slopes.T, asset_slopes.T
    )
    y_lower = jnp.transpose(y_lower, (0, 2, 1))[jnp.asarray([0, 2, 1, 3])]
    y_upper = jnp.transpose(y_upper, (0, 2, 1))[jnp.asarray([0, 2, 1, 3])]
    cell_lower = jnp.maximum(x_lower, y_lower)
    cell_upper = jnp.minimum(x_upper, y_upper)
    lower = jnp.full_like(values, -jnp.inf)
    upper = jnp.full_like(values, jnp.inf)
    for corner, (asset_offset, log_offset) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        padding = ((asset_offset, 1 - asset_offset), (log_offset, 1 - log_offset))
        lower = jnp.maximum(lower, jnp.pad(cell_lower[corner], padding, constant_values=-jnp.inf))
        upper = jnp.minimum(upper, jnp.pad(cell_upper[corner], padding, constant_values=jnp.inf))
    return lower, upper


@jax.jit
def prepare_monotone_bicubic(
    values: Array, asset_grid: Array, log_human_capital_grid: Array
) -> Array:
    """Prepare ``[V, V_A, V_z, V_Az]`` once per Bellman age.

    Grids must be strictly increasing with at least two nodes per axis. Input
    values are reproduced exactly, including any existing nodal violation of
    monotonicity. The limiter guarantees monotonicity, up to local floating-
    point noise in the retained nodal values, within cells whose two parallel
    edges have the same monotonic sense. Machine-scale edge differences are
    treated as flat for constraints only. Derivative-bound failures
    larger than roundoff yield NaNs rather than an uncertified interpolant;
    ``bicubic_constraint_violations`` exposes every class of inequality.
    """

    values = jnp.asarray(values)
    asset_grid = jnp.asarray(asset_grid)
    log_grid = jnp.asarray(log_human_capital_grid)
    asset_slopes = _limit_edge_slopes(asset_grid, values, pchip_slopes(asset_grid, values))
    log_slopes = _limit_edge_slopes(log_grid, values.T, pchip_slopes(log_grid, values.T)).T
    initial_twists = 0.5 * (
        pchip_slopes(log_grid, asset_slopes, axis=1) + pchip_slopes(asset_grid, log_slopes)
    )
    x_lower, x_upper = _cross_difference_bounds(asset_grid, log_grid, values, asset_slopes)
    y_lower, y_upper = _cross_difference_bounds(log_grid, asset_grid, values.T, log_slopes.T)
    # Both operations use the same starting derivative tables. Each operation
    # only shrinks magnitudes, so the other axis's difference bounds relax.
    limited_log = _limit_cross_differences(x_lower, x_upper, log_slopes)
    limited_asset = _limit_cross_differences(y_lower, y_upper, asset_slopes.T).T
    lower, upper = _twist_bounds(values, asset_grid, log_grid, limited_asset, limited_log)
    finite_bound_size = jnp.maximum(
        jnp.where(jnp.isfinite(lower), jnp.abs(lower), 0.0),
        jnp.where(jnp.isfinite(upper), jnp.abs(upper), 0.0),
    )
    tolerance = 64.0 * jnp.finfo(values.dtype).eps * jnp.maximum(1.0, finite_bound_size)
    crossed = lower > upper
    twists = jnp.where(crossed, 0.5 * (lower + upper), jnp.clip(initial_twists, lower, upper))
    twists = jnp.where(lower > upper + tolerance, jnp.nan, twists)
    return jnp.stack((values, limited_asset, limited_log, twists))


def bicubic_constraint_violations(
    prepared: Array, asset_grid: Array, log_human_capital_grid: Array
) -> dict[str, Array]:
    """Maximum algebraic residuals; ``value_monotonicity`` checks inputs too."""

    values, asset_slopes, log_slopes, twists = prepared
    log_grid = log_human_capital_grid
    edge_violation = jnp.maximum(
        jnp.max(jnp.abs(asset_slopes - _limit_edge_slopes(asset_grid, values, asset_slopes))),
        jnp.max(jnp.abs(log_slopes.T - _limit_edge_slopes(log_grid, values.T, log_slopes.T))),
    )
    x_lower, x_upper = _cross_difference_bounds(asset_grid, log_grid, values, asset_slopes)
    y_lower, y_upper = _cross_difference_bounds(log_grid, asset_grid, values.T, log_slopes.T)
    cross_violation = jnp.maximum(
        jnp.max(
            jnp.maximum(
                x_lower - jnp.diff(log_slopes, axis=0), jnp.diff(log_slopes, axis=0) - x_upper
            )
        ),
        jnp.max(
            jnp.maximum(
                y_lower - jnp.diff(asset_slopes.T, axis=0),
                jnp.diff(asset_slopes.T, axis=0) - y_upper,
            )
        ),
    )
    lower, upper = _twist_bounds(values, asset_grid, log_grid, asset_slopes, log_slopes)
    return {
        "edge_slope": edge_violation,
        "cross_slope_difference": jnp.maximum(0.0, cross_violation),
        "mixed_derivative": jnp.maximum(0.0, jnp.max(jnp.maximum(lower - twists, twists - upper))),
        "mixed_bound_intersection": jnp.maximum(0.0, jnp.max(lower - upper)),
        "value_monotonicity": jnp.maximum(
            0.0, jnp.maximum(-jnp.min(jnp.diff(values, axis=0)), -jnp.min(jnp.diff(values, axis=1)))
        ),
    }


def monotone_bicubic_interpolate(
    prepared: Array,
    asset_grid: Array,
    log_human_capital_grid: Array,
    assets: ArrayLike,
    log_human_capital: ArrayLike,
    *,
    asset_grid_curvature: float | None = None,
    uniform_log_grid: bool = False,
) -> Array:
    """Evaluate prepared bicubic Hermite, clipping queries to the state domain."""

    values, asset_slopes, log_slopes, twists = prepared
    log_grid = log_human_capital_grid
    asset_query, log_query = jnp.broadcast_arrays(
        jnp.clip(assets, asset_grid[0], asset_grid[-1]),
        jnp.clip(log_human_capital, log_grid[0], log_grid[-1]),
    )
    i = _left_index(asset_grid, asset_query, asset_grid_curvature)
    j = _left_index(log_grid, log_query, 1.0 if uniform_log_grid else None)
    asset_width = asset_grid[i + 1] - asset_grid[i]
    log_width = log_grid[j + 1] - log_grid[j]
    asset_weight = (asset_query - asset_grid[i]) / asset_width
    log_weight = (log_query - log_grid[j]) / log_width

    def along_assets(table, derivative_table, log_offset):
        return _hermite(
            table[i, j + log_offset],
            table[i + 1, j + log_offset],
            derivative_table[i, j + log_offset],
            derivative_table[i + 1, j + log_offset],
            asset_width,
            asset_weight,
        )

    return _hermite(
        along_assets(values, asset_slopes, 0),
        along_assets(values, asset_slopes, 1),
        along_assets(log_slopes, twists, 0),
        along_assets(log_slopes, twists, 1),
        log_width,
        log_weight,
    )
