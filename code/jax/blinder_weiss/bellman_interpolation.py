"""Experimental tensor PCHIP continuation interpolation implemented in JAX.

Interpolation first follows assets at every relevant log-human-capital row,
then log human capital. One-dimensional PCHIP preserves the shape of its input
data; that property does not imply coordinate monotonicity or an order-
preserving Bellman operator for this sequential two-dimensional interpolant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _interior_slope(
    left_width: Array, right_width: Array, left_secant: Array, right_secant: Array
) -> Array:
    same_sign = (jnp.sign(left_secant) == jnp.sign(right_secant)) & (left_secant != 0.0)
    safe_left = jnp.where(same_sign, left_secant, 1.0)
    safe_right = jnp.where(same_sign, right_secant, 1.0)
    first_weight = 2.0 * right_width + left_width
    second_weight = right_width + 2.0 * left_width
    harmonic = (first_weight + second_weight) / (
        first_weight / safe_left + second_weight / safe_right
    )
    return jnp.where(same_sign, harmonic, 0.0)


def _endpoint_slope(
    edge_width: Array, next_width: Array, edge_secant: Array, next_secant: Array
) -> Array:
    slope = ((2.0 * edge_width + next_width) * edge_secant - edge_width * next_secant) / (
        edge_width + next_width
    )
    wrong_sign = jnp.sign(slope) != jnp.sign(edge_secant)
    limited = (jnp.sign(edge_secant) != jnp.sign(next_secant)) & (
        jnp.abs(slope) > 3.0 * jnp.abs(edge_secant)
    )
    return jnp.where(wrong_sign, 0.0, jnp.where(limited, 3.0 * edge_secant, slope))


def pchip_slopes(grid: Array, values: Array, *, axis: int = 0) -> Array:
    """Return PCHIP knot derivatives on a nonuniform grid along one axis.

    The grid must be strictly increasing and contain at least two nodes.
    Zero and oppositely signed secants have zero derivative. Safe denominators
    also keep autodiff finite when one of the original secants vanishes.
    """

    grid = jnp.asarray(grid)
    values = jnp.asarray(values)
    ordered_values = jnp.moveaxis(values, axis, 0)
    widths = jnp.diff(grid).reshape((-1,) + (1,) * (values.ndim - 1))
    secants = jnp.diff(ordered_values, axis=0) / widths
    if grid.size == 2:
        derivatives = jnp.concatenate((secants, secants), axis=0)
    else:
        interior = _interior_slope(widths[:-1], widths[1:], secants[:-1], secants[1:])
        first = _endpoint_slope(widths[0], widths[1], secants[0], secants[1])
        last = _endpoint_slope(widths[-1], widths[-2], secants[-1], secants[-2])
        derivatives = jnp.concatenate((first[None, ...], interior, last[None, ...]), axis=0)
    return jnp.moveaxis(derivatives, 0, axis)


def _hermite(
    left_value: Array,
    right_value: Array,
    left_slope: Array,
    right_slope: Array,
    width: Array,
    weight: Array,
) -> Array:
    # This form reproduces constant data exactly and needs fewer operations
    # than evaluating the four Hermite basis functions separately.
    change = right_value - left_value
    left_tangent = width * left_slope
    right_tangent = width * right_slope
    quadratic = 3.0 * change - 2.0 * left_tangent - right_tangent
    cubic = left_tangent + right_tangent - 2.0 * change
    return left_value + weight * (left_tangent + weight * (quadratic + weight * cubic))


def _left_index(grid: Array, query: Array, curvature: float | None) -> Array:
    if curvature is None:
        return jnp.clip(jnp.searchsorted(grid, query, side="right") - 1, 0, grid.size - 2)
    # The optional inverse-grid shortcut requires grid[i] = grid[0] +
    # (grid[-1]-grid[0])*(i/(n-1))**curvature. Cell comparisons repair rounding
    # at knots; the indexing operation has no mathematical derivative.
    normalized = jax.lax.stop_gradient((query - grid[0]) / (grid[-1] - grid[0]))
    position = (grid.size - 1) * jnp.maximum(normalized, 0.0) ** (1.0 / curvature)
    index = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, grid.size - 2)
    index = jnp.where(query < grid[index], jnp.maximum(index - 1, 0), index)
    return jnp.where(query >= grid[index + 1], jnp.minimum(index + 1, grid.size - 2), index)


def tensor_pchip_interpolate(
    values: Array,
    asset_grid: Array,
    log_human_capital_grid: Array,
    assets: ArrayLike,
    log_human_capital: ArrayLike,
    *,
    asset_derivatives: Array | None = None,
    asset_grid_curvature: float | None = None,
    uniform_log_grid: bool = False,
) -> Array:
    """Evaluate sequential asset-then-log-human-capital PCHIP in a JAX batch.

    Precompute ``pchip_slopes(asset_grid, values)`` once per Bellman age and
    pass ``asset_derivatives`` to avoid rebuilding the full derivative table
    at each optimizer evaluation. A query then gathers two asset rows and
    four log-human-capital columns from values and derivatives. Queries are
    clipped to the represented domain, matching the bilinear evaluator.

    ``asset_grid_curvature`` and ``uniform_log_grid`` enable direct cell
    lookup only when the grids have the stated forms. Otherwise binary search
    supports arbitrary nonuniform grids. Tensor PCHIP is smooth across grid
    knots in strictly monotone neighborhoods; slope-limiter branch changes can
    create derivative kinks in the orthogonal coordinate or in parameters.
    """

    values = jnp.asarray(values)
    asset_grid = jnp.asarray(asset_grid)
    log_grid = jnp.asarray(log_human_capital_grid)
    asset_query, log_query = jnp.broadcast_arrays(
        jnp.clip(assets, asset_grid[0], asset_grid[-1]),
        jnp.clip(log_human_capital, log_grid[0], log_grid[-1]),
    )
    derivatives = (
        pchip_slopes(asset_grid, values) if asset_derivatives is None else asset_derivatives
    )
    asset_index = _left_index(asset_grid, asset_query, asset_grid_curvature)
    log_index = _left_index(log_grid, log_query, 1.0 if uniform_log_grid else None)
    log_stencil = jnp.clip(log_index[..., None] + jnp.arange(-1, 3), 0, log_grid.size - 1)
    asset_width = asset_grid[asset_index + 1] - asset_grid[asset_index]
    asset_weight = (asset_query - asset_grid[asset_index]) / asset_width
    along_assets = _hermite(
        values[asset_index[..., None], log_stencil],
        values[asset_index[..., None] + 1, log_stencil],
        derivatives[asset_index[..., None], log_stencil],
        derivatives[asset_index[..., None] + 1, log_stencil],
        asset_width[..., None],
        asset_weight[..., None],
    )
    left_value, right_value = along_assets[..., 1], along_assets[..., 2]
    log_width = log_grid[log_index + 1] - log_grid[log_index]
    middle_secant = (right_value - left_value) / log_width
    if log_grid.size == 2:
        left_slope = right_slope = middle_secant
    else:
        previous_width = log_grid[log_stencil[..., 1]] - log_grid[log_stencil[..., 0]]
        next_width = log_grid[log_stencil[..., 3]] - log_grid[log_stencil[..., 2]]
        # Boundary stencils repeat a node. The corresponding secant is unused
        # by the selected endpoint formula, but must remain safe for autodiff.
        previous_secant = (left_value - along_assets[..., 0]) / jnp.where(
            previous_width > 0.0, previous_width, 1.0
        )
        next_secant = (along_assets[..., 3] - right_value) / jnp.where(
            next_width > 0.0, next_width, 1.0
        )
        left_slope = jnp.where(
            log_index == 0,
            _endpoint_slope(log_width, next_width, middle_secant, next_secant),
            _interior_slope(previous_width, log_width, previous_secant, middle_secant),
        )
        right_slope = jnp.where(
            log_index == log_grid.size - 2,
            _endpoint_slope(log_width, previous_width, middle_secant, previous_secant),
            _interior_slope(log_width, next_width, middle_secant, next_secant),
        )
    log_weight = (log_query - log_grid[log_index]) / log_width
    return _hermite(left_value, right_value, left_slope, right_slope, log_width, log_weight)
