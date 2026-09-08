"""Shared floating-point classification of an active endpoint asset constraint."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike

# Covers cancellation and a few elementary-function/multiply/divide operations
# in the exact transition and the endpoint consumption bound. This is a
# float64 arithmetic budget, not an economic near-boundary band.
_FLOOR_CONTACT_ULPS = 32.0


def endpoint_floor_contact(
    next_assets: ArrayLike,
    asset_floor: ArrayLike,
    consumption: ArrayLike,
    endpoint_capacity: ArrayLike,
    consumption_factor: ArrayLike,
) -> Array:
    """Identify endpoint binding within the exact transition's arithmetic error.

    The asset transition subtracts ``consumption_factor * consumption`` from
    gross resources. Endpoint capacity represents those resources net of the
    floor, so their cancellation scale is available without recomputing policy
    dynamics. Both the asset residual and the consumption-capacity slack must
    fit 32 float64 epsilons times that scale. A merely nearby interior state
    with a slack control is not a contact. Initial atoms must still be tagged
    by exact initial-state equality, not this endpoint-control criterion.

    Comparing ``consumption >= endpoint_capacity`` bit for bit is unreliable:
    the minimum over within-period checkpoints and a separately compiled
    endpoint bound can differ by several ulps at the same binding constraint.
    Missing that contact can repeatedly move face mass to the first interior
    grid node. This classifier must be shared by quadrature and transport.
    """
    next_assets = jnp.asarray(next_assets)
    floor = jnp.asarray(asset_floor)
    consumption = jnp.asarray(consumption)
    capacity = jnp.asarray(endpoint_capacity)
    factor = jnp.asarray(consumption_factor)
    error_budget = _FLOOR_CONTACT_ULPS * jnp.finfo(jnp.float64).eps * (
        jnp.abs(floor) + jnp.abs(factor * capacity) + jnp.abs(factor * consumption)
    )
    return (
        jnp.isfinite(next_assets) & jnp.isfinite(consumption) & jnp.isfinite(capacity)
        & (factor > 0)
        & (jnp.abs(next_assets - floor) <= error_budget)
        & (jnp.abs((capacity - consumption) * factor) <= error_budget)
    )
