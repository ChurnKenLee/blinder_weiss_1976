"""Generated-grid lookup must preserve the reference interpolation exactly."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss.bellman import BellmanConfig, _interpolate_jax, bellman_state_grids


@pytest.mark.parametrize("curvature", [0.7, 1.0, 2.0, 3.5])
def test_direct_grid_indexing_matches_searchsorted_at_and_around_nodes(curvature):
    cfg = replace(
        BellmanConfig(), asset_nodes=41, human_capital_nodes=29, asset_grid_curvature=curvature
    )
    a, y = bellman_state_grids(cfg)
    rng = np.random.default_rng(182)
    aq = np.concatenate(
        [
            a,
            np.nextafter(a, -np.inf),
            np.nextafter(a, np.inf),
            rng.uniform(a[0] - 1, a[-1] + 1, 1000),
        ]
    )
    yq = np.concatenate(
        [
            y,
            np.nextafter(y, -np.inf),
            np.nextafter(y, np.inf),
            rng.uniform(y[0] - 1, y[-1] + 1, 1000),
        ]
    )
    # Include all pairs of exact state nodes and values on both sides of them.
    aq, yq = np.meshgrid(aq, yq, indexing="ij")
    values = jnp.asarray(rng.normal(size=(a.size, y.size)))
    args = (values, jnp.asarray(a), jnp.asarray(y), jnp.asarray(aq), jnp.asarray(yq))
    expected = jax.jit(_interpolate_jax)(*args)
    actual = jax.jit(lambda *xs: _interpolate_jax(*xs, asset_grid_curvature=curvature))(*args)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_direct_indexing_preserves_autodiff_inside_cells():
    a, y = bellman_state_grids(BellmanConfig())
    values = jnp.log1p(jnp.asarray(a[:, None])) + jnp.sin(jnp.asarray(y[None, :]))
    state = jnp.asarray([2.17, 0.123])

    def interpolate(z, curvature):
        return _interpolate_jax(
            values, jnp.asarray(a), jnp.asarray(y), z[0], z[1], asset_grid_curvature=curvature
        )

    np.testing.assert_allclose(
        jax.grad(lambda z: interpolate(z, 2.0))(state),
        jax.grad(lambda z: interpolate(z, None))(state),
        rtol=1e-13,
        atol=1e-13,
    )
