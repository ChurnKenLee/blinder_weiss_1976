"""The optimizer extension must preserve the feasible economic technology."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss.model import effective_earnings_share


def _original_earnings(control):
    hours, training = control
    slope = jnp.sqrt(1.25) - 0.5
    safe_hours = jnp.where(hours > 0.0, hours, 1.0)
    return hours - slope * training - jnp.square(slope) * jnp.square(training) / safe_hours


def _extended_earnings(control):
    return effective_earnings_share(control[0], control[1])


def test_feasible_values_and_gradients_match_original_exactly() -> None:
    hours = np.geomspace(1e-18, 1.0, 25)
    fractions = np.asarray([0.0, 0.1, 0.4, 0.8, 1.0])
    h, x = np.meshgrid(hours, fractions)
    controls = np.vstack((np.column_stack((h.ravel(), (h * x).ravel())), [[0.0, 0.0]]))
    original = jax.vmap(jax.value_and_grad(_original_earnings))(jnp.asarray(controls))
    extended = jax.vmap(jax.value_and_grad(_extended_earnings))(jnp.asarray(controls))
    np.testing.assert_array_equal(extended[0], original[0])
    np.testing.assert_array_equal(extended[1], original[1])


@pytest.mark.parametrize("hours", [1e-12, 0.1, 0.8])
def test_positive_training_boundary_has_matching_one_sided_gradients(hours: float) -> None:
    slope = np.sqrt(1.25) - 0.5
    expected = np.asarray([1.0 + slope**2, -slope - 2.0 * slope**2])
    controls = jnp.asarray(
        [
            [hours, hours * (1.0 - 1e-9)],
            [hours, hours],
            [hours, hours * (1.0 + 1e-9)],
        ]
    )
    gradient = jax.jit(jax.vmap(jax.grad(_extended_earnings)))(controls)
    np.testing.assert_allclose(gradient, np.broadcast_to(expected, (3, 2)), atol=2e-9)


@pytest.mark.parametrize(
    ("hours", "training"),
    [(1e-18, 1e-16), (1.3695622208931231e-18, 1.232035497509927e-16), (0.0, 1e-16), (1e-30, 1e-9)],
)
def test_near_retirement_infeasible_trials_have_bounded_finite_gradients(
    hours: float, training: float
) -> None:
    slope = np.sqrt(1.25) - 0.5
    value, gradient = jax.jit(jax.value_and_grad(_extended_earnings))(
        jnp.asarray([hours, training])
    )
    expected_value = hours - slope * training - slope**2 * (2.0 * training - hours)
    np.testing.assert_allclose(value, expected_value, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(gradient, [1.0 + slope**2, -slope - 2.0 * slope**2], atol=1e-14)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) < 2.0


def test_retirement_origin_preserves_autodiff_convention() -> None:
    value, gradient = jax.jit(jax.value_and_grad(_extended_earnings))(jnp.zeros(2))
    assert float(value) == 0.0
    np.testing.assert_allclose(gradient, [1.0, -(np.sqrt(1.25) - 0.5)], atol=1e-14)


def test_extension_preserves_concavity_across_training_boundary() -> None:
    # Includes h=0,q>0 trial points as well as the feasible cone. Concavity
    # follows analytically from the convex quadratic-then-linear perspective.
    random = np.random.default_rng(47)
    left = random.uniform(0.0, 1.0, (200, 2))
    right = random.uniform(0.0, 1.0, (200, 2))
    left[:10, 0] = 0.0
    mixture_weight = random.uniform(0.0, 1.0, (200, 1))
    mixed = mixture_weight * left + (1.0 - mixture_weight) * right
    evaluate = jax.jit(jax.vmap(_extended_earnings))
    lower_bound = mixture_weight[:, 0] * evaluate(left) + (1.0 - mixture_weight[:, 0]) * evaluate(
        right
    )
    assert np.min(np.asarray(evaluate(mixed) - lower_bound)) >= -1e-14
