from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from blinder_weiss.bellman import BellmanConfig, solve_bellman
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationTargets,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    InitialAtom,
    PopulationResult,
    PopulationStateMoments,
    SyntheticInitialDistribution,
    initial_quadrature,
    sample_initial_population,
    simulate_population,
)
from blinder_weiss.model import benchmark_params
from blinder_weiss.population import CohortMoments


def test_quadrature_exact_continuous_joint_moments_and_atoms():
    law = SyntheticInitialDistribution(
        correlation=-0.3,
        asset_floor_mass=0.2,
        atoms=(InitialAtom(5.0, 1.0, 0.1), InitialAtom(0.001, 1.1, 0.05)),
    )
    nodes = initial_quadrature(law, nodes_per_dimension=8, asset_floor=0.001)
    assert nodes.weights.size == 64 + 8 + 2
    assert np.min(nodes.weights) > 0.0
    assert nodes.weights.sum() == pytest.approx(1.0, abs=2e-16)
    np.testing.assert_allclose(nodes.weights[nodes.component == "asset_floor"].sum(), 0.2)
    np.testing.assert_allclose(nodes.weights[nodes.assets == 0.001].sum(), 0.25)
    interior = nodes.component == "interior"
    weights = nodes.weights[interior] / nodes.weights[interior].sum()
    a, y = nodes.assets[interior], np.log(nodes.human_capital[interior])
    assert a @ weights == pytest.approx(5.0, abs=1e-14)
    assert y @ weights == pytest.approx(0.0, abs=1e-14)
    variance_a = (a - 5.0) ** 2 @ weights
    variance_y = y**2 @ weights
    covariance = ((a - 5.0) * y) @ weights
    assert variance_a == pytest.approx(3.0, abs=1e-13)
    assert variance_y == pytest.approx(0.5**2 / 12.0, abs=1e-14)
    assert covariance / np.sqrt(variance_a * variance_y) == pytest.approx(-0.3, abs=1e-14)
    rerun = initial_quadrature(law, nodes_per_dimension=8, asset_floor=0.001)
    np.testing.assert_array_equal(nodes.weights, rerun.weights)
    np.testing.assert_array_equal(nodes.assets, rerun.assets)


def test_quadrature_refinement_and_zero_components_preserve_shapes():
    law = SyntheticInitialDistribution(asset_floor_mass=0.0, atoms=(InitialAtom(5, 1, 0),))
    previous = None
    for order in [8, 16, 32, 64]:
        nodes = initial_quadrature(law, nodes_per_dimension=order, asset_floor=0.001)
        assert len(nodes.assets) == order**2 + order + 1
        actual = nodes.human_capital @ nodes.weights
        exact = (np.exp(0.25) - np.exp(-0.25)) / 0.5
        assert actual == pytest.approx(exact, abs=1e-14)
        if previous is not None:
            assert actual == pytest.approx(previous, abs=1e-14)
        previous = actual


@pytest.mark.parametrize(
    "changes",
    [
        {"correlation": 0.34},
        {"asset_floor_mass": -0.1},
        {"asset_floor_mass": 1.0},
        {"asset_lower": -1.0},
        {"log_human_capital_lower": 1.0},
        {"asset_upper": np.inf},
        {"log_human_capital_upper": 1000.0},
        {"log_human_capital_lower": -1000.0},
        {"atoms": (InitialAtom(1, 0, 0.1),)},
    ],
)
def test_invalid_initial_laws(changes):
    with pytest.raises(ValueError):
        initial_quadrature(replace(SyntheticInitialDistribution(), **changes), asset_floor=0.001)


@pytest.mark.parametrize("order", [0, 1, 2.5, True])
def test_invalid_quadrature_order(order):
    with pytest.raises(ValueError, match="integer at least two"):
        initial_quadrature(nodes_per_dimension=order, asset_floor=0.001)


def test_iid_sampler_matches_the_same_initial_measure():
    law = SyntheticInitialDistribution()
    nodes = sample_initial_population(law, people=100_000, asset_floor=0.001)
    quadrature = initial_quadrature(law, nodes_per_dimension=16, asset_floor=0.001)
    assert nodes.assets @ nodes.weights == pytest.approx(
        quadrature.assets @ quadrature.weights, abs=0.025
    )
    interior = nodes.component == "interior"
    actual_corr = np.corrcoef(nodes.assets[interior], np.log(nodes.human_capital[interior]))[0, 1]
    assert actual_corr == pytest.approx(law.correlation, abs=0.012)
    assert np.mean(nodes.component == "asset_floor") == pytest.approx(0.05, abs=0.003)
    assert np.mean(nodes.component == "point_atom") == pytest.approx(0.05, abs=0.003)
    again = sample_initial_population(law, people=100_000, asset_floor=0.001)
    np.testing.assert_array_equal(nodes.assets, again.assets)


def fake_population():
    return PopulationResult(
        "quadrature",
        CohortMoments(
            time=np.array([0.0, 1.0, 2.0]),
            hours=np.array([0.2, 0.4, 0.6]),
            participation=np.ones(3),
            training_time=np.zeros(3),
            consumption=np.ones(3),
            earnings=np.ones(3),
            participation_hours_threshold=0.02,
        ),
        PopulationStateMoments(
            time=np.array([0.0, 1.0, 2.0, 3.0]),
            assets=np.array([5.0, 6.0, 7.0, 8.0]),
            human_capital=np.ones(4),
            log_human_capital=np.zeros(4),
            asset_floor_mass=np.zeros(4),
        ),
        {},
        None,
    )


def synthetic_targets():
    return CalibrationTargets(
        (
            AgeMomentTarget(
                "hours",
                np.array([20.5, 21.0]),
                np.array([0.2, 0.2]),
                0.1,
                np.array([1.0, 2.0]),
                MOMENT_UNITS["hours"],
            ),
            AgeMomentTarget(
                "assets", np.array([23.0]), np.array([7.0]), 2.0, 1.0, MOMENT_UNITS["assets"]
            ),
        ),
        20.0,
        0.02,
        "synthetic test; no survey input",
    )


def test_scaled_weighted_loss_with_age_origin_and_terminal_states():
    loss = weighted_age_moment_loss(fake_population(), synthetic_targets())
    np.testing.assert_allclose(loss.predicted["hours"], [0.3, 0.4])
    np.testing.assert_allclose(loss.standardized_residuals["hours"], [1.0, 2.0])
    assert loss.loss == pytest.approx((1.0 + 2.0 * 4.0 + 0.25) / 4.0)
    assert loss.total_weight == 4.0


@pytest.mark.parametrize(
    "change,message",
    [
        ({"ages": np.array([20.0, 23.0])}, "no extrapolation"),
        ({"units": "annual hours"}, "converted to units"),
        ({"scale": 0.0}, "scales must be positive"),
        ({"weights": -1.0}, "weights nonnegative"),
        ({"values": np.array([np.nan, 0.0])}, "must be finite"),
    ],
)
def test_invalid_target_measurement_contract(change, message):
    targets = synthetic_targets()
    bad_profile = replace(targets.profiles[0], **change)
    with pytest.raises(ValueError, match=message):
        weighted_age_moment_loss(fake_population(), replace(targets, profiles=(bad_profile,)))


def test_participation_convention_and_zero_weights_cannot_change_silently():
    targets = synthetic_targets()
    with pytest.raises(ValueError, match="thresholds must match"):
        weighted_age_moment_loss(
            fake_population(), replace(targets, participation_hours_threshold=0.01)
        )
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_age_moment_loss(
            fake_population(), replace(targets, profiles=(replace(targets.profiles[0], weights=0),))
        )


def test_quadrature_rollout_preserves_mass_and_initial_floor_atom():
    solution = solve_bellman(
        benchmark_params(horizon=2.0),
        BellmanConfig(
            periods=2,
            asset_nodes=5,
            human_capital_nodes=5,
            asset_minimum=1e-3,
            asset_maximum=12.0,
            log_human_capital_minimum=-1.0,
            log_human_capital_maximum=1.0,
            hours_nodes=3,
            investment_nodes=3,
            consumption_nodes=5,
            refinement_steps=2,
            control_batch_size=16,
            neighbor_policy_sweeps=0,
            compute_platform="cpu",
        ),
    )
    nodes = initial_quadrature(nodes_per_dimension=2, asset_floor=solution.config.asset_minimum)
    result = simulate_population(solution, nodes, participation_hours_threshold=0.02)
    assert result.backend == "quadrature"
    assert result.state_moments.asset_floor_mass[0] == pytest.approx(0.05, abs=1e-14)
    assert result.diagnostics["maximum_mass_drift"] < 1e-14
    assert result.moments.time.shape == (2,)
    assert result.state_moments.time.shape == (3,)
    np.testing.assert_allclose(
        result.state_moments.assets, result.simulation.assets @ nodes.weights
    )


def test_initial_near_floor_mass_remains_interior():
    from blinder_weiss.continuum import _cached_floor_contacts

    config = BellmanConfig(periods=1, asset_minimum=0.001, compute_platform="cpu")
    params = benchmark_params(horizon=1.0)
    near_floor = config.asset_minimum + 5e-11
    states = np.array([[[near_floor, 0.0]], [[near_floor, 0.0]]])
    controls = np.array([[[1e-9, 0.5, 0.0]]])
    contacts = np.asarray(_cached_floor_contacts(config)(params, states, controls))
    assert not contacts[0, 0]
    assert not contacts[1, 0]
