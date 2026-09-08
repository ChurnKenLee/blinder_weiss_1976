from __future__ import annotations

from dataclasses import replace
from typing import Any

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
        {"asset_floor_mass": 1.1},
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
    law = SyntheticInitialDistribution(atoms=(InitialAtom(5.0, 1.0, 0.05),))
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


@pytest.fixture(scope="module")
def small_solution():
    return solve_bellman(
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


def test_quadrature_rollout_preserves_mass_and_initial_floor_atom(small_solution):
    solution = small_solution
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


def test_failure_diagnostics_can_be_saved_as_strict_json():
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "tools/benchmark_continuum.py"
    spec = importlib.util.spec_from_file_location("continuum_benchmark", script)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    jsonable = benchmark.jsonable

    failed = {
        "completed": False,
        "error": "nonfinite policy caused an invalid destination",
        "diagnostics": {
            "array": np.array([0.0, np.nan, np.inf, -np.inf]),
            "scalar": np.float64(np.nan),
            "nested": (float("inf"),),
        },
    }
    encoded = json.dumps(jsonable(failed), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["completed"] is False
    assert decoded["error"] == failed["error"]
    assert decoded["diagnostics"]["array"] == [0.0, None, None, None]
    assert decoded["diagnostics"]["scalar"] is None
    assert decoded["diagnostics"]["nested"] == [None]


def test_default_has_no_interior_atom_and_stress_case_is_explicit():
    from blinder_weiss.continuum import synthetic_initial_scenarios

    cases = synthetic_initial_scenarios()
    assert cases["baseline"].atoms == ()
    assert cases["point_atom_stress"].atoms == (InitialAtom(5.0, 1.0, 0.05),)
    baseline = initial_quadrature(cases["baseline"], nodes_per_dimension=4, asset_floor=0.001)
    stress = initial_quadrature(
        cases["point_atom_stress"], nodes_per_dimension=4, asset_floor=0.001
    )
    assert np.sum(baseline.component == "point_atom") == 0
    assert stress.weights[stress.component == "point_atom"].sum() == pytest.approx(0.05)
    assert baseline.initial_law == cases["baseline"]
    assert baseline.quadrature_order == 4


def test_near_floor_band_includes_floor_and_retains_distinct_interior_mass(small_solution):
    floor = small_solution.config.asset_minimum
    law = SyntheticInitialDistribution(
        asset_floor_mass=0.0,
        atoms=(
            InitialAtom(floor, 1.0, 0.2),
            InitialAtom(floor + 0.005, 1.0, 0.3),
            InitialAtom(0.5, 1.0, 0.5),
        ),
    )
    nodes = initial_quadrature(law, nodes_per_dimension=2, asset_floor=floor)
    result = simulate_population(
        small_solution, nodes, participation_hours_threshold=0.02, near_asset_floor_width=0.01
    )
    assert result.state_moments.asset_floor_mass[0] == pytest.approx(0.2)
    assert result.state_moments.near_asset_floor_mass is not None
    assert result.state_moments.near_asset_floor_mass[0] == pytest.approx(0.5)
    assert result.state_moments.near_asset_floor_width == 0.01
    target = CalibrationTargets(
        (
            AgeMomentTarget(
                "near_asset_floor_mass",
                np.array([0.0]),
                np.array([0.5]),
                0.02,
                1.0,
                MOMENT_UNITS["near_asset_floor_mass"],
            ),
        ),
        0.0,
        0.02,
        "synthetic band-definition test",
        near_asset_floor_width=0.01,
    )
    assert weighted_age_moment_loss(result, target).loss == 0.0
    with pytest.raises(ValueError, match="require a finite positive"):
        weighted_age_moment_loss(result, replace(target, near_asset_floor_width=None))
    with pytest.raises(ValueError, match="widths must match"):
        weighted_age_moment_loss(result, replace(target, near_asset_floor_width=0.02))
    with pytest.raises(ValueError, match="near_asset_floor_width"):
        simulate_population(
            small_solution, nodes, participation_hours_threshold=0.02, near_asset_floor_width=-0.01
        )


def test_initial_floor_share_fit_reuses_policy_and_recovers_initial_mean(small_solution):
    from blinder_weiss.calibration import fit_initial_law_calibration

    true_share = 0.08
    expected_assets = 5.0 * (1.0 - true_share) + small_solution.config.asset_minimum * true_share
    targets = CalibrationTargets(
        (
            AgeMomentTarget(
                "assets",
                np.array([0.0]),
                np.array([expected_assets]),
                0.1,
                1.0,
                MOMENT_UNITS["assets"],
            ),
        ),
        0.0,
        0.02,
        "synthetic known initial mean",
    )
    fitted = fit_initial_law_calibration(
        "asset_floor_mass",
        (0.0, 0.15),
        solution=small_solution,
        law=SyntheticInitialDistribution(),
        targets=targets,
        nodes_per_dimension=2,
        parameter_tolerance=1e-6,
    )
    assert fitted.success
    assert fitted.x == pytest.approx(true_share, abs=1e-6)
    assert fitted.best_evaluation.population.simulation.solution is small_solution
    assert all(item.population.simulation.solution is small_solution for item in fitted.evaluations)
    with pytest.raises(ValueError, match="explicit valid atom_index"):
        fit_initial_law_calibration(
            "atom_mass",
            (0.0, 0.1),
            solution=small_solution,
            law=SyntheticInitialDistribution(),
            targets=targets,
            nodes_per_dimension=2,
        )
    with pytest.raises(ValueError, match="correlation"):
        fit_initial_law_calibration(
            "correlation",
            (-0.5, 0.1),
            solution=small_solution,
            law=SyntheticInitialDistribution(),
            targets=targets,
            nodes_per_dimension=2,
        )


def test_acceptance_requires_distinct_resolutions_same_law_and_stable_fit():
    from dataclasses import asdict
    from types import SimpleNamespace

    from blinder_weiss.calibration import (
        CalibrationEvaluation,
        NumericalAcceptanceThresholds,
        compare_calibration_resolutions,
    )

    params = benchmark_params()
    config = BellmanConfig()
    base = fake_population()
    reference = replace(
        base,
        simulation=SimpleNamespace(solution=SimpleNamespace(config=config)),
        diagnostics={
            "initial_law": asdict(SyntheticInitialDistribution()),
            "quadrature_order": 16,
            "maximum_full_period_floor_violation": 0.0,
        },
    )
    coarse = replace(
        reference,
        moments=replace(reference.moments, hours=reference.moments.hours + 0.001),
        diagnostics={**reference.diagnostics, "quadrature_order": 8},
    )
    targets = synthetic_targets()

    def evaluation(population):
        return CalibrationEvaluation(
            params, population, weighted_age_moment_loss(population, targets), 0.0, 0.0, 0.0
        )

    kwargs: dict[str, Any] = dict(
        fitted_parameters=(1.0, 1.001),
        optimizer_success=(True, True),
        thresholds=NumericalAcceptanceThresholds(0.02, 0.03, 0.002),
    )
    passed = compare_calibration_resolutions(
        evaluation(coarse), evaluation(reference), targets, **kwargs
    )
    assert passed["passed"]
    same_setup = compare_calibration_resolutions(
        evaluation(reference), evaluation(reference), targets, **kwargs
    )
    assert not same_setup["passed"]
    assert not same_setup["criteria"]["distinct_numerical_resolutions"]
    wrong_law = replace(
        coarse,
        diagnostics={
            **coarse.diagnostics,
            "initial_law": asdict(SyntheticInitialDistribution(correlation=0)),
        },
    )
    assert not compare_calibration_resolutions(
        evaluation(wrong_law), evaluation(reference), targets, **kwargs
    )["passed"]
    unstable = {**kwargs, "fitted_parameters": (0.9, 1.0)}
    assert not compare_calibration_resolutions(
        evaluation(coarse), evaluation(reference), targets, **unstable
    )["passed"]
    nominal_change = replace(
        reference,
        simulation=SimpleNamespace(
            solution=SimpleNamespace(
                config=replace(config, compute_platform="gpu", path_checkpoints=27)
            )
        ),
    )
    assert not compare_calibration_resolutions(
        evaluation(nominal_change), evaluation(reference), targets, **kwargs
    )["criteria"]["distinct_numerical_resolutions"]
    infeasible = replace(
        coarse,
        diagnostics={**coarse.diagnostics, "maximum_full_period_floor_violation": 3e-5},
    )
    failed = compare_calibration_resolutions(
        evaluation(infeasible), evaluation(reference), targets, **kwargs
    )
    assert not failed["passed"]
    assert not failed["criteria"]["both_full_period_paths_feasible"]
    shifted_floor = replace(
        coarse,
        simulation=SimpleNamespace(
            solution=SimpleNamespace(config=replace(config, asset_minimum=2e-4))
        ),
    )
    assert not compare_calibration_resolutions(
        evaluation(shifted_floor), evaluation(reference), targets, **kwargs
    )["criteria"]["same_explicit_initial_law"]


def test_scalar_fit_keeps_incumbent_and_preserves_search_termination(monkeypatch):
    from types import SimpleNamespace

    import blinder_weiss.calibration as calibration

    def objective(params, *args, **kwargs):
        # A narrow isolated optimum that bounded scalar search does not sample.
        loss = 0.0 if params.leisure_weight == 1.0 else 1.0 + (params.leisure_weight - 1.03) ** 2
        return SimpleNamespace(params=params, objective=SimpleNamespace(loss=loss))

    monkeypatch.setattr(calibration, "evaluate_calibration", objective)
    recorded = []
    fit = calibration.fit_scalar_calibration(
        "leisure_weight",
        (0.9, 1.1),
        params=benchmark_params(),
        config=BellmanConfig(),
        initial_nodes=initial_quadrature(nodes_per_dimension=2, asset_floor=0.001),
        targets=synthetic_targets(),
        max_evaluations=30,
        evaluation_callback=recorded.append,
    )
    assert recorded == fit.evaluations
    assert fit.success
    assert fit.x == 1.0 and fit.fun == 0.0
    assert fit.selection_source == "initial_parameter"
    assert fit.raw_optimizer["fun"] >= 1.0
    assert fit.raw_optimizer["x"] != fit.x
    assert fit.nfev == len(fit.evaluations) <= 30
    limited = calibration.fit_scalar_calibration(
        "leisure_weight",
        (0.9, 1.1),
        params=benchmark_params(),
        config=BellmanConfig(),
        initial_nodes=initial_quadrature(nodes_per_dimension=2, asset_floor=0.001),
        targets=synthetic_targets(),
        max_evaluations=1,
    )
    assert not limited.success
    assert limited.raw_optimizer is None
    assert limited.nfev == 1 and limited.fun == 0.0


@pytest.mark.parametrize("maximum", [1, 2, 3])
@pytest.mark.parametrize("bounds", [(0.9, 1.1), (1.2, 1.4)])
def test_scalar_fit_hard_evaluation_cap_inside_and_outside_bounds(monkeypatch, maximum, bounds):
    from types import SimpleNamespace

    import blinder_weiss.calibration as calibration

    calls = []

    def objective(params, *args, **kwargs):
        calls.append(params.leisure_weight)
        return SimpleNamespace(
            params=params, objective=SimpleNamespace(loss=(params.leisure_weight - 1.04) ** 2)
        )

    monkeypatch.setattr(calibration, "evaluate_calibration", objective)
    fit = calibration.fit_scalar_calibration(
        "leisure_weight",
        bounds,
        params=benchmark_params(),
        config=BellmanConfig(),
        initial_nodes=initial_quadrature(nodes_per_dimension=2, asset_floor=0.001),
        targets=synthetic_targets(),
        max_evaluations=maximum,
    )
    assert fit.nfev == len(calls) == len(fit.evaluations) <= maximum
    assert all(bounds[0] <= value <= bounds[1] for value in calls)
    assert fit.fun == min(item.objective.loss for item in fit.evaluations)
