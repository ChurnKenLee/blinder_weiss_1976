"""Check fixed-target scales and native-age probability comparisons."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationTargets,
    MomentLoss,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from validate_population_quadrature import (  # noqa: E402
    compare_native_profiles,
    compare_target_predictions,
    restore_problem,
)


def test_restore_preserves_explicit_atom_law_and_band_measurement():
    report = {
        "initial_law": {
            "asset_lower": 2.0,
            "asset_upper": 8.0,
            "log_human_capital_lower": -0.25,
            "log_human_capital_upper": 0.25,
            "correlation": -0.2,
            "asset_floor_mass": 0.1,
            "atoms": [{"assets": 5.0, "human_capital": 1.0, "mass": 0.03}],
        },
        "targets": {
            "profiles": [
                {
                    "moment": "near_asset_floor_mass",
                    "ages": [20.0, 21.0],
                    "values": [0.1, 0.2],
                    "scale": [0.02, 0.03],
                    "weights": [1.0, 0.0],
                    "units": MOMENT_UNITS["near_asset_floor_mass"],
                }
            ],
            "age_origin": 20.0,
            "participation_hours_threshold": 0.04,
            "source": "fixed synthetic smoke target",
            "near_asset_floor_width": 0.015,
        },
    }
    law, targets = restore_problem(report)
    assert law.correlation == -0.2 and law.asset_floor_mass == 0.1
    assert len(law.atoms) == 1 and law.atoms[0].mass == 0.03
    assert targets.near_asset_floor_width == 0.015
    assert targets.participation_hours_threshold == 0.04
    np.testing.assert_array_equal(targets.profiles[0].ages, [20, 21])
    np.testing.assert_array_equal(targets.profiles[0].values, [0.1, 0.2])
    del report["initial_law"]["atoms"]
    with pytest.raises(KeyError):
        restore_problem(report)


def test_scaled_target_comparison_excludes_zero_weights_without_retargeting_loss():
    targets = CalibrationTargets(
        (
            AgeMomentTarget(
                "hours",
                np.array([20.0, 21.0]),
                np.zeros(2),
                np.array([0.02, 0.04]),
                np.array([2.0, 0.0]),
                MOMENT_UNITS["hours"],
            ),
        ),
        20.0,
        0.02,
        "fixed synthetic target",
    )
    actual = MomentLoss(3.0, {"hours": np.array([0.03, 1000.0])}, {}, {}, 2.0)
    reference = MomentLoss(0.7, {"hours": np.array([0.01, 0.0])}, {}, {}, 2.0)
    result = compare_target_predictions(actual, reference, targets)
    assert result["maximum_standardized_moment_difference"] == pytest.approx(1.0)
    assert result["weighted_standardized_moment_rms"] == pytest.approx(1.0)
    assert result["absolute_loss_difference_against_fixed_target"] == pytest.approx(2.3)
    assert result["moments"]["hours"]["maximum_standardized_difference_age"] == 20.0
    assert result["moments"]["hours"]["positive_weight_observations"] == 1


def population(floor_mass):
    return SimpleNamespace(
        moments=SimpleNamespace(
            time=np.array([0.0, 0.5]),
            assets=np.ones(2),
            human_capital=np.ones(2),
            hours=np.zeros(2),
            participation=np.zeros(2),
            training_time=np.zeros(2),
            consumption=np.ones(2),
            earnings=np.zeros(2),
        ),
        state_moments=SimpleNamespace(
            time=np.array([0.0, 0.5, 1.0]),
            assets=np.ones(3),
            human_capital=np.ones(3),
            log_human_capital=np.zeros(3),
            asset_floor_mass=np.array(floor_mass),
            near_asset_floor_mass=np.ones(3) * 0.2,
        ),
    )


def test_native_floor_mass_comparison_keeps_endpoint_age_and_rejects_changed_mesh():
    actual, reference = population([0.1, 0.2, 0.1]), population([0.1, 0.05, 0.1])
    actual.state_moments.assets[-1] += 0.4
    result = compare_native_profiles(actual, reference)
    assert result["assets"]["maximum_absolute"] == pytest.approx(0.4)
    assert result["assets"]["maximum_difference_model_age"] == 1.0
    floor = result["asset_floor_mass"]
    assert floor["maximum_absolute"] == pytest.approx(0.15)
    assert floor["maximum_difference_model_age"] == 0.5
    assert floor["rmse"] == pytest.approx(0.15 / np.sqrt(3))
    assert result["near_asset_floor_mass"]["maximum_absolute"] == 0.0
    reference.state_moments.time = np.array([0.0, 0.6, 1.0])
    with pytest.raises(ValueError, match="ages"):
        compare_native_profiles(actual, reference)
