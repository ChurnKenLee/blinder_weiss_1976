"""Compare population quadrature orders using one saved policy and fixed targets.

No Bellman solve or parameter fit is performed. The highest requested quadrature
order is a numerical comparison reference, not a continuum certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationTargets,
    MomentLoss,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    InitialAtom,
    SyntheticInitialDistribution,
    initial_quadrature,
    simulate_population,
)
from validate_calibration_grid import error_metrics, load_solution

_STATE_MOMENTS = {
    "assets",
    "human_capital",
    "log_human_capital",
    "asset_floor_mass",
    "near_asset_floor_mass",
}


def moment_owner(population, name):
    """State means include the terminal boundary, even when control moments duplicate them."""
    return population.state_moments if name in _STATE_MOMENTS else population.moments


def restore_problem(
    report: dict[str, Any],
) -> tuple[SyntheticInitialDistribution, CalibrationTargets]:
    """Restore every supplied initial-law and target measurement field explicitly."""
    law_data = dict(report["initial_law"])
    law_data["atoms"] = tuple(InitialAtom(**atom) for atom in law_data["atoms"])
    law = SyntheticInitialDistribution(**law_data)
    target_data = dict(report["targets"])
    target_data["profiles"] = tuple(
        AgeMomentTarget(
            **{
                **profile,
                "ages": np.asarray(profile["ages"], dtype=float),
                "values": np.asarray(profile["values"], dtype=float),
            }
        )
        for profile in target_data["profiles"]
    )
    return law, CalibrationTargets(**target_data)


def compare_target_predictions(
    actual: MomentLoss, reference: MomentLoss, targets: CalibrationTargets
) -> dict[str, Any]:
    """Compare fixed-target predictions, honoring observation scales and zero weights."""
    moments, total_square, total_weight = {}, 0.0, 0.0
    for profile in targets.profiles:
        weights = np.broadcast_to(np.asarray(profile.weights), profile.ages.shape)
        selected = weights > 0
        if not selected.any():
            continue
        scale = np.broadcast_to(np.asarray(profile.scale), profile.ages.shape)[selected]
        difference = actual.predicted[profile.moment] - reference.predicted[profile.moment]
        standardized = difference[selected] / scale
        worst = int(np.argmax(np.abs(standardized)))
        moments[profile.moment] = {
            "maximum_absolute_difference": float(np.max(np.abs(difference[selected]))),
            "maximum_standardized_difference": float(np.max(np.abs(standardized))),
            "maximum_standardized_difference_age": float(profile.ages[selected][worst]),
            "weighted_standardized_rms": float(
                np.sqrt(np.average(standardized**2, weights=weights[selected]))
            ),
            "positive_weight_observations": int(selected.sum()),
            "units": profile.units,
        }
        total_square += float(np.sum(weights[selected] * standardized**2))
        total_weight += float(weights[selected].sum())
    if total_weight <= 0:
        raise ValueError("comparison requires positive-weight target observations")
    return {
        "moments": moments,
        "maximum_standardized_moment_difference": max(
            row["maximum_standardized_difference"] for row in moments.values()
        ),
        "weighted_standardized_moment_rms": float(np.sqrt(total_square / total_weight)),
        "signed_loss_difference_against_fixed_target": actual.loss - reference.loss,
        "absolute_loss_difference_against_fixed_target": abs(actual.loss - reference.loss),
    }


def compare_native_profiles(actual, reference) -> dict[str, Any]:
    """Compare every native age directly; state boundary probabilities are not interpolated."""
    result = {}
    for name in MOMENT_UNITS:
        owner = moment_owner(actual, name)
        other = moment_owner(reference, name)
        values, other_values = getattr(owner, name), getattr(other, name)
        if values is None and other_values is None:
            continue
        if values is None or other_values is None or not np.array_equal(owner.time, other.time):
            raise ValueError("native profiles require matching measurement definitions and ages")
        row = error_metrics(np.asarray(values), np.asarray(other_values))
        row["maximum_difference_model_age"] = float(owner.time[row["maximum_index"][0]])
        row["units"] = MOMENT_UNITS[name]
        result[name] = row
    return result


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_report(folder: Path, report, arrays: dict[str, Any]):
    """Write complete files by atomic rename, never partial JSON or NPZ bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "profiles.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    temporary.replace(folder / "profiles.npz")
    temporary = folder / "report.json.tmp"
    temporary.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-folder", type=Path, required=True)
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    orders = sorted(set(args.orders))
    if len(orders) < 2 or orders[0] < 2:
        parser.error("at least two distinct quadrature orders >=2 are required")
    # Read the running stability report exactly once: atomic checkpoints may replace it later.
    stability_bytes = args.stability_report.read_bytes()
    stability = json.loads(stability_bytes)
    law, targets = restore_problem(stability)
    policy_bytes = (args.policy_folder / "report.json").read_bytes()
    policy_report = json.loads(policy_bytes)
    if policy_report["params"] != stability["params"]:
        parser.error("saved policy and fixed-target report must use identical model parameters")
    if law.asset_floor_mass > 0 and (
        policy_report["config"]["asset_minimum"] != stability["config"]["asset_minimum"]
    ):
        parser.error("a fixed initial floor component requires the same numerical floor")
    solution, _ = load_solution(args.policy_folder, args.platform)
    if (args.policy_folder / "report.json").read_bytes() != policy_bytes:
        raise RuntimeError("policy report changed while loading; retry from a stable artifact")
    model_root = Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss"
    report: dict[str, Any] = {
        "purpose": "saved-policy quadrature refinement against fixed calibration targets",
        "status": "running",
        "bellman_solves_performed": 0,
        "parameter_fits_performed": 0,
        "continuum_convergence_certified": False,
        "policy_folder": str(args.policy_folder),
        "policy_report_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "policy_arrays_sha256": hashlib.sha256(
            (args.policy_folder / "policies.npz").read_bytes()
        ).hexdigest(),
        "stability_report": str(args.stability_report),
        "stability_report_sha256": hashlib.sha256(stability_bytes).hexdigest(),
        "fixed_target_reference": stability.get("target_reference"),
        "config": asdict(solution.config),
        "params": solution.params._asdict(),
        "legacy_missing_feasibility_field": "asset_feasibility" not in policy_report["config"],
        "policy_recovery": "current source with the saved explicit feasibility method",
        "initial_law": asdict(law),
        "targets": asdict(targets),
        "orders": orders,
        "comparison_reference_order": orders[-1],
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in model_root.glob("*.py")
        },
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": {},
        "comparisons": {},
        "limitations": [
            "Only initial population quadrature changes; the saved Bellman grid is fixed.",
            "Fixed-target losses are never retargeted to the highest requested order.",
            "Native endpoint and near-floor masses are compared at identical boundaries.",
            "No fitted-parameter stability, empirical fit, or continuum certificate is supplied.",
            "Every timing includes first population compilation for that quadrature shape.",
        ],
    }
    arrays: dict[str, Any] = {}
    populations, objectives = {}, {}
    for order in orders:
        print(json.dumps({"phase": "population", "order": order}), flush=True)
        nodes = initial_quadrature(
            law, nodes_per_dimension=order, asset_floor=solution.config.asset_minimum
        )
        started = perf_counter()
        population = simulate_population(
            solution,
            nodes,
            backend="quadrature",
            participation_hours_threshold=targets.participation_hours_threshold,
            near_asset_floor_width=targets.near_asset_floor_width,
        )
        objective = weighted_age_moment_loss(population, targets)
        elapsed = perf_counter() - started
        populations[order], objectives[order] = population, objective
        report["runs"][str(order)] = {
            "status": "completed",
            "nodes": int(nodes.weights.size),
            "population_and_loss_seconds_including_first_compile": elapsed,
            "loss_against_fixed_target": objective.loss,
            "predicted_at_target_ages": objective.predicted,
            "standardized_target_residuals": objective.standardized_residuals,
            "diagnostics": population.diagnostics,
        }
        prefix = f"q{order}"
        arrays[f"{prefix}_moment_time"] = population.moments.time
        arrays[f"{prefix}_state_time"] = population.state_moments.time
        arrays[f"{prefix}_initial_assets"] = nodes.assets
        arrays[f"{prefix}_initial_human_capital"] = nodes.human_capital
        arrays[f"{prefix}_initial_weights"] = nodes.weights
        arrays[f"{prefix}_initial_component"] = nodes.component
        for name in MOMENT_UNITS:
            owner = (
                population.moments
                if hasattr(population.moments, name)
                else population.state_moments
            )
            value = getattr(owner, name)
            if value is not None:
                arrays[f"{prefix}_{name}"] = value
        save_report(args.output, report, arrays)
        print(
            json.dumps(
                {
                    "phase": "population complete",
                    "order": order,
                    "loss": objective.loss,
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )
    reference_order = orders[-1]
    for order in orders[:-1]:
        comparison = compare_target_predictions(
            objectives[order], objectives[reference_order], targets
        )
        comparison["whole_native_profile_differences"] = compare_native_profiles(
            populations[order], populations[reference_order]
        )
        thresholds = stability.get("thresholds", {})
        comparison["illustrative_thresholds_from_stability_report"] = thresholds
        comparison["target_moments_within_stability_tolerance"] = (
            comparison["maximum_standardized_moment_difference"]
            <= thresholds["maximum_standardized_moment_difference"]
            if "maximum_standardized_moment_difference" in thresholds
            else None
        )
        comparison["loss_difference_within_stability_tolerance"] = (
            comparison["absolute_loss_difference_against_fixed_target"]
            <= thresholds["maximum_loss_difference"]
            if "maximum_loss_difference" in thresholds
            else None
        )
        report["comparisons"][f"q{order}_versus_q{reference_order}"] = comparison
    report["status"] = "completed"
    save_report(args.output, report, arrays)
    print(json.dumps({"phase": "complete", "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
