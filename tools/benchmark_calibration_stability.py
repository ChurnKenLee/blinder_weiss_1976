"""Initial-law sensitivity and synthetic calibration across numerical resolutions.

The finest time solution and quadrature rule generate one fixed target. Every
coarser evaluation and fitted parameter is compared with that target; it is not
regenerated separately at each resolution. Passing explicit tolerances gives
local numerical stability evidence, never empirical identification.

CPU smoke: PYTHONPATH=code/jax JAX_PLATFORMS=cpu python tools/benchmark_calibration_stability.py \
  --periods 4 8 --orders 4 8 16 --fit-initial-law --output DIR
GPU work should be coordinated; --config-report imports a recorded numerical
configuration but re-solves the current implementation at each requested period count.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from blinder_weiss.bellman import BellmanConfig, solve_bellman
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationEvaluation,
    CalibrationTargets,
    NumericalAcceptanceThresholds,
    compare_calibration_resolutions,
    evaluate_initial_law,
    fit_initial_law_calibration,
    fit_scalar_calibration,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    initial_quadrature,
    simulate_population,
    synthetic_initial_scenarios,
)
from blinder_weiss.model import benchmark_params


def _benchmark_helpers():
    spec = importlib.util.spec_from_file_location(
        "continuum_benchmark_helpers", Path(__file__).with_name("benchmark_continuum.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--config-report", type=Path)
    parser.add_argument(
        "--resolution-config-reports",
        type=Path,
        nargs="+",
        help="One saved config per period count; permits joint time/state-grid refinement",
    )
    parser.add_argument("--periods", type=int, nargs="+")
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--parameter", default="leisure_weight")
    parser.add_argument("--bounds", type=float, nargs=2, default=[0.9, 1.1])
    parser.add_argument("--fit-evaluations", type=int, default=30)
    parser.add_argument("--parameter-tolerance", type=float, default=1e-3)
    parser.add_argument("--near-floor-width", type=float, default=0.01)
    parser.add_argument(
        "--moment-tolerance",
        type=float,
        default=0.1,
        help="Maximum change as a fraction of a target residual scale",
    )
    parser.add_argument("--loss-tolerance", type=float, default=0.001)
    parser.add_argument("--fit-tolerance", type=float, default=0.002)
    parser.add_argument("--fit-initial-law", action="store_true")
    args = parser.parse_args()
    specified_configs = {}
    if args.resolution_config_reports:
        if args.config_report is not None:
            parser.error("use config-report or resolution-config-reports, not both")
        specifications = [json.loads(path.read_text()) for path in args.resolution_config_reports]
        for path, specification in zip(args.resolution_config_reports, specifications, strict=True):
            count = specification["config"]["periods"]
            if count in specified_configs:
                parser.error("resolution-config-reports must have distinct period counts")
            if specification["params"] != specifications[0]["params"]:
                parser.error("resolution configurations must share structural parameters")
            specified_configs[count] = BellmanConfig(
                **{**specification["config"], "compute_platform": args.platform}
            )
        if args.periods is not None and set(args.periods) != set(specified_configs):
            parser.error("periods must match supplied resolution-config-reports")
        args.periods = sorted(specified_configs)
        # Select by actual count rather than the input order of report paths.
        args.config_report = max(
            args.resolution_config_reports,
            key=lambda path: json.loads(path.read_text())["config"]["periods"],
        )
    if args.periods is None:
        starting_periods = (
            json.loads(args.config_report.read_text())["config"]["periods"]
            if args.config_report
            else 4
        )
        args.periods = [starting_periods, 2 * starting_periods]
    periods, orders = sorted(set(args.periods)), sorted(set(args.orders))
    if min(periods) < 2 or min(orders) < 2 or len(periods) * len(orders) < 2:
        parser.error("at least two distinct resolutions with periods/orders >=2 required")
    if not np.isfinite(args.near_floor_width) or args.near_floor_width <= 0:
        parser.error("near-floor-width must be finite and positive")
    if args.config_report:
        specification = json.loads(args.config_report.read_text())
        base_config = BellmanConfig(
            **{**specification["config"], "compute_platform": args.platform}
        )
        params = benchmark_params(**specification["params"])
    else:
        base_config = BellmanConfig(
            periods=periods[-1],
            asset_nodes=7,
            human_capital_nodes=7,
            asset_minimum=1e-3,
            asset_maximum=12.0,
            log_human_capital_minimum=-1.5,
            log_human_capital_maximum=1.0,
            hours_nodes=5,
            investment_nodes=5,
            consumption_nodes=7,
            refinement_steps=4,
            control_batch_size=32,
            neighbor_policy_sweeps=2,
            compute_platform=args.platform,
        )
        params = benchmark_params(horizon=4.0)
    configs = {
        count: specified_configs.get(count, replace(base_config, periods=count))
        for count in periods
    }
    if any(config.asset_minimum != base_config.asset_minimum for config in configs.values()):
        parser.error("this initial floor-mixture comparison requires the same numerical floor")
    known_parameter = float(getattr(params, args.parameter))
    if not args.bounds[0] < known_parameter < args.bounds[1]:
        parser.error("synthetic known parameter must lie strictly inside fitting bounds")
    law = synthetic_initial_scenarios()["baseline"]
    threshold = 0.02
    helpers = _benchmark_helpers()
    timer = helpers.CompilationTimer()
    args.output.mkdir(parents=True, exist_ok=True)
    model_root = Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss"
    thresholds = NumericalAcceptanceThresholds(
        args.moment_tolerance, args.loss_tolerance, args.fit_tolerance
    )
    report = {
        "purpose": "synthetic calibration stability and initial-law sensitivity",
        "empirical_calibration": False,
        "continuum_convergence_certified": False,
        "config": asdict(base_config),
        "resolution_configs": {str(count): asdict(config) for count, config in configs.items()},
        "resolution_config_sources": (
            [str(path) for path in args.resolution_config_reports]
            if args.resolution_config_reports
            else None
        ),
        "params": params._asdict(),
        "initial_law": asdict(law),
        "parameter": args.parameter,
        "known_parameter": known_parameter,
        "periods": periods,
        "quadrature_orders": orders,
        "near_asset_floor_width": args.near_floor_width,
        "thresholds": asdict(thresholds),
        "tolerance_provenance": "illustrative numerical tolerances, not survey standard errors",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in model_root.glob("*.py")
        },
        "resolutions": {},
        "initial_law_sensitivity": {},
        "comparisons": {},
    }
    arrays = {}

    def checkpoint():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(helpers.jsonable(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "report.json")
        temporary = args.output / "profiles.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, allow_pickle=False, **arrays)
        temporary.replace(args.output / "profiles.npz")

    solutions = {}
    for count in periods:
        print(json.dumps({"phase": "policy", "periods": count}), flush=True)
        solutions[count], timing = timer.measure(
            lambda count=count: solve_bellman(params, configs[count])
        )
        report.setdefault("policy_timings", {})[str(count)] = timing
    reference_solution = solutions[periods[-1]]
    reference_nodes = initial_quadrature(
        law, nodes_per_dimension=orders[-1], asset_floor=base_config.asset_minimum
    )
    reference_population = simulate_population(
        reference_solution,
        reference_nodes,
        participation_hours_threshold=threshold,
        near_asset_floor_width=args.near_floor_width,
    )
    # Common ages stop at the earliest last-decision time: no terminal controls/extrapolation.
    common_time = np.linspace(0.0, params.horizon * (1.0 - 1.0 / periods[0]), 7)
    scales = {
        "hours": 0.02,
        "participation": 0.02,
        "training_time": 0.02,
        "consumption": 0.1,
        "earnings": 0.1,
        "assets": 0.2,
        "human_capital": 0.1,
        "near_asset_floor_mass": 0.02,
    }
    profiles = []
    for name, scale in scales.items():
        owner = (
            reference_population.moments
            if hasattr(reference_population.moments, name)
            else reference_population.state_moments
        )
        profiles.append(
            AgeMomentTarget(
                name,
                common_time + 20.0,
                np.interp(common_time, owner.time, getattr(owner, name)),
                scale,
                1.0,
                MOMENT_UNITS[name],
            )
        )
    targets = CalibrationTargets(
        tuple(profiles),
        20.0,
        threshold,
        "synthetic fixed target from finest time/quadrature resolution; no survey estimates",
        near_asset_floor_width=args.near_floor_width,
    )
    report["targets"] = asdict(targets)
    reference_label = f"periods{periods[-1]}_q{orders[-1]}"
    report["target_reference"] = reference_label
    baseline_evaluations, fitted = {}, {}
    for count in periods:
        solution = solutions[count]
        for order in orders:
            label = f"periods{count}_q{order}"
            nodes = initial_quadrature(
                law, nodes_per_dimension=order, asset_floor=base_config.asset_minimum
            )
            print(
                json.dumps({"phase": "baseline and structural fit", "resolution": label}),
                flush=True,
            )
            started = perf_counter()
            population = simulate_population(
                solution,
                nodes,
                participation_hours_threshold=threshold,
                near_asset_floor_width=args.near_floor_width,
            )
            objective = weighted_age_moment_loss(population, targets)
            elapsed = perf_counter() - started
            baseline_evaluations[label] = CalibrationEvaluation(
                params, population, objective, 0.0, elapsed, elapsed
            )
            fit, fit_timing = timer.measure(
                lambda solution=solution, nodes=nodes: fit_scalar_calibration(
                    args.parameter,
                    tuple(args.bounds),
                    params=params,
                    config=solution.config,
                    initial_nodes=nodes,
                    targets=targets,
                    parameter_tolerance=args.parameter_tolerance,
                    max_evaluations=args.fit_evaluations,
                )
            )
            fitted[label] = (float(fit.x), bool(fit.success))
            report["resolutions"][label] = {
                "periods": count,
                "quadrature_order": order,
                "asset_nodes": solution.config.asset_nodes,
                "human_capital_nodes": solution.config.human_capital_nodes,
                "baseline_loss_against_fixed_target": objective.loss,
                "fitted_parameter": float(fit.x),
                "parameter_error": abs(fit.x - known_parameter),
                "fitted_loss": float(fit.fun),
                "optimizer_success": bool(fit.success),
                "optimizer_message": str(fit.message),
                "selection_source": fit.selection_source,
                "raw_optimizer": fit.raw_optimizer,
                "selected_fit_no_worse_than_initial_parameter": fit.fun <= objective.loss + 1e-12,
                "fit_timing": fit_timing,
                "diagnostics": population.diagnostics,
                "fit_trace": [
                    {
                        "parameter": float(getattr(item.params, args.parameter)),
                        "loss": item.objective.loss,
                        "seconds": item.total_seconds,
                    }
                    for item in fit.evaluations
                ],
            }
            arrays[f"{label}_time"] = population.moments.time
            arrays[f"{label}_state_time"] = population.state_moments.time
            for name in (*scales, "asset_floor_mass"):
                owner = (
                    population.moments
                    if hasattr(population.moments, name)
                    else population.state_moments
                )
                arrays[f"{label}_{name}"] = getattr(owner, name)
            del fit
            checkpoint()
    for label, evaluation in baseline_evaluations.items():
        if label == reference_label:
            continue
        report["comparisons"][label] = compare_calibration_resolutions(
            evaluation,
            baseline_evaluations[reference_label],
            targets,
            fitted_parameters=(fitted[label][0], fitted[reference_label][0]),
            optimizer_success=(fitted[label][1], fitted[reference_label][1]),
            thresholds=thresholds,
        )
    report["all_tested_resolutions_within_tolerance"] = all(
        comparison["passed"] for comparison in report["comparisons"].values()
    )
    for label, scenario in synthetic_initial_scenarios().items():
        evaluation = evaluate_initial_law(
            reference_solution, scenario, targets, nodes_per_dimension=orders[-1]
        )
        report["initial_law_sensitivity"][label] = {
            "initial_law": asdict(scenario),
            "loss_against_baseline_target": evaluation.objective.loss,
            "seconds": evaluation.total_seconds,
            "predicted": evaluation.objective.predicted,
            "diagnostics": evaluation.population.diagnostics,
        }
        for name in ("hours", "participation", "earnings"):
            arrays[f"scenario_{label}_{name}"] = getattr(evaluation.population.moments, name)
        checkpoint()
    if args.fit_initial_law:
        fit = fit_initial_law_calibration(
            "asset_floor_mass",
            (0.0, 0.15),
            solution=reference_solution,
            law=law,
            targets=targets,
            nodes_per_dimension=orders[-1],
            parameter_tolerance=args.parameter_tolerance,
            max_evaluations=args.fit_evaluations,
        )
        report["initial_law_fit"] = {
            "parameter": "asset_floor_mass",
            "known_value": law.asset_floor_mass,
            "fitted_value": float(fit.x),
            "optimizer_success": bool(fit.success),
            "loss": float(fit.fun),
            "message": str(fit.message),
            "structural_policy_reused": True,
            "trace": [
                {
                    "floor_share": item.law.asset_floor_mass,
                    "loss": item.objective.loss,
                    "seconds": item.total_seconds,
                }
                for item in fit.evaluations
            ],
        }
    checkpoint()
    print(
        json.dumps(
            {
                "phase": "complete",
                "report": str(args.output / "report.json"),
                "local_checks_passed": report["all_tested_resolutions_within_tolerance"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
