"""Compare two existing direct-path initial guesses for one saved cohort type.

CPU example:
  PYTHONPATH=code/jax JAX_PLATFORMS=cpu python tools/benchmark_branch_reference.py
Both paths retain the existing solver/diagnostic thresholds. Two accepted local
solutions are evidence about these initial guesses, not global optimality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from blinder_weiss import SolverConfig, benchmark_params, diagnose_path, solve_lifecycle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_arrays(path: Path, result) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            time=result.time,
            A=result.assets,
            K=result.human_capital,
            c=result.consumption,
            h=result.hours,
            q=result.training_time,
            decision=result.decision,
            log_human_capital=result.log_human_capital,
        )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/solver_benchmarks"))
    parser.add_argument("--person", type=int, default=46)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        parser.error("this independent reference script requires JAX_PLATFORMS=cpu")
    validation_path = args.validation or args.root / "calibration_domain_validation.json"
    validation = json.loads(validation_path.read_text())
    cohort = validation["cohort"]
    if not 0 <= args.person < cohort["people"]:
        parser.error("person index must lie in the saved cohort")
    output = args.output or args.root / f"branch_reference_type{args.person}"
    output.parent.mkdir(parents=True, exist_ok=True)
    params = benchmark_params(
        initial_assets=cohort["assets"][args.person],
        initial_human_capital=cohort["human_capital"][args.person],
    )
    config = SolverConfig(intervals=48, objective_tolerance=1e-11)
    report = {
        "purpose": "two-initial-guess direct collocation for one heterogeneous household",
        "person": args.person,
        "params": params._asdict(),
        "config": asdict(config),
        "source_validation": str(validation_path),
        "source_validation_sha256": _sha256(validation_path),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "current_source_sha256": {
            path.name: _sha256(path)
            for path in (Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss").glob(
                "*.py"
            )
        },
        "mesh_convergence_verified": False,
        "global_optimality_verified": False,
        "branches": [],
    }
    results = []
    started = perf_counter()
    for label, reference_name in (
        ("baseline48", "direct_reference"),
        ("high48", "direct_type_high"),
    ):
        reference_path = args.root / f"{reference_name}.npz"
        reference_metadata = json.loads((args.root / f"{reference_name}.json").read_text())
        if not reference_metadata["reference_accepted"]:
            raise ValueError(f"warm-start reference {reference_name} is not accepted")
        with np.load(reference_path) as reference:
            initial_decision = reference["decision"].copy()
        solve_started = perf_counter()
        print(
            json.dumps({"phase": "solving", "person": args.person, "initial_guess": label}),
            flush=True,
        )
        result = solve_lifecycle(params, config, initial_decision=initial_decision)
        solve_seconds = perf_counter() - solve_started
        print(
            json.dumps(
                {
                    "phase": "diagnosing",
                    "initial_guess": label,
                    "utility": result.lifetime_utility,
                    "optimizer_success": result.optimizer_success,
                    "iterations": result.iterations,
                }
            ),
            flush=True,
        )
        diagnostics = diagnose_path(result, independent_integration=True)
        branch_output = output.with_name(output.name + f"_from_{label}")
        branch = {
            "initial_guess": label,
            "initial_guess_source": str(reference_path),
            "initial_guess_sha256": _sha256(reference_path),
            "initial_state_equalities": "enforced by solver; seed decision retained from source",
            "config": asdict(config),
            "params": params._asdict(),
            "lifetime_utility": result.lifetime_utility,
            "diagnostics": diagnostics.as_dict(),
            "reference_accepted": bool(diagnostics.accepted_success),
            "comparison_usable": bool(diagnostics.accepted_success),
            "mesh_convergence_verified": False,
            "global_optimality_verified": False,
            "optimizer_success": result.optimizer_success,
            "solver_success": result.success,
            "optimizer_message": result.message,
            "iterations": result.iterations,
            "independent_integration": True,
            "solve_seconds": solve_seconds,
            "wall_seconds": perf_counter() - solve_started,
            "path_artifact": str(branch_output.with_suffix(".npz")),
        }
        _write_arrays(branch_output.with_suffix(".npz"), result)
        _write_json(branch_output.with_suffix(".json"), branch)
        report["branches"].append(branch)
        results.append(result)
        report["total_wall_seconds"] = perf_counter() - started
        _write_json(output.with_suffix(".json"), report)
        print(
            json.dumps(
                {"phase": "validated", "initial_guess": label, "diagnostics": diagnostics.as_dict()}
            ),
            flush=True,
        )
    accepted = [
        index for index, branch in enumerate(report["branches"]) if branch["reference_accepted"]
    ]
    selected = (
        max(accepted, key=lambda index: results[index].lifetime_utility) if accepted else None
    )
    report["reference_accepted"] = selected is not None
    report["comparison_usable"] = selected is not None
    report["selected_branch"] = selected
    report["branch_utility_difference_high_minus_baseline"] = (
        results[1].lifetime_utility - results[0].lifetime_utility
    )
    report["between_branch_path_errors"] = {
        field: {
            "rmse": float(
                np.sqrt(np.mean((getattr(results[1], field) - getattr(results[0], field)) ** 2))
            ),
            "maximum_absolute": float(
                np.max(np.abs(getattr(results[1], field) - getattr(results[0], field)))
            ),
        }
        for field in ("assets", "human_capital", "consumption", "hours", "training_time")
    }
    if selected is not None:
        selected_branch = report["branches"][selected]
        report["lifetime_utility"] = selected_branch["lifetime_utility"]
        report["diagnostics"] = selected_branch["diagnostics"]
        report["path_artifact"] = str(output.with_suffix(".npz"))
        _write_arrays(output.with_suffix(".npz"), results[selected])
    report["total_wall_seconds"] = perf_counter() - started
    _write_json(output.with_suffix(".json"), report)
    print(
        json.dumps(
            {
                "phase": "complete",
                "reference_accepted": selected is not None,
                "selected_branch": selected,
                "wall_seconds": report["total_wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
