"""Reproduce synchronized GPU solve timings and policy diagnostics.

Run with PYTHONPATH=code/jax python tools/benchmark_bellman.py --output <directory>.
The first call includes compilation; subsequent model parameters stay dynamic.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from blinder_weiss import BellmanConfig, benchmark_params, diagnose_bellman, solve_bellman


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-nodes", type=int, default=31)
    parser.add_argument("--human-capital-nodes", type=int, default=25)
    parser.add_argument("--asset-maximum", type=float, default=35.0)
    parser.add_argument("--log-human-capital-minimum", type=float, default=-2.0)
    parser.add_argument("--log-human-capital-maximum", type=float, default=2.25)
    parser.add_argument("--periods", type=int, default=70)
    parser.add_argument(
        "--asset-feasibility", choices=["continuous", "checkpoints"], default="continuous"
    )
    parser.add_argument("--control-nodes", type=int, default=11)
    parser.add_argument("--consumption-nodes", type=int, default=15)
    parser.add_argument("--refinement-steps", type=int, default=24)
    parser.add_argument("--refinement-starts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--polish-consumption", action="store_true")
    parser.add_argument("--neighbor-destinations", action="store_true")
    parser.add_argument("--neighbor-sweeps", type=int, default=2)
    parser.add_argument("--neighbor-tolerance", type=float, default=None)
    parser.add_argument(
        "--interpolation", choices=["bilinear", "pchip", "monotone_bicubic"], default="bilinear"
    )
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cfg = BellmanConfig(
        compute_platform="gpu",
        asset_nodes=args.asset_nodes,
        human_capital_nodes=args.human_capital_nodes,
        asset_maximum=args.asset_maximum,
        log_human_capital_minimum=args.log_human_capital_minimum,
        log_human_capital_maximum=args.log_human_capital_maximum,
        periods=args.periods,
        asset_feasibility=args.asset_feasibility,
        hours_nodes=args.control_nodes,
        investment_nodes=args.control_nodes,
        consumption_nodes=args.consumption_nodes,
        refinement_steps=args.refinement_steps,
        refinement_starts=args.refinement_starts,
        control_batch_size=args.batch_size,
        value_interpolation=args.interpolation,
        neighbor_destination_candidates=args.neighbor_destinations,
        neighbor_policy_sweeps=args.neighbor_sweeps,
        neighbor_policy_tolerance=args.neighbor_tolerance,
        **({"consumption_polish": True} if args.polish_consumption else {}),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        jax=jax.__version__,
        python=platform.python_version(),
        config=asdict(cfg),
        timing_includes_host_transfer=True,
        results=[],
    )
    solution = None
    for i in range(args.repeats + 1):
        # Alternating nearby parameters detects accidental parameter-specialized JIT caches.
        params = benchmark_params(leisure_weight=1.0 + (0.01 if i % 2 else 0.0))
        start = perf_counter()
        solution = solve_bellman(params, cfg)
        wall = perf_counter() - start
        row = dict(
            call=i,
            first_call_in_process=i == 0,
            wall_seconds=wall,
            solve_seconds=solution.solve_seconds,
            leisure_weight=params.leisure_weight,
            initial_value=solution.initial_value,
        )
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        report["device"] = solution.device
        report["device_memory_stats"] = jax.devices("gpu")[cfg.device_index].memory_stats()
        report["params"] = params._asdict()
        temp = args.output / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output / "report.json")
    assert solution is not None
    if args.diagnostics:
        start = perf_counter()
        report["diagnostics"] = diagnose_bellman(solution).as_dict()
        report["diagnostics_seconds"] = perf_counter() - start
    arrays = dict(
        values=solution.values,
        c=solution.consumption_policy,
        h=solution.hours_policy,
        q=solution.training_time_policy,
        A=solution.asset_grid,
        y=solution.log_human_capital_grid,
        time=solution.time,
    )
    with (args.output / "policies.npz.tmp").open("wb") as output:
        np.savez_compressed(output, **arrays)
    (args.output / "policies.npz.tmp").replace(args.output / "policies.npz")
    (args.output / "report.json.tmp").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "report.json.tmp").replace(args.output / "report.json")


if __name__ == "__main__":
    main()
