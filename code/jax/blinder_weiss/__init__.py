"""JAX solver for the Blinder--Weiss (1976) lifecycle model."""

import jax

# This is a long-horizon constrained optimal-control problem. JAX defaults to
# float32, which is not adequate for its feasibility and KKT tolerances.
jax.config.update("jax_enable_x64", True)

from .bellman import (  # noqa: E402
    BellmanConfig,
    BellmanDiagnostics,
    BellmanSimulation,
    BellmanSolution,
    constant_control_transition,
    diagnose_bellman,
    greedy_policy_at,
    policy_at,
    simulate_policy,
    solve_bellman,
)
from .diagnostics import (  # noqa: E402
    PathDiagnostics,
    diagnose_path,
    regime_labels,
)
from .model import (  # noqa: E402
    ModelParams,
    benchmark_params,
    bequest_utility,
    dynamics,
    earnings_tradeoff,
    flow_utility,
)
from .solver import (  # noqa: E402
    CollocationResult,
    SolverConfig,
    solve_lifecycle,
    solve_mesh_sequence,
)

__all__ = [
    "BellmanConfig",
    "BellmanDiagnostics",
    "BellmanSimulation",
    "BellmanSolution",
    "CollocationResult",
    "ModelParams",
    "PathDiagnostics",
    "SolverConfig",
    "benchmark_params",
    "bequest_utility",
    "constant_control_transition",
    "diagnose_bellman",
    "diagnose_path",
    "dynamics",
    "earnings_tradeoff",
    "flow_utility",
    "greedy_policy_at",
    "policy_at",
    "regime_labels",
    "solve_lifecycle",
    "simulate_policy",
    "solve_bellman",
    "solve_mesh_sequence",
]
