# JAX lifecycle solvers

For a derivation of the transcription, optimization theory, JAX autodiff, and
the validation strategy, see [SOLUTION_STRATEGY.md](SOLUTION_STRATEGY.md).
For the backward dynamic-programming implementation and feedback policies, see
[BELLMAN_STRATEGY.md](BELLMAN_STRATEGY.md).

There are two independent active implementations of Blinder and Weiss (1976):

- direct collocation solves one deterministic lifecycle path for specified
  initial assets and human capital; and
- semi-Lagrangian backward induction approximates `V(t, A, K)` and the
  feedback policies `c*(t, A, K)`, `h*(t, A, K)`, and `q*(t, A, K)`.

The direct solution is an independent benchmark for the Bellman policies along
the realized lifecycle. The Bellman solution is the base for heterogeneous
initial states and future stochastic extensions.

## Model contract

The implementation maximizes

```text
integral_0^T exp(-rho*t) [U(c(t)) + V(1-h(t))] dt
    + exp(-rho*T) B(A(T))
```

subject to

```text
A_dot = r*A + h*g(x)*K - c
K_dot = (a*x*h - delta)*K
```

and `A >= 0`, `c > 0`, `0 <= h <= 1`, `0 <= x <= 1`. The initial
values of `A` and `K` are fixed; terminal assets and human capital are free.

Internally, the solver uses `log(K)` and training time `q = h*x`. The
constraint `0 <= q <= h` is exactly equivalent to the original control set,
but removes the undefined value of `x` in retirement. For the benchmark
quadratic tradeoff, the perspective `h*g(q/h)` is evaluated analytically,
without division by zero or clipping.

The terminal bequest is discounted. The human-capital productivity `a` that
was omitted from one equation in the old writeup is included consistently.

## Benchmark calibration

| Quantity | Value | Interpretation |
|---|---:|---|
| `T` | 70 | Years/model periods |
| `A(0)` | 5 | Wealth that can finance an initial schooling spell |
| `K(0)` | 1 | Human-capital normalization |
| `rho` | 0.03 | Discount rate |
| `r` | 0.05 | Asset return |
| `delta` | 0.05 | Human-capital depreciation |
| `a` | 0.22 | Maximum annual human-capital growth |
| CRRA coefficients | 2 | Power exponents `1 - sigma = -1` for consumption, leisure, and bequest |

This calibration is a numerical benchmark, not an estimate from the paper.
It generates a finite schooling spell, declining on-the-job training, pure
work, and retirement. The legacy value `a = 1` implies potential continuous
growth of approximately `exp(0.95 * 70)` and is retained only as a stress
case through the parameter controls.

## Direct-collocation method

States and controls at all time nodes are optimized jointly. JAX provides the
objective gradient and constraint Jacobians in float64. The default
Hermite--Simpson transcription is solved by SLSQP; a sparse IPOPT backend is
also available. The borrowing limit is enforced both at state nodes and at
Hermite--Simpson midpoints, preventing the cubic state interpolant from dipping
below the boundary between two feasible nodes. The problem is nonconvex, so
convergence to a feasible KKT point is not a proof of global optimality.

The default SLSQP solve performs one warm restart. This resets SLSQP's quasi-
Newton approximation after it discovers the active borrowing-limit arc; the
restart is retained only when it remains successful and does not worsen the
objective.

A path is treated as numerically credible only when:

1. the optimizer reports success;
2. collocation and bound violations are below tolerance;
3. the projected KKT residual is small;
4. a separate adaptive Diffrax integration on a grid ten times denser reproduces
   the state path and respects the borrowing limit to the documented numerical
   tolerance; and
5. the objective and paths stabilize under mesh refinement.

The continuous-time Pontryagin residuals reported by the diagnostics exclude
the part of the path at or before the last contact with the borrowing limit,
where the state-constraint multiplier changes the simple costate equations.

## Bellman method

The Bellman solver works backward over a bounded grid in assets and log human
capital. Controls are constant within a Bellman period, allowing the state
equations to be integrated exactly. The next-period value is evaluated by
monotone bilinear interpolation; no finite-difference derivative of the value
function is used.

At each state node, a batched global control grid is followed by projected
autodiff refinement. The largest feasible consumption is calculated from the
asset floor at multiple within-period checkpoints, so the borrowing constraint
enters the control set rather than a guessed boundary derivative. Candidate
transitions outside the represented state domain are rejected.

After local refinement, configurable neighboring-policy sweeps evaluate the
absolute controls selected at poorer adjacent states. A feasible inherited
control replaces the incumbent only when it raises value. This guards against
isolated optimizer misses without mechanically altering values or breaking the
Bellman identity.

The default annual configuration stores only 55,025 state-age values. The
result includes value and policy arrays, a forward policy simulator, and
diagnostics for Bellman consistency, feasibility, monotonicity, and domain
containment. State, control, time, and domain convergence remain necessary.

The complete backward recursion is compiled as one `jax.lax.scan`. State and
control grids, continuation values, and intermediate policies remain on the
selected JAX device for all ages; the completed value and policy histories are
copied to NumPy once. `compute_platform="auto"` uses CUDA when it is available.
Use `compute_platform="gpu"` to require CUDA and fail clearly instead of
silently running a calibration job on the CPU.

For off-grid states, `greedy_policy_at()` re-maximizes the Bellman objective in
a JAX batch against the stored next-period value function. The interpolated
node policy is included as a fallback candidate. `simulate_policy()` uses this
greedy recovery by default; `policy_method="interpolate"` retains the cheaper
control-interpolation rollout as an explicit convergence comparison.

## Running

Enter the project environment and synchronize it:

```bash
devenv shell
uv sync
```

Verify that the CUDA plugin sees the NVIDIA device:

```bash
gpu-run uv run python -c \
  "import jax; print(jax.default_backend()); print(jax.devices())"
```

Open the baseline notebook:

```bash
uv run marimo edit code/jax/notebooks/01_baseline.py
```

Open the convergence notebook:

```bash
uv run marimo edit code/jax/notebooks/02_mesh_convergence.py
```

Open the Bellman policy notebook:

```bash
gpu-run uv run marimo edit code/jax/notebooks/03_bellman_policy_functions.py
```

The reusable feedback-policy API is:

```python
from blinder_weiss import (
    BellmanConfig,
    diagnose_bellman,
    greedy_policy_at,
    simulate_policy,
    solve_bellman,
)

solution = solve_bellman(config=BellmanConfig(compute_platform="gpu"))
greedy_simulation = simulate_policy(solution, policy_method="greedy")
interpolated_simulation = simulate_policy(solution, policy_method="interpolate")
diagnostics = diagnose_bellman(solution, greedy_simulation)

consumption, hours, training = greedy_policy_at(
    solution, period=20, assets=[2.0, 5.0], human_capital=[1.0, 1.5]
)

print(solution.backend, solution.device)
```

Run automated checks:

```bash
uv run pytest
uv run ruff check .
uv run basedpyright code/jax/blinder_weiss tests
```
