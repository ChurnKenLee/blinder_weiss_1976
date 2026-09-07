# Solution procedure

The code solves the lifecycle model in two independent ways. The main method
computes value and feedback-policy functions by backward induction; direct
collocation supplies a pathwise check.

## Bellman solution

The state is assets \(A\) and log human capital \(y=\log K\). Controls are
consumption \(c\), working time \(h\), and training time \(q=hx\), so
\(0\leq q\leq h<1\) remains well defined at retirement.

For \(N\) periods of length \(\Delta t=T/N\), the code applies

\[
V_n(A,y)=\max_{c,h,q}
\left\{D_u(\Delta t)u(c,h)
+e^{-\rho\Delta t}V_{n+1}(A',y')\right\},
\qquad
D_u(\Delta t)=\frac{1-e^{-\rho\Delta t}}{\rho}.
\]

Controls are constant within a period, so \((A',y')\) is computed analytically.
Candidate controls must respect the asset floor at several within-period
checkpoints and keep the next state inside the computational domain.
Continuation values are obtained by bilinear interpolation on a curved asset
grid and a uniform log-human-capital grid. In the final decision period, the
terminal bequest is evaluated exactly rather than interpolated.

At every state node, the optimizer:

1. searches a batched tensor grid of controls;
2. retains several leading candidates plus the next-age policy;
3. improves each start with projected autodiff and backtracking; and
4. tests feasible policies inherited from neighboring state nodes.

The best feasible control and its value are stored. Backward induction over
age is compiled with `jax.lax.scan`; states and controls within each age are
evaluated in parallel.

## Accuracy and use

`solve_bellman_converged()` repeatedly expands the computational domain and
refines time, state, control, feasibility-checkpoint, and local-search
resolution. Every level is compared on the original target grid. A level
passes only when value changes, preceding-policy regret, optimizer shortfall,
monotonicity, feasibility, and domain containment meet their tolerances. Two
consecutive passes are required by default; otherwise the finest result is
returned with `converged=False`.

`policy_at()` and `value_at()` interpolate the stored arrays. For simulation,
`greedy_policy_at()` re-solves the one-period problem at each off-grid state
and is more reliable than directly interpolating controls.

```python
from blinder_weiss import (
    BellmanConfig,
    greedy_policy_at,
    solve_bellman_converged,
    value_at,
)

result = solve_bellman_converged(config=BellmanConfig(compute_platform="auto"))
solution = result.solution
c, h, q = greedy_policy_at(solution, 20, [2.0, 5.0], [1.0, 1.5])
values = value_at(solution, 20, [2.0, 5.0], [1.0, 1.5])
```

## Independent check

The direct solver optimizes one lifecycle path with a Hermite--Simpson
transcription and JAX derivatives. Mesh refinement, feasibility, KKT
residuals, and independent integration validate that path. Agreement between
this path and a Bellman-policy rollout is the final cross-check.

Full derivations are in [BELLMAN_STRATEGY.md](BELLMAN_STRATEGY.md) and
[SOLUTION_STRATEGY.md](SOLUTION_STRATEGY.md).
