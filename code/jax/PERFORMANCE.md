# GPU performance and policy accuracy

Measured on 2026-09-07 using the NVIDIA RTX PRO 6000 Blackwell Server Edition,
JAX/JAXlib/CUDA plugin 0.11.0, float64, and Python 3.13.11 in the live environment.
The project dependency manifest specifies a different reproducible environment
(Python >=3.14 and JAX 0.10.1); these measurements describe the actual installed
runtime. They are not measurements of the RTX 5070 Ti.

## Implemented improvements

- Backward induction, greedy off-grid recovery, and population rollout reuse
  compiled functions across changes in model parameters. Static grid settings,
  device, and batch shape determine compilation; model parameter values do not.
- Known curved asset and uniform log-human-capital grids use arithmetic cell
  lookup with exact-node rounding corrections. General interpolation helpers
  retain binary search for arbitrary grids.
- `consumption_fraction_minimum` now sets the global seed-grid endpoint only.
  Local refinement can reach the actual consumption floor. Previously its
  effective lower bound increased as the time step decreased.
- `BellmanConfig(consumption_polish=True)` optionally maximizes consumption
  exactly conditional on hours and training for bilinear continuation. It scans
  all asset interpolation intervals and uses analytic stationary points and
  endpoints; terminal bequest uses bracketed marginal-utility bisection.
  This is conditional optimality, not proof of global optimality over all
  controls. Finite-search choices in subsequent periods can still change.
- `BellmanConfig(value_interpolation="pchip")` selects experimental sequential
  cubic continuation: PCHIP in assets, then log human capital. Asset derivative
  tables are prepared once per age. Recovery, value queries, and diagnostics
  use the same representation. Node policy interpolation remains bilinear;
  greedy recovery is still the default simulator. Exact bilinear consumption
  polish cannot be combined with PCHIP.
- `simulate_cohort` batches heterogeneous lifecycles in one compiled scan.
  `cohort_moments` returns weighted model age profiles. IPOPT is imported only
  when its separate native backend is requested.

The cache correction follows the function-identity and dynamic-input rules in
[JAX's compilation documentation](https://docs.jax.dev/en/latest/201/jit.html).
Timings synchronize on completed results and include device-to-host output
transfer, consistent with [JAX's benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html).

## Measured timings

| Configuration | Original repeated solve | Improved repeated solve |
|---|---:|---:|
| 70 periods, 31 x 25 states, 11 x 11 x 15 control seeds, 24 local steps | 4.63–4.94 s | 0.73–0.79 s, bilinear |
| Same, conditional consumption polish | — | 0.82 s |
| Same, experimental PCHIP | — | 0.93 s |
| 140 periods, 61 x 49 states, 15 x 15 x 19 seeds, 48 local steps | 11.43 s | 3.96 s, bilinear + polish |
| Same fine grid, experimental PCHIP | — | 4.53 s |

Original measurements rebuild the original compiled closure on every call.
Improved warm calls reuse compilation, including after a 1% change in leisure
weight. Initial calls are slower: roughly 7 s for coarse PCHIP and 11 s for fine
PCHIP including first-process setup. Avoid treating an asynchronous kernel
launch as a completed calibration evaluation.

A 70-period greedy cohort rollout, feasibility checks, host transfer, and
weighted moments took 0.72 s for 100 people and 0.81 s for 1,000 people after
compilation. Initial cohort calls took about 5 s. These measurements use the
coarse bilinear solution with conditional consumption polish and synthetic
initial assets in [2,8], log human capital in [-0.25,0.25], seed 125. They do not
measure data loading, estimation, or the cost of a fine-grid cohort.

A complete coarse PCHIP evaluation (solve, 256-person weighted rollout, and a
28-moment synthetic loss) takes 1.73–1.74 s after compilation. The first call
was 12.26 s. Repeating the baseline at the end reproduces zero synthetic loss
exactly; +/-1% and +/-2% leisure-weight perturbations move the loss away from
zero. These same-grid synthetic targets isolate numerical performance; they
are not an empirical calibration. Reproduce with:

```bash
PYTHONPATH=code/jax python tools/benchmark_calibration.py \
  --interpolation pchip --people 256 \
  --output output/solver_benchmarks/calibration_pchip.json
```

Raw results and arrays are in [`output/solver_benchmarks`](../../output/solver_benchmarks).
The original numerical baseline is commit `f331628`; the temporary baseline
copy only moved the unused IPOPT import to its backend branch.

## Accuracy and smoothness evidence

At baseline economic parameters:

| Quantity | Bilinear + polish, coarse | PCHIP, coarse | PCHIP, fine |
|---|---:|---:|---:|
| Initial represented value | -100.218064 | -99.542813 | -99.483157 |
| Greedy realized utility | -99.525436 | -99.496576 | -99.490306 |
| Represented value minus rollout utility | -0.692628 | -0.046237 | 0.007149 |
| Consumption RMSE vs direct reference | 0.033048 | 0.003259 | 0.000685 |
| Active-time RMSE vs direct reference | 0.006702 | 0.002012 | 0.000843 |
| Training-time RMSE vs direct reference | 0.010044 | 0.003234 | 0.001380 |
| Retirement consumption Euler residual, RMS per year | 0.044016 | 0.000840 | 0.000674 |

The independent direct reference uses 48 Hermite–Simpson intervals and tightened
SLSQP objective tolerance `1e-11`. Its utility is -99.4888795, projected KKT
residual 2.42e-7, independently integrated asset error 1.48e-4, and human-capital
relative error 1.14e-7. It passes the existing diagnostic thresholds, including
the documented integration allowance at the borrowing boundary. This is one
accepted mesh, not a mesh-convergence certificate. All four reference attempts
are recorded in `direct_reference.json`.

Control RMSE compares constant-period controls with the direct path at period
midpoints. In unconstrained retirement the independent consumption condition is
`log(c[n+1]/c[n])/dt = (r-rho)/(1-consumption_power) = 0.01`. The Euler comparison
uses pairs of retired periods away from the asset floor. These are quantitative
smoothness and economic consistency checks, not cosmetic smoothing of curves.
See `lifecycle_comparison.png` and `accuracy_summary.json` for the paths and
six-age dense derivative checks over A in [0.05,25] and log K in [-0.5,1].

All three runs pass node feasibility and their own Bellman identity. The
simulation domain check now allows `1e-10` boundary roundoff, matching control
acceptance; a value such as `9.9999999999989e-5` at a `1e-4` floor is no longer
reported as a substantive domain exit. Regression tests separately reject real
exits.

## Remaining acceptance work

**PCHIP remains experimental.** One-dimensional PCHIP is C1 and shape-preserving,
as documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).
Sequential two-dimensional PCHIP does not guarantee coordinate monotonicity or
ordering with respect to nodal values. Actual solved tables have small late-life
negative derivatives with respect to human capital, including in the validation
interior: the fine run reaches about -0.0229 at age 65. They shrink under the
current refinement but have not disappeared. This rules out claiming that the
full policy surface is numerically accepted for calibration.

The next numerical experiment is a jointly monotone bicubic Hermite
construction with limited first and mixed derivatives, following the
[Carlson–Fritsch monotone bicubic method](https://epubs.siam.org/doi/10.1137/0726013).
A port must independently verify all cell inequalities and both coordinate
monotonicity constraints. Simply zeroing mixed derivatives or postprocessing
policy plots does not establish those properties. Further acceptance needs:

1. monotonicity and regime-aware smoothness over the state domain;
2. stable values, policy regret, and population moments under state/time/control
   and domain refinement;
3. stable repeated parameter perturbations at proposed calibration settings;
4. comparison against independent collocation paths beyond one initial state.

The existing `accepted_node_solution` flag tests consistency and feasibility;
it does not include shape or mesh convergence. No new experiment has been
marked `solve_bellman_converged(...).converged=True`.

## Reproduction and calibration API

From an environment with the project dependencies and a CUDA-enabled JAX:

```bash
PYTHONPATH=code/jax python tools/benchmark_bellman.py \
  --output output/solver_benchmarks/pchip_default \
  --interpolation pchip --diagnostics --repeats 2

PYTHONPATH=code/jax python tools/benchmark_bellman.py \
  --output output/solver_benchmarks/pchip_fine \
  --asset-nodes 61 --human-capital-nodes 49 --periods 140 \
  --control-nodes 15 --consumption-nodes 19 --refinement-steps 48 \
  --interpolation pchip --diagnostics --repeats 2
```

```python
from blinder_weiss import (
    BellmanConfig, benchmark_params, solve_bellman,
    simulate_cohort, cohort_moments,
)

config = BellmanConfig(compute_platform="gpu")
for leisure_weight in [1.0, 1.01, 0.99]:
    solution = solve_bellman(benchmark_params(leisure_weight=leisure_weight), config)
    cohort = simulate_cohort(solution, [2.0, 5.0, 8.0], [0.8, 1.0, 1.2],
                             weights=[1.0, 2.0, 1.0])
    moments = cohort_moments(cohort, participation_hours_threshold=0.02)
```

Keep discretization and cohort size fixed within an estimation stage. Changing
parameter values or initial states reuses compiled kernels. This interface
supports repeated derivative-free/finite-difference evaluations; it does not
promise valid derivatives through discrete policy selection.

The moments are model quantities, not automatically ACS/ATUS measurements.
`h` is active time and includes training; a threshold on `h` is not automatically
survey employment. Training `q`, effective earnings, calendar-age origin, time
endowment, wage units, survey universes, and weights need an explicit empirical
measurement specification. No ACS/ATUS data or estimated parameters were
invented for this benchmark.
