# GPU performance and policy accuracy

Measured on 2026-09-07 with the NVIDIA RTX PRO 6000 Blackwell Server Edition,
JAX/JAXlib/CUDA plugin 0.11.0, float64, and Python 3.13.11. The project manifest
specifies Python >=3.14 and JAX 0.10.1; these measurements describe the installed
runtime. They are not RTX 5070 Ti measurements.

Repeated solves are substantially faster. The latest bicubic run passes the
reported value-shape audit, but population accuracy and policy smoothness are
not yet accepted: the current human-capital domain truncates retirement paths,
large cohort discrepancies remain, and retirement consumption still oscillates.

## Implemented changes

- Backward induction, greedy recovery, and population rollout reuse compiled
  functions across model-parameter changes. Grid settings, device, and batch
  shape remain static. Search batches limit intermediate memory, and known
  curved/uniform grids use arithmetic cell lookup with node-rounding corrections.
- The consumption seed-grid minimum no longer restricts local refinement above
  the actual consumption floor. Optional bilinear consumption polish searches
  each asset interpolation interval conditional on hours and training.
- `value_interpolation="monotone_bicubic"` uses jointly limited bicubic Hermite
  continuation. Neighbor propagation and destination candidates address
  optimizer misses while retaining feasible, improving controls. Propagation
  accuracy matters for cubic shape; see the audit below.
- The earnings perspective has a bounded tangent extension for infeasible
  `q > h` trials near retirement. Feasible values and gradients are unchanged.
  Direct-path diagnostics reconstruct admissible multipliers when SLSQP returns
  stale duals, without relaxing stationarity or feasibility thresholds.
- `simulate_cohort` performs one batched greedy rollout; `cohort_moments` returns
  weighted model age profiles. Parameter values and initial states can change
  without recompilation at fixed configuration and cohort size.

The cache design follows [JAX compilation rules](https://docs.jax.dev/en/latest/201/jit.html).
Timings synchronize on completed results and include host transfer, following
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html).

## Measured speed and retained history

Coarse means 70 periods, 31 x 25 states, 11 x 11 x 15 control seeds, and 24 local
steps. Fine means 140 periods, 61 x 49 states, 15 x 15 x 19 seeds, and 48 steps.
Both currently use assets `[1e-4,35]` and log human capital `[-2,2.25]`.

| Repeated solve | Original | Improved bilinear | Historical sequential PCHIP | Latest bicubic |
|---|---:|---:|---:|---:|
| Coarse | 4.63–4.94 s | 0.73–0.79 s | 0.93 s | 1.08–1.09 s |
| Fine | 11.43 s | 3.96 s with consumption polish | 4.53 s | 4.78–4.79 s |

Coarse bilinear with conditional consumption polish took 0.82 s. Original
calls rebuilt a compiled closure each time; improved calls reuse compilation,
including after a 1% leisure-weight change. The original baseline is commit
`f331628`, with only the unused IPOPT import moved in its temporary benchmark copy.

Latest results are [`bicubic_adaptive_default/report.json`](../../output/solver_benchmarks/bicubic_adaptive_default/report.json)
and [`bicubic_exact_fine/report.json`](../../output/solver_benchmarks/bicubic_exact_fine/report.json).
The coarse run used 64 neighbor sweeps with tolerance `1e-13`; the fine run used
128 sweeps and tolerance zero. Both enabled destination candidates. First calls
including process setup took 9.65 s and 13.94 s respectively. The positive
coarse stopping tolerance is retained here as a measured configuration, not an
accepted propagation setting.

Historical coarse bilinear cohort rollout, checks, host transfer, and moments
took 0.72 s for 100 people and 0.81 s for 1,000 after compilation. Historical
coarse PCHIP solve + 256-person rollout + a 28-moment synthetic loss took
1.73–1.74 s, with a 12.26 s first call. Those are implementation-specific
measurements, not timings for the latest cubic calibration pipeline. The new
259-person validation rollouts, including their first compilation, took 5.92 s
coarse and 7.85 s fine. No warm cohort timing is inferred from those first calls.

## Current path and population evidence

[`calibration_grid_validation.json`](../../output/solver_benchmarks/calibration_grid_validation.json)
and its NPZ compare the same 256 weighted types: seed 125, assets uniform on
`[2,8]`, log human capital uniform on `[-0.25,0.25]`, and normalized weights
originally drawn on `[0.5,1.5]`. Three zero-weight reference probes share the
batch. States are compared at common quarter-year boundaries using exact
within-period transitions; controls and moments use common interval midpoints.

All three direct references pass the existing independent integration and KKT
checks at 48 Hermite–Simpson intervals, SLSQP objective tolerance `1e-11`. They
are accepted individual meshes, not mesh-converged truth. Reference state
comparisons independently integrate their linear controls with DOP853; control
comparisons use the direct controls at Bellman-period midpoints.

| Initial `(A,K)` | Utility shortfall, coarse → fine | Fine consumption RMSE | Fine hours RMSE | Fine training RMSE |
|---|---:|---:|---:|---:|
| Baseline `(5,1)` | 0.007759 → 0.001452 | 0.002021 | 0.000827 | 0.001416 |
| Low `(2,0.8)` | 0.012261 → 0.001848 | 0.001825 | 0.000802 | 0.002209 |
| High `(8,1.2)` | 1.325745 → 1.309049 | 0.012589 | 0.125744 | 0.126012 |

Utility shortfall is reference utility minus realized greedy utility. Fine
asset/human-capital RMSE is 0.01334/0.00406 for the baseline, 0.01230/0.00423 for
the low type, and 0.15115/0.05601 for the high type. Baseline reference utility
is -99.4888795; the low and high references are -118.5130407 and -83.9288792.

**The high-type discrepancy has a concrete domain cause.** Its direct retired
path reaches `log K=-3.28086` (`K=0.037596`). Both Bellman paths contact the
artificial `log K=-2` floor around age 56 and then maintain training near
`q=delta/a=0.22727`; terminal hours remain about 0.28–0.30 instead of retirement.
The fine cohort spends 10.9% of weighted sampled state-age observations in its
lowest human-capital cell. A successful domain-containment flag therefore does
not establish that the computational boundary is economically harmless.

Changes in weighted age profiles from coarse to fine are:

| Model moment | RMS change | Maximum absolute change |
|---|---:|---:|
| Consumption | 0.007629 | 0.032643 |
| Active time `h` | 0.007574 | 0.028130 |
| Training time `q` | 0.007421 | 0.028130 |
| Participation, `h > 0.02` | 0.020061 | 0.055738 |
| Earnings | 0.014891 | 0.041802 |

Participation changes correspond to 2.01 percentage points RMS and 5.57 points
maximum. Average profiles conceal much larger individual discrepancies: maximum
asset change is 11.836 for type 200; type 177 reaches changes of 3.476 in human
capital, 0.477 in consumption, 0.583 in hours, and 0.588 in training. The largest
individual utility change is 0.60314 for type 56. Type indices are zero-based
positions in the saved fixed cohort.

Types 177 and 200 follow schooling/borrowing-floor paths on the coarse grid,
but choose a different initial training pattern on the fine grid and later
contact the human-capital floor at ages 50.5 and 57. Their fine recovered-value
minus realized-utility gaps are -0.21636 and 0.03151. Across all 256 types the
maximum absolute gap is 0.33814 coarse and 0.21636 fine. These observations do
not establish equivalent optima or population convergence. Grid/time/control
settings and propagation tolerance changed together; neither run expands the
domain, so the comparison cannot isolate those effects.

## Shape and remaining smoothness limits

The [`bicubic_exact_shape_audit.json`](../../output/solver_benchmarks/bicubic_exact_shape_audit.json)
audit evaluates 46,513 full-domain queries at each of 141 ages: 6,558,333 points.
All derivatives are finite, with zero asset or human-capital derivative counts
below `-1e-8`. Minimum derivatives are 0.00040718 in assets and approximately
-1.78e-15 in log human capital. Bicubic cell constraints have maximum cross-slope
residual 4.34e-16; other reported shape residuals are zero.

The zero-tolerance propagation setting matters: stopping at a positive `1e-13`
tolerance left tiny optimizer-seed differences that cubic derivatives could
amplify near retirement. The fine audit uses `neighbor_policy_tolerance=0.0`.
Historical sequential two-dimensional PCHIP did not satisfy the same shape
check: its fine grid reached `dV/dlog K` near -0.0229 at age 65. It remains an
accuracy/speed comparison, not the accepted shape construction.

Value monotonicity does not settle policy smoothness. On the saved baseline
cohort probe, retirement consumption Euler RMS residual is 0.00807 per year
coarse and 0.00919 fine; the fine maximum is 0.02556. The continuous retirement
condition is `log(c[n+1]/c[n])/dt=(r-rho)/(1-consumption_power)=0.01`.

An [independent scalar audit](../../output/solver_benchmarks/bicubic_adaptive_fine/retirement_scalar_audit.json)
of the earlier fine cubic table reoptimized consumption at 12 saved retired
states across feasible asset-cell intervals. Its largest consumption correction
was 9.82e-6 and largest objective gain 2.69e-11; the selected-pair Euler RMS
remained about 0.02221. The [reconstruction audit](../../output/solver_benchmarks/bicubic_adaptive_fine/retirement_reconstruction_audit.json)
identifies sensitivity to limited asset slopes in flat retirement regions.
This evidence points beyond inadequate scalar consumption optimization. Slope
alternatives remain under investigation; none is documented here as a completed
smoothness fix.

Further acceptance requires a domain that contains retirement decay, renewed
heterogeneous comparisons after domain expansion, stable values and population
moments under refinement, and resolution of the consumption oscillations. The
`accepted_node_solution` flag checks consistency and feasibility; it does not
include shape, domain independence, or mesh convergence. No new run has been
marked `solve_bellman_converged(...).converged=True`.

## Reproduction and empirical calibration

Reproduce the measured narrow-domain fine run and fixed-cohort comparison:

```bash
PYTHONPATH=code/jax python tools/benchmark_bellman.py \
  --output output/solver_benchmarks/bicubic_exact_fine \
  --asset-nodes 61 --human-capital-nodes 49 --periods 140 \
  --control-nodes 15 --consumption-nodes 19 --refinement-steps 48 \
  --interpolation monotone_bicubic --neighbor-destinations \
  --neighbor-sweeps 128 --neighbor-tolerance 0 --diagnostics --repeats 2

PYTHONPATH=code/jax python tools/validate_calibration_grid.py \
  --folders bicubic_adaptive_default bicubic_exact_fine \
  --output output/solver_benchmarks/calibration_grid_validation.json
```

Keep configuration and cohort size fixed within an estimation stage. The
`solve_bellman`, `simulate_cohort`, and `cohort_moments` API supports repeated
finite-difference or derivative-free evaluations; it does not promise valid
derivatives through discrete policy selection.

Validated 2024 ACS and ATUS extracts are now downloaded and remotely backed up:
see the [IPUMS acquisition and measurement README](../../data/ipums/README.md).
This does not yet supply an empirical calibration. Survey universes, calendar
ages, time endowment, active time versus employment, training coverage, wage
units, and survey weighting still need an explicit measurement specification.
No empirical moments or estimated parameters are claimed by these benchmarks.
