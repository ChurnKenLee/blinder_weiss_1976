# GPU performance and policy accuracy

Measured on 2026-09-07 with the NVIDIA RTX PRO 6000 Blackwell Server Edition,
JAX/JAXlib/CUDA plugin 0.11.0, float64, and Python 3.13.11. The project manifest
specifies Python >=3.14 and JAX 0.10.1; these measurements describe the installed
runtime. They are not RTX 5070 Ti measurements.

The latest refinement (2026-09-08) reduces retirement consumption Euler jitter
by 37–54% across three independent reference households. It uses 121 asset
nodes on the padded domain and takes 9.37 s per repeated GPU solve, compared
with 6.94–6.95 s for the preceding 61-node run. Full population, policy and
time-grid convergence is not established.

## Measured retirement smoothness improvement

The new configuration keeps 140 periods, 79 log-human-capital nodes, 15×15×19
control seeds, 48 local steps, the padded `[-4.5,2.25]` log-K domain and physical
asset coordinate. Asset nodes rise from 61 to 121; the neighboring-policy
propagation cap also rises from 160 to 240, at zero gain tolerance. This is a
joint grid/search-budget comparison, not an isolated estimate of grid effects.

| Initial (A,K) | Retirement Euler RMS, previous → refined | Reduction | Refined utility shortfall |
|---|---:|---:|---:|
| (5,1) | 0.008915 → 0.004131 | 53.7% | 0.001412 |
| (2,0.8) | 0.008081 → 0.003851 | 52.3% | 0.001837 |
| (8,1.2) | 0.004559 → 0.002886 | 36.7% | 0.000706 |

The baseline maximum absolute Euler residual falls from 0.02470 to 0.01198 per
model year. The statistic uses adjacent retired periods away from the numerical
asset floor, and compares consumption growth with `(r-rho)/(1-consumption_power)`.
It does not penalize economic work/retirement switches. `retirement_euler_diagnostics`
now provides this reproducible measure for individual or batched paths.
The low type's realized utility declines by 0.0000120; baseline and high-type
utility improve by 0.0001296 and 0.0000308. Direct references remain accepted
48-interval paths, not mesh-converged truth.

The refined full-domain shape audit checks 21,227,973 points: all derivatives
are finite, none is below `-1e-8`, and the largest bicubic constraint residual
is 1.42e-14. See the [smoothness comparison](../../output/solver_benchmarks/asset_refinement_smoothness.json),
[policy checkpoint](../../output/solver_benchmarks/bicubic_asset121_fine/report.json),
and [shape audit](../../output/solver_benchmarks/bicubic_asset121_shape_audit.json).
An attempted CRRA asset-coordinate transformation was exact on analytic
retirement values but worsened full-model utility and did not improve jitter;
it was removed from production and preserved only as a reproducible
[rejected experiment](../../output/solver_benchmarks/experiments/utility_coordinate/).

On the historical fixed 256-type weighted cohort, the paired refinement check
reduces the weighted Euler RMS on identical retired time pairs by **57.9%**:
0.006885 to 0.002897. The maximum absolute recovered-value minus realized-utility
gap falls from 0.283608 to 0.035358 (87.5%). Weighted realized utility rises by
0.005163, with 94.94% of population mass improving. This cohort uses the prior
seed-125 initial states and weights; it is distinct from the new correlated
continuum mixture with floor and point atoms.

Moment RMS changes are consumption 0.001338, hours 0.001044, training 0.001001,
earnings 0.003169 and participation 0.003595 (0.36 percentage points). Individual
tails remain important: type 33 loses 0.009084 utility, while type 158 changes
its asset path by as much as 11.235 and gains 0.278917 utility. These results
support the refinement but do not establish population mesh convergence. See
[paired population validation](../../output/solver_benchmarks/asset_refinement_population.json),
[population Euler diagnostics](../../output/solver_benchmarks/asset_refinement_population_smoothness.json),
and [tail summary](../../output/solver_benchmarks/asset_refinement_population_tail_summary.json).

Reproduce the refined solution and the comparison:

```bash
PYTHONPATH=code/jax python tools/benchmark_bellman.py \
  --output output/solver_benchmarks/bicubic_asset121_fine \
  --asset-nodes 121 --human-capital-nodes 79 --periods 140 \
  --log-human-capital-minimum -4.5 \
  --control-nodes 15 --consumption-nodes 19 --refinement-steps 48 \
  --interpolation monotone_bicubic --neighbor-destinations \
  --neighbor-sweeps 240 --neighbor-tolerance 0 --diagnostics --repeats 2
PYTHONPATH=code/jax python tools/validate_bellman_smoothness.py \
  --output output/solver_benchmarks/asset_refinement_smoothness.json
```

For the new population backends and synthetic calibration pilot, see
[CONTINUUM_CALIBRATION.md](CONTINUUM_CALIBRATION.md).

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

The historical coarse grid has 70 periods, 31 x 25 states, 11 x 11 x 15 control
seeds, and 24 local steps. Historical fine has 140 periods, 61 x 49 states,
15 x 15 x 19 seeds, and 48 steps. Both use assets `[1e-4,35]` and log human
capital `[-2,2.25]`.

| Repeated solve | Original | Improved bilinear | Historical sequential PCHIP | Jointly monotone bicubic |
|---|---:|---:|---:|---:|
| Narrow coarse | 4.63–4.94 s | 0.73–0.79 s | 0.93 s | 1.08–1.09 s |
| Narrow fine | 11.43 s | 3.96 s with consumption polish | 4.53 s | 4.78–4.79 s |
| Padded fine, 61 x 79 states | — | — | — | 6.94–6.95 s |

Padded fine retains 140 periods, the fine control search, and assets `[1e-4,35]`,
but extends log human capital to `[-4.5,2.25]`. Its spacing is 0.08654, close to
0.08854 on narrow fine. Coarse bilinear with conditional consumption polish took
0.82 s. Original calls rebuilt a compiled closure each time; improved calls reuse
compilation, including after a 1% leisure-weight change. The original baseline
is commit `f331628`, with only the unused IPOPT import moved in its temporary
benchmark copy.

Saved reports are [narrow coarse](../../output/solver_benchmarks/bicubic_adaptive_default/report.json),
[narrow fine](../../output/solver_benchmarks/bicubic_exact_fine/report.json), and
[padded fine](../../output/solver_benchmarks/bicubic_padded_fine/report.json).
Their neighbor sweep caps are 64, 128, and 160; tolerances are `1e-13`, zero,
and zero. All enable destination candidates. First calls including process
setup took 9.65, 13.94, and 16.44 s respectively. The positive coarse tolerance
is a measured historical setting, not an accepted propagation setting.

Historical coarse bilinear cohort rollout, checks, host transfer, and moments
took 0.72 s for 100 people and 0.81 s for 1,000 after compilation. Historical
coarse PCHIP solve + 256-person rollout + a 28-moment synthetic loss took
1.73–1.74 s, with a 12.26 s first call. These are implementation-specific
measurements. The latest padded 259-person validation rollout took 7.76 s,
including its first compilation; no warm cohort timing is inferred from it.

## Current path and population evidence

[`calibration_domain_validation.json`](../../output/solver_benchmarks/calibration_domain_validation.json)
and its NPZ compare narrow fine with padded fine on the same 256 weighted types:
seed 125, assets uniform on `[2,8]`, log human capital uniform on
`[-0.25,0.25]`, and normalized weights originally drawn on `[0.5,1.5]`.
Three zero-weight reference probes share the batch. States are compared at
common quarter-year boundaries using exact within-period transitions; controls
and moments use common interval midpoints.

All three direct references pass the existing independent integration and KKT
checks at 48 Hermite–Simpson intervals, SLSQP objective tolerance `1e-11`. They
are accepted individual meshes, not mesh-converged truth. Reference state
comparisons independently integrate their linear controls with DOP853; control
comparisons use the direct controls at Bellman-period midpoints.

| Initial `(A,K)` | Utility shortfall, narrow → padded fine | Padded consumption RMSE | Padded hours RMSE | Padded training RMSE |
|---|---:|---:|---:|---:|
| Baseline `(5,1)` | 0.001452 → 0.001542 | 0.002036 | 0.000912 | 0.001426 |
| Low `(2,0.8)` | 0.001848 → 0.001825 | 0.001636 | 0.000816 | 0.002106 |
| High `(8,1.2)` | 1.309049 → 0.000737 | 0.001506 | 0.000387 | 0.000617 |

Utility shortfall is reference utility minus realized greedy utility. Padded
asset/human-capital RMSE is 0.00714/0.00233 for the baseline, 0.00440/0.00093 for
the low type, and 0.00630/0.00029 for the high type. Reference utilities are
-99.4888795, -118.5130407, and -83.9288792 respectively.

**The wider domain resolves the identified retirement boundary distortion.**
The high-type direct path reaches `log K=-3.28086`. Narrow Bellman paths
contacted the artificial `log K=-2` floor around age 56 and maintained training
near `q=delta/a=0.22727`, with terminal hours about 0.28. Padded fine ends with
`h=q=0` and reduces this type's hours RMSE from 0.12574 to 0.000387. The padded
cohort's minimum log human capital is -3.72827, leaving 0.77173 to the new floor;
none of its sampled states occupies either outer human-capital cell. The narrow
cohort had 10.9% of weighted state-age observations in its lowest cell. Asset
floor contact remains economically allowed; maximum assets are 17.64, well below
35. Domain containment alone had concealed the old boundary distortion.

Weighted cohort utility improves from -99.48057 to -98.94652, a gain of 0.53405.
The corresponding changes in age profiles are substantial:

| Model moment | RMS change, narrow → padded fine | Maximum absolute change |
|---|---:|---:|
| Consumption | 0.031938 | 0.048335 |
| Active time `h` | 0.051729 | 0.089379 |
| Training time `q` | 0.049320 | 0.080703 |
| Participation, `h > 0.02` | 0.236183 | 0.395438 |
| Earnings | 0.062412 | 0.129218 |

Participation changes are 23.62 percentage points RMS and 39.54 points maximum;
its time-average declines 18.36 points. These are measured domain corrections,
not estimates of the remaining padded-grid error. Maximum individual changes
are 11.784 in assets, 3.508 in human capital, 0.504 in consumption, 0.590 in hours,
and 0.602 in training. Maximum individual utility change is 2.07668.

The earlier [narrow coarse/fine comparison](../../output/solver_benchmarks/calibration_grid_validation.json)
had smaller average moment changes but large tails: maximum asset difference
11.836 for type 200, and human-capital/hours/training differences
3.476/0.583/0.588 for type 177. Types 56, 177, and 200 all now retire with
`h=q=0`; padding improves their utility by 1.58803, 1.79418, and 1.23176.
Their padded recovered-value minus realized-utility gaps are 0.02915, 0.05067,
and -0.01516. Indices refer to zero-based positions in the saved cohort.

Across all 256 padded types the weighted mean value gap is -0.00407, but the
maximum absolute gap remains 0.28361 (narrow fine: 0.21636). The boundary failure
is resolved for the inspected paths; population convergence is not established.
Further refinement on the padded domain must distinguish remaining search and
interpolation error from economically different, nearly optimal paths.

## Shape and remaining smoothness limits

The [padded shape audit](../../output/solver_benchmarks/bicubic_padded_shape_audit.json)
evaluates 75,433 full-domain queries at each of 141 ages: 10,636,053 points.
All derivatives are finite, with zero asset or human-capital derivative counts
below `-1e-8`. Minimum derivatives are 0.00040718 in assets and approximately
-1.21e-15 in log human capital. Bicubic cell constraints have maximum cross-slope
residual 9.09e-13; other reported shape residuals are zero. The earlier
[narrow fine audit](../../output/solver_benchmarks/bicubic_exact_shape_audit.json)
also passed at 6,558,333 points.

The zero-tolerance propagation setting matters: stopping at a positive `1e-13`
tolerance left tiny optimizer-seed differences that cubic derivatives could
amplify near retirement. Both fine audits use `neighbor_policy_tolerance=0.0`.
Historical sequential two-dimensional PCHIP did not satisfy the same shape
check: its fine grid reached `dV/dlog K` near -0.0229 at age 65.

Value monotonicity does not settle policy smoothness. On the saved baseline
probe, retirement consumption Euler RMS residual is 0.00891 per year on padded
fine, versus 0.00919 on narrow fine; the padded maximum is 0.02470. These use
adjacent retired periods (`h < 1e-6`) away from the borrowing floor. The
continuous retirement condition is
`log(c[n+1]/c[n])/dt=(r-rho)/(1-consumption_power)=0.01`.

An [independent scalar audit](../../output/solver_benchmarks/bicubic_adaptive_fine/retirement_scalar_audit.json)
of an earlier fine cubic table reoptimized consumption at 12 saved retired
states across feasible asset-cell intervals. Its largest consumption correction
was 9.82e-6 and largest objective gain 2.69e-11; the selected-pair Euler RMS
remained about 0.02221. The [reconstruction audit](../../output/solver_benchmarks/bicubic_adaptive_fine/retirement_reconstruction_audit.json)
identifies sensitivity to limited asset slopes in flat retirement regions.
This evidence points beyond inadequate scalar consumption optimization.
No tested slope alternative has established a smoothness fix.

Further acceptance requires stable values, paths, and population moments under
refinement within the padded domain, plus resolution of the consumption
oscillations. The `accepted_node_solution` flag checks consistency and
feasibility; it does not include shape, domain independence, or mesh convergence.
No new run has been marked `solve_bellman_converged(...).converged=True`.

## Reproduction and empirical calibration

Reproduce the measured padded run and comparison with the saved narrow fine run:

```bash
PYTHONPATH=code/jax python tools/benchmark_bellman.py \
  --output output/solver_benchmarks/bicubic_padded_fine \
  --asset-nodes 61 --human-capital-nodes 79 --periods 140 \
  --log-human-capital-minimum -4.5 \
  --control-nodes 15 --consumption-nodes 19 --refinement-steps 48 \
  --interpolation monotone_bicubic --neighbor-destinations \
  --neighbor-sweeps 160 --neighbor-tolerance 0 --diagnostics --repeats 2

PYTHONPATH=code/jax python tools/validate_calibration_grid.py \
  --folders bicubic_exact_fine bicubic_padded_fine \
  --output output/solver_benchmarks/calibration_domain_validation.json
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
