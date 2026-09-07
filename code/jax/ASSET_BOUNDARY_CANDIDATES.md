# Candidate treatments of the zero-assets boundary

**To:** Main solver thread  
**From:** Side discussion with the user  
**Date:** 2026-09-07  
**Status:** Proposed experiments; no solver or notebook changes made here.

## User request

The user asked whether a logit mapping, or a related transformation, could
handle the minimum-assets constraint more effectively than clipping. They
asked for this memo so the main thread can evaluate alternative approaches as
candidates alongside its work on smooth policies and efficient JAX calibration.

The recommendation is to distinguish **constraint enforcement**, **state-grid
coordinates**, and **terminal utility**. A smooth transformation can help one
of these without resolving the others. None of the candidates below is a
predetermined replacement for the current method.

## Baseline to revalidate before experimenting

The project documentation describes a borrowing constraint enforced mainly
through maximum feasible consumption at within-period checkpoints. Controls
whose transitions leave the represented state domain are rejected. Clipping
also bounds interpolation queries and optimizer coordinates; this is different
from accepting an infeasible transition and resetting negative assets to zero.

The discussed consumption parameterization is

\[
c=c_{\min}+s\,[\bar c(A,K,h,q)-c_{\min}],\qquad s\in[0,1].
\]

An explicit endpoint `s=1` permits the capacity constraint to bind. Preserve
that possibility when comparing optimizer parameterizations. Inspect the
current code first: the main thread is developing the solver concurrently.

## Candidates

### 1. Explicit zero-assets state and constrained branch

Represent the economic borrowing limit `A=0` at intermediate ages. Evaluate a
binding-constraint candidate alongside interior control candidates, then select
by the same feasible Bellman objective. An active-set or complementarity
formulation is another implementation option; compare its cost with explicit
branch evaluation in JAX batches.

This permits a true borrowing-constrained spell without sending an optimizer
coordinate to infinity. It also separates a genuine regime boundary from
numerical irregularity in an interior policy.

At `A=0`, instantaneous feasibility requires

\[
\dot A(0)=K\,\phi(h,q)-c\geq0,
\qquad \phi(h,q)=h\,g(q/h).
\]

Include this initial-drift condition when evaluating an exact boundary state.
For constant controls and changing human capital, endpoint feasibility alone
does not establish feasibility throughout the period. Check the exact asset
trajectory, including any interior minimum, and compare the existing checkpoint
approximation with a tighter or analytic treatment. A binding candidate need
not imply that assets remain identically zero throughout a finite period.

**Hypothesis:** Better boundary fidelity and fewer optimizer misses near
borrowing contact, with manageable branching cost. This does not imply globally
smooth policies across changes in the active constraint.

### 2. Transformed asset grid retaining zero as a finite point

Compare the existing curved asset grid with

\[
u=\log(1+A/a_{\mathrm{scale}}),\qquad
A=a_{\mathrm{scale}}\,[\exp(u)-1],
\quad a_{\mathrm{scale}}>0.
\]

A uniform grid in `u` concentrates asset nodes near zero and contains `A=0`
exactly at `u=0`. Vary the scale and compare alternatives at equal node counts
and equal wall time.

Keep transitions and feasibility calculations in physical assets initially;
transform the destination only for value reconstruction. This isolates the
coordinate experiment from changes in the dynamics. If interpolation is built
in `u`, convert derivatives consistently when testing economic conditions:
`V_A = V_u / (a_scale + A)`.

**Hypothesis:** More useful resolution near the borrowing boundary without an
infinite-coordinate endpoint. The coordinate change still requires explicit
constraint enforcement, and it may sacrifice accuracy farther from zero.

### 3. Logit, exponential, or softplus as controlled comparisons

For example,

\[
A=A_{\min}+(A_{\max}-A_{\min})\,\sigma(z)
\]

maps finite `z` into an open interval. Its derivative approaches zero at either
boundary. Likewise, `A=A_min+exp(z)` and a shifted softplus exclude the lower
endpoint for finite `z`. These formulas and endpoint behavior are documented in
[Stan's constraint-transform reference](https://mc-stan.org/docs/reference-manual/transforms.html).

Evaluate these as **interior parameterizations with explicit endpoint
candidates**, or as deliberately approximate interior-only methods with a
reported boundary tolerance. A sigmoid applied to the consumption fraction
also excludes `s=1`; retain a separate binding candidate if trying that route.

Measure conditioning near the boundary: a small gradient with respect to `z`
can result from a vanishing transform derivative while economically relevant
stationarity error remains. For a bounded logit, the upper bound may be only a
numerical domain choice, so include upper-domain sensitivity as well.

**Hypothesis:** Smoother optimizer coordinates might reduce projected-step
artifacts for interior choices. The transform alone is unlikely to be the best
way to represent an optimal spell exactly at the borrowing limit.

### 4. Next-period assets as a control coordinate

For fixed hours and training, the exact transition has the form

\[
A'=D(A,K,h,q)-I_c\,c.
\]

Explore choosing feasible next-period assets (including the boundary endpoint)
and recovering consumption analytically. Compare this with the existing
consumption-capacity coordinate and conditional consumption optimizer.

The feasible interval must incorporate positive consumption, the represented
state domain, and the within-period borrowing constraint. Choosing `A' >= 0`
by itself is insufficient. Endogenous-grid or complementarity methods can be
considered as larger follow-up experiments if their assumptions are justified
for the joint hours/training problem.

**Hypothesis:** A coordinate directly aligned with the active asset constraint
may improve boundary search. Its benefits are empirical; nonconcavity in the
remaining controls still requires safeguards.

## Separate terminal singularity from intermediate borrowing contact

For the benchmark negative-power bequest, `B(0)=-infinity`. This does not imply
that the value of arriving at zero assets at an earlier age must be infinite:
positive consumption and positive terminal assets may remain feasible through
future earnings.

Test an exact intermediate zero node while retaining analytic terminal bequest
handling. Do not simply set the numerical asset floor to zero and pass a terminal
table containing `-inf` through interpolation or cubic derivative preparation.
The terminal branch must bypass unsuitable preprocessing as well as unsuitable
value evaluation. Verify finite nonterminal values and gradients where feasible,
and correct rejection where no feasible positive-consumption path exists.

## Comparison and acceptance checks

Run focused boundary tests first, then full solves and cohort simulations.
Keep other numerical choices fixed initially so effects can be attributed to
the boundary treatment.

1. **Economic fidelity:** Distinguish the true economic floor from a positive
   numerical floor. Compare exact zero with a sequence of positive numerical
   floors; check borrowing-contact and release times.
2. **Feasibility:** Inspect the continuous within-period asset minimum or an
   independently refined check, positive consumption, and genuine domain exits.
   Report roundoff separately from material violations.
3. **Optimality and consistency:** Preserve Bellman identities; measure policy
   regret and the appropriate interior or boundary optimality conditions.
   Do not replace a real binding inequality with an interior Euler equality.
4. **Smoothness:** Evaluate within regimes and separately at borrowing contact,
   release, schooling, and retirement transitions. Preserve genuine economic
   kinks; do not accept a method merely because plotted curves look smoother.
5. **Independent accuracy:** Compare paths and utility with credible direct
   collocation references, including initial states near zero, and refine time,
   state grids, control search, and domain limits.
6. **Calibration behavior:** Compare weighted cohort moments and nearby
   parameter perturbations. Check that boundary approximation does not materially
   shift the moments targeted by ACS/ATUS calibration.
7. **GPU cost:** Record cold compilation, synchronized warm solves, complete
   solve-plus-cohort evaluations, peak memory, and compilation reuse across
   parameter changes. Retain the current implementation as a benchmark.

## Suggested experiment order

Start with an explicit boundary candidate and the terminal/intermediate
separation. Compare the transformed grid independently, then combine promising
variants. Use logit/softplus plus explicit endpoints as a conditioning comparison.
Try next-assets coordinates if the preceding tests identify control search as
the remaining bottleneck.

These experiments complement the main thread's interpolation work. Boundary
handling and continuation reconstruction should eventually be assessed together,
but a successful result in one does not establish correctness of the other.
