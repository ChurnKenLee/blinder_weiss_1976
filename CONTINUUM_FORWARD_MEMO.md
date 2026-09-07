# Continuum population approximation and forward distribution solver

Implementation memo for the main thread, requested by the user on 2026-09-07.
This is a design handoff; the implementation and acceptance checks below remain
main-thread work.

Please add a **deterministic continuum population backend** to the JAX solver,
with both a quadrature reference and a conservative forward distribution solver.
Keep the current solver improvements active. The existing 256-person run is a
performance benchmark; population resolution should now be determined by
convergence of moments and calibration results.

1. **Define the population as a probability measure.**

   Use the existing state coordinates \(x=(A,y)\), where \(y=\log K\), and an
   age-specific distribution \(\mu_a\). Allow continuous heterogeneity, discrete
   types, and mass concentrated on constraints.

   For example, an asset-boundary component can be represented as

   \[
   \mu_a(dA,dy)
   =
   f_a(A,y)\,dA\,dy
   +\delta_{\underline A}(dA)\,\beta_a(dy)
   +\mu_a^{\mathrm{other}}(dA,dy).
   \]

   Here \(\beta_a\) is mass distributed along the asset constraint; other
   components can represent additional singular support. Constraints can
   generate point masses, so a smooth density alone is insufficient.
   [Achdou et al.](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf)

   Audit which existing bounds are economic constraints and which are numerical
   cutoffs. In particular, establish whether the current asset floor represents
   the economic borrowing constraint or approximates it with a positive epsilon.

2. **Implement quadrature as the reference continuum approximation.**

   Specify an explicit joint initial distribution \(F_0(A,K)\), including
   correlation and any initial atoms. Keep assumed or synthetic distributions
   clearly identified until empirical inputs support them.

   Construct deterministic nodes and nonnegative normalized weights, then reuse
   the existing cohort rollout:

   \[
   M_a(\theta)\approx
   \sum_j w_j(\theta)\,
   g_a\!\left(x_a(x_{0j};\theta),\theta\right).
   \]

   Use the same initial distribution, feasible policies, transition equations,
   and moment definitions in both backends. Refine quadrature—for example,
   \(8^2,16^2,32^2,64^2\) nodes for two continuous dimensions—to establish a
   reference tolerance.

   Initial-state heterogeneity can share a policy solution. Preference or
   technology types generally need their own policy solutions and explicit
   population weights.

3. **Build a forward operator matched to the current lifecycle dynamics.**

   For the deterministic model, the interior density satisfies

   \[
   \partial_a f
   =-\partial_A(b_Af)-\partial_y(b_yf),
   \qquad b_y=b_K/K.
   \]

   Start with a conservative approximation to the existing discrete transition
   map \(T_a(x;\theta)\). At each age, recover feasible controls on the
   distribution grid, evaluate their destinations, and construct nonnegative
   transfer weights:

   \[
   P_a[i,j]\ge0,\qquad
   \sum_jP_a[i,j]=1,\qquad
   p_{a+\Delta a}=P_a^\top p_a.
   \]

   Store **probability masses** in \(p\). On a nonuniform grid, density values
   require cell-volume factors; they cannot be propagated as though they were
   masses. Likewise, converting densities between \(K\) and \(\log K\) requires
   the Jacobian.

   This is a deterministic distribution propagation method of the kind used in
   non-stochastic simulation.
   [Young](https://www.wouterdenhaan.com/suite/finalversion-young.pdf)
   The continuous-time counterpart uses the adjoint of the policy-induced
   generator.
   [Numerical appendix](https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf)

   Build the transport operator separately from the cubic value interpolator.
   Cubic interpolation and its derivatives do not generally supply nonnegative
   probability weights. Any claimed discrete adjoint identity must refer to this
   explicit transport operator. Discounting belongs in valuation, not population
   transport.

4. **Handle boundary mass explicitly.**

   Use separate boundary-face and corner mass compartments, or an equivalent
   conservative representation that distinguishes exact constraint mass from
   nearby interior mass.

   Apply these rules:

   - When the feasible transition reaches an economic constraint, transfer the
     associated mass to that boundary component.
   - While constrained, let mass move along the boundary according to the
     remaining state dynamics.
   - Permit mass to leave the boundary when the feasible policy points into the
     interior. A borrowing constraint is not automatically absorbing.
   - Keep first-cell interior mass separate from exact boundary mass. Ordinary
     interpolation onto an endpoint must not manufacture an economic atom.
   - Account for transfers between intersecting faces and corners exactly once.

   Boundary remapping needs its own treatment; a generic four-corner
   interpolation stencil is insufficient if it confounds interior and boundary
   mass. Any numerical spreading of concentrated mass must be measured under
   refinement.

   Artificial grid limits require separate diagnostics. Record attempted exits
   and affected mass before any roundoff correction. Material exits should
   trigger domain expansion and renewed policy checks. Reflection, clipping, or
   absorption at an artificial upper bound would change the model.

   Preserve mass through retirement and terminal-age accounting unless the
   model explicitly contains mortality or another exit mechanism. Constraints
   on controls, such as zero work or binding training limits, should enter
   moment calculations directly; they do not automatically create
   state-distribution atoms.

5. **Integrate the backend into JAX and calibration.**

   Expose a selectable population backend, retaining cohort simulation for
   comparison. Return moments, optional distribution snapshots, and
   conservation/boundary diagnostics through a consistent interface.

   Use fixed-shape arrays, compiled scans, and local scatter-add transfers or
   sparse operators. Avoid a dense state-to-state matrix. Keep the distribution
   grid independently configurable from the Bellman grid, and preserve
   compilation caching across parameter changes.

   Compute age-specific moments by integrating over every population component,
   including boundary mass. Match existing timing conventions for controls,
   earnings, and states. Keep cohort evolution distinct from cross-sectional
   age pooling; the latter requires explicit age or cohort weights.

   Participation thresholds and policy switches can still make losses
   irregular. If gradients are used, validate directional derivatives against
   finite differences across several step sizes.

6. **Require numerical acceptance before making it the calibration default.**

   Include meaningful checks for:

   - Conservation and positivity over the full lifecycle. Suggested float64
     targets are row-sum errors below \(10^{-12}\) and total-mass drift below
     \(10^{-10}\), with actual errors reported.
   - Identity and known-translation dynamics, including numerical diffusion
     under refinement.
   - Mass reaching a constraint, remaining there, moving along it, and
     subsequently leaving it.
   - Initial boundary atoms, corner transfers, and nearby interior mass that
     must remain distinct.
   - Detection of material exits through artificial boundaries.
   - The operator identity

     \[
     \langle P^\top p,\phi\rangle=\langle p,P\phi\rangle.
     \]

   - Agreement with refined quadrature on hours, consumption, earnings,
     participation, state moments, and constraint mass.

   Refine quadrature, the distribution grid, the Bellman grid, and the time step
   separately to identify each source of error. Compare CDFs and integrated
   moments where atoms make density comparisons misleading. Check
   representative parameter perturbations as well as the baseline.

   Set calibration tolerances relative to empirical uncertainty and the loss
   changes the optimizer must resolve. Do not hide conservation failures
   through repeated normalization.

7. **Deliver a reproducible comparison and working calibration path.**

   Provide the two continuum backends, boundary diagnostics, focused tests, and
   a benchmark comparing the existing cohort calculation, refined quadrature,
   and forward transport.

   Measure the complete solve-and-moment-loss calculation, including policy
   recovery and operator construction, with compilation reported separately.
   Document the accuracy/runtime tradeoff and remaining limitations. Make
   forward transport the default only after it meets the recorded acceptance
   criteria.

   Follow the project's notebook editing and autosave requirements for
   implementation, and verify successful remote persistence.
