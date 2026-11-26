# Viva guide

## What happens in one optimisation (follow this path through the code)

1. `params.calibrated_params()` loads the 737-800 data and fits three constants so the model
   reproduces the real aircraft: wing-mass coefficient `k_w`, tank coefficient `k_f`, effective TSFC `c_T`.
2. `model.AircraftModel` computes range, the constraints and their analytic gradients for a
   design x = (S, AR, V, W_fuel, W_i).
3. `problems.ReducedProblem` removes W_fuel using the weight build-up (W_fuel = W_i − everything
   else) and scales all variables to about 1. Every point the solver visits is then a
   weight-consistent aircraft.
4. `alm.solve_alm` runs the augmented Lagrangian loop. Each subproblem is an unconstrained
   minimisation solved by `unconstrained.minimize` (BFGS by default, with Armijo backtracking).
5. `reference.solve_slsqp` solves the original 5-variable problem with SciPy as an independent
   check. `reference.kkt_check` re-estimates the multipliers and tests the KKT and second-order
   conditions at our answer.

## Likely questions and short answers

**Why minimise −ln R instead of maximising R?** ln is monotonic, so the optimum is the same. The
product in Breguet becomes a sum (simpler gradients), and every multiplier becomes an elasticity:
% range per % relaxation of a constraint.

**Why an augmented Lagrangian and not a pure penalty?** A pure penalty needs μ → ∞, which makes the
subproblem ill-conditioned. The multiplier update λ ← max(0, λ + μg) lets μ stay moderate, and at
the end λ are the KKT multipliers.

**Why eliminate W_fuel?** It appears linearly in the weight equation, so it can be computed
exactly. Without elimination an iterate could drift to W_i − W_fuel → 0 (an aircraft made only
of fuel) where ln(W_i/(W_i − W_fuel)) → ∞. SLSQP still solves the full 5-variable problem, so it
also checks the elimination.

**Why is the Mach limit always active?** Range ∝ V·L/D and the model has no wave drag, so going
faster never hurts. The Mach cap stands in for wave drag; this is a stated limitation.

**Why doesn't c_T change the design?** In −ln R, c_T only adds the constant ln c_T. Constants do
not move the minimiser, so range scales as 1/c_T and the geometry stays the same (the c_T sweep
confirms it).

**What stops AR from going to its upper bound?** The wing gets heavier (∝ AR^0.5) and thinner, so
it holds less fuel (∝ 1/√AR). These balance the lower induced drag.

**Newton vs modified Newton?** Newton uses H directly. When H is not positive definite the
direction may not go downhill, so we fall back to −∇f (counted as a fallback). Modified Newton
shifts H by ηI when its smallest eigenvalue is ≤ 0, so it always gives a descent direction.
When H is already positive definite the two are identical.

**Why does SR1 need resets and BFGS not?** The SR1 update does not keep B positive definite, so it
can produce an uphill direction; we then reset B = I. BFGS keeps B positive definite whenever
sᵀy > 0, and we skip the update otherwise.

**Why is gradient descent so slow?** As μ grows, the term μ∇g∇gᵀ makes L_A badly conditioned.
Steepest descent zig-zags and converges only linearly. It still reaches the answer, but it hits
the inner iteration cap in several subproblems (reported in the solver table).

**How did you measure the convergence order?** If e_{k+1} ≈ e_k^p, taking logs gives
p = ln e_{k+1} / ln e_k, where e_k is the gradient norm. We measured p ≈ 1 for gradient descent
(linear, each step only shrinks the gradient by about 1 %), about 1.2-1.4 for BFGS and SR1
(superlinear) and about 2 for Newton (quadratic), as the theory says.

**Why does the optimum sit there? (design-space picture)** With V and W_i at their limits, range
grows towards a bigger, more slender wing, but the fuel the MTOW allows stops fitting in the tanks.
The optimum is where the range contours touch the fuel-volume line. The 36 m limit moves it along
that line to where it crosses the span curve.

**Do your assumptions decide the result? (tornado)** No. Changing C_D0, e, C_L,max, reserve fuel or
the payload by ±10 % keeps the gain positive (about +4 % to +10 %). C_D0 and e give identical bars:
after the c_T calibration only their product matters, because a constant factor on L/D is absorbed
into c_T.

**Did you deviate from the proposal?** No in substance. The report's "Differences from the proposal"
section lists the implementation choices the proposal left open (effective c_T, ρ from altitude,
Raymer exponent 0.749, numerical Newton Hessian).

**How do you know it is the global optimum?** We don't, strictly: the problem is non-convex. We
have evidence: 40 Latin-hypercube starts all reach the same design, SLSQP agrees, and in the demo
thousands of random allowed designs never beat it. The KKT and second-order checks prove a
strict *local* minimum.

**What does a multiplier mean physically?** For MTOW, λ ≈ 2.6 means 1 % more MTOW gives about
2.6 % more range. In units, one extra tonne is worth about 190 km. We checked each prediction by
re-solving with the limit raised by 1 %.

**Is the model validated?** No, it is calibrated. It reproduces the 737-800 because its constants
were fitted to the 737-800. Predictions for very different aircraft are extrapolations.

**Where does each number come from?** The parameter table in the report gives the source of every
parameter (published, calibrated or assumed), and `docs/assumptions.md` gives the reasons.
`tests/test_parameters_and_reports.py` checks the published values against Boeing's imperial units.
