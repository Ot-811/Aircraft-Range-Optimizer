"""Augmented Lagrangian method (Powell-Hestenes-Rockafellar) for

    min f(z)  s.t.  c_E(z) = 0,  c_I(z) <= 0

    L_A(z; lam, nu, mu) = f + nu'c_E + mu/2 |c_E|^2
                          + 1/(2 mu) sum( max(0, lam_i + mu c_I,i)^2 - lam_i^2 )

Outer loop
  1. minimise L_A over z with one of the hand-written methods (unconstrained.py)
  2. update the multipliers  nu <- nu + mu c_E,   lam <- max(0, lam + mu c_I)   (keeps lam >= 0)
  3. if the violation did not drop by a factor 4, increase the penalty  mu <- 10 mu
  4. stop when violation, complementarity <= tol and the KKT stationarity residual <= 1e-6

At convergence lam and nu are the KKT (Lagrange) multipliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

from .unconstrained import minimize, Counter


@dataclass
class ALMResult:
    z: np.ndarray
    f: float
    lam: np.ndarray          # inequality multipliers (>= 0)
    nu: np.ndarray           # equality multipliers
    outer_it: int
    inner_it: int
    nf: int
    ng: int
    nh: int
    max_violation: float
    converged: bool
    time_s: float
    fallbacks: int = 0
    history: list = field(default_factory=list)


def augmented_lagrangian(prob, lam, nu, mu):
    """Return the functions L_A(z) and grad L_A(z) for fixed multipliers and penalty."""
    def LA(z):
        fz = prob.f(z)
        if not np.isfinite(fz):
            return np.inf
        cE, cI = prob.eq(z), prob.ineq(z)
        return (fz + nu @ cE + 0.5 * mu * cE @ cE
                + (np.maximum(0.0, lam + mu * cI) ** 2 - lam ** 2).sum() / (2 * mu))

    def grad_LA(z):
        g = prob.grad(z) + prob.jac_ineq(z).T @ np.maximum(0.0, lam + mu * prob.ineq(z))
        if len(nu):
            g = g + prob.jac_eq(z).T @ (nu + mu * prob.eq(z))
        return g

    return LA, grad_LA


def solve_alm(prob, z0, method="bfgs", mu0=10.0, mu_max=1e8, tol=1e-8, max_outer=50,
              inner_maxiter=3000):
    z = np.asarray(z0, float).copy()
    lam = np.zeros(len(prob.ineq(z)))
    nu = np.zeros(len(prob.eq(z)))
    mu = mu0
    cnt = Counter()
    hist = []
    inner_total = fallbacks = 0
    prev_viol = np.inf
    converged = False
    t0 = time.perf_counter()

    for k in range(1, max_outer + 1):
        LA, grad_LA = augmented_lagrangian(prob, lam, nu, mu)
        gtol = max(tol, 10.0 ** (-(k + 1)))           # tighten the inner tolerance as we go
        r = minimize(LA, grad_LA, z, method=method, gtol=gtol, maxiter=inner_maxiter, cnt=cnt)
        z = r.x
        inner_total += r.nit
        fallbacks += r.fallbacks

        cE, cI = prob.eq(z), prob.ineq(z)
        nu = nu + mu * cE
        lam = np.maximum(0.0, lam + mu * cI)

        viol = max(np.abs(cE).max(initial=0.0), np.maximum(cI, 0).max(initial=0.0))
        comp = np.abs(np.minimum(-cI, lam)).max(initial=0.0)       # complementarity measure
        grad_L = prob.grad(z) + prob.jac_eq(z).T @ nu + prob.jac_ineq(z).T @ lam
        stat = float(np.linalg.norm(grad_L, np.inf))
        hist.append(dict(outer=k, f=float(prob.f(z)), viol=float(viol), comp=float(comp),
                         stat=stat, mu=mu, inner_it=r.nit, inner_converged=r.converged))
        if viol <= tol and comp <= tol and stat <= 1e-6:
            converged = True
            break
        if max(viol, comp) > 0.25 * prev_viol:
            mu = min(mu * 10.0, mu_max)
        prev_viol = max(viol, comp)

    return ALMResult(z, float(prob.f(z)), lam, nu, k, inner_total, cnt.nf, cnt.ng, cnt.nh,
                     float(hist[-1]["viol"]), converged, time.perf_counter() - t0, fallbacks, hist)
