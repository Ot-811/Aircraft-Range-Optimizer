"""Hand-written unconstrained minimisers with Armijo backtracking line search.

Methods
    'gd'         steepest descent            d = -g
    'newton'     Newton                      d = -H^-1 g      (falls back to -g if not a descent direction)
    'modnewton'  modified Newton (lecture)   d = -(H + eta I)^-1 g,  eta = delta - lambda_min(H) if lambda_min <= 0
    'bfgs'       quasi-Newton, inverse form  d = -B g,  B updated with the BFGS formula
    'sr1'        quasi-Newton, inverse form  d = -B g,  B updated with the SR1 formula

Newton-type methods use the Hessian obtained by central differences of the *analytic*
gradient (second derivatives were not hand-coded). Functions may return +inf outside their
domain; the line search then simply backtracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

METHODS = ("gd", "newton", "modnewton", "bfgs", "sr1")


@dataclass
class Counter:
    nf: int = 0      # function evaluations
    ng: int = 0      # gradient evaluations
    nh: int = 0      # Hessian evaluations


@dataclass
class UncResult:
    x: np.ndarray
    f: float
    gnorm: float
    nit: int
    converged: bool
    fallbacks: int = 0          # times we had to use -g (or reset B) because d was not a descent direction
    history: list = field(default_factory=list)


def armijo(fun, x, fx, gx, d, cnt, alpha0=1.0, c1=1e-4, shrink=0.5, max_halvings=60):
    """Backtracking: halve alpha until f(x + alpha d) <= f(x) + c1 alpha g'd.
    Returns (alpha, f_new), or (0, fx) if no acceptable step was found."""
    slope = float(gx @ d)
    a = alpha0
    for _ in range(max_halvings):
        fn = fun(x + a * d)
        cnt.nf += 1
        if np.isfinite(fn) and fn <= fx + c1 * a * slope:
            return a, fn
        a *= shrink
    return 0.0, fx


def fd_hessian(grad, x, cnt, h=1e-6):
    """Hessian by central differences of the analytic gradient, symmetrised."""
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2 * h)
    cnt.ng += 2 * n
    cnt.nh += 1
    return 0.5 * (H + H.T)


def make_positive_definite(H, delta=1e-6):
    """Modified Newton (lecture 13): if the smallest eigenvalue is <= 0, shift the Hessian
    by eta I with eta = delta - lambda_min, so that H + eta I is positive definite."""
    lam_min = np.linalg.eigvalsh(H).min()
    if lam_min <= 0:
        return H + (delta - lam_min) * np.eye(len(H))
    return H


def minimize(fun, grad, x0, method="bfgs", gtol=1e-8, maxiter=2000, cnt: Counter | None = None,
             record=False):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    cnt = cnt or Counter()
    x = np.asarray(x0, float).copy()
    n = x.size
    fx = fun(x); cnt.nf += 1
    gx = grad(x); cnt.ng += 1
    B = np.eye(n)             # inverse-Hessian approximation for BFGS and SR1
    first = True              # first quasi-Newton step: B is still the identity
    fallbacks = 0
    hist = []
    for k in range(maxiter):
        gnorm = float(np.linalg.norm(gx, np.inf))
        if record:
            hist.append((k, fx, gnorm))
        if gnorm <= gtol:
            return UncResult(x, fx, gnorm, k, True, fallbacks, hist)

        # ---- search direction
        if method == "gd":
            d = -gx
        elif method in ("newton", "modnewton"):
            H = fd_hessian(grad, x, cnt)
            if method == "modnewton":
                H = make_positive_definite(H)
            try:
                d = -np.linalg.solve(H, gx)
            except np.linalg.LinAlgError:
                d = -gx
        else:                                  # bfgs, sr1
            d = -B @ gx
        if not np.all(np.isfinite(d)) or gx @ d >= 0:   # not a descent direction
            d = -gx
            B, first = np.eye(n), True
            fallbacks += 1

        # ---- step length
        alpha, fn = armijo(fun, x, fx, gx, d, cnt)
        if alpha == 0.0:
            if method in ("bfgs", "sr1") and not first:   # bad B: reset to identity and retry
                B, first = np.eye(n), True
                fallbacks += 1
                continue
            return UncResult(x, fx, gnorm, k, False, fallbacks, hist)
        s = alpha * d
        gn = grad(x + s); cnt.ng += 1
        y = gn - gx
        sy = float(s @ y)

        # ---- quasi-Newton update of the inverse Hessian B
        if method in ("bfgs", "sr1") and sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            if first:          # scale the identity to the curvature seen (Nocedal & Wright eq. 6.20)
                B = (sy / float(y @ y)) * np.eye(n)
                first = False
            if method == "bfgs":
                rho = 1.0 / sy
                I = np.eye(n)
                B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            else:              # SR1: B += (s - B y)(s - B y)' / ((s - B y)' y), skipped if the denominator is tiny
                r = s - B @ y
                den = float(r @ y)
                if abs(den) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(y):
                    B = B + np.outer(r, r) / den
        x, fx, gx = x + s, fn, gn
    gnorm = float(np.linalg.norm(gx, np.inf))
    return UncResult(x, fx, gnorm, maxiter, gnorm <= gtol, fallbacks, hist)
