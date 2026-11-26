"""Hand-written unconstrained minimisers with Armijo backtracking line search.

Methods: 'gd' (steepest descent), 'newton', 'modnewton' (eigenvalue-modified Newton),
'bfgs' (inverse-Hessian BFGS), 'sr1' (symmetric rank-one with eigenvalue safeguard).

Newton-type methods use the Hessian obtained by central differences of the *analytic*
gradient (the objective is analytic; second derivatives were not hand-coded).
Functions may return +inf outside their domain; the line search then backtracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

METHODS = ("gd", "newton", "modnewton", "bfgs", "sr1")


@dataclass
class Counter:
    nf: int = 0
    ng: int = 0
    nh: int = 0


@dataclass
class UncResult:
    x: np.ndarray
    f: float
    gnorm: float
    nit: int
    converged: bool
    fallbacks: int = 0
    history: list = field(default_factory=list)


def armijo(fun, x, fx, gx, d, cnt, alpha0=1.0, c1=1e-4, shrink=0.5, max_halvings=60):
    """Backtracking until f(x + a d) <= f(x) + c1 a g'd. Returns (alpha, f_new) or (0, fx)."""
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
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        gp, gm = grad(x + e), grad(x - e)
        cnt.ng += 2
        H[:, i] = (gp - gm) / (2 * h)
    cnt.nh += 1
    return 0.5 * (H + H.T)


def _eig_modify(H, delta=1e-8):
    w, V = np.linalg.eigh(H)
    w = np.maximum(np.abs(w), delta * max(1.0, np.abs(w).max()))
    return (V * w) @ V.T


def minimize(fun, grad, x0, method="bfgs", gtol=1e-8, maxiter=2000, cnt: Counter | None = None,
             record=False):
    if method not in METHODS:
        raise ValueError(method)
    cnt = cnt or Counter()
    x = np.asarray(x0, float).copy()
    n = x.size
    fx = fun(x); cnt.nf += 1
    gx = grad(x); cnt.ng += 1
    Hinv = np.eye(n)          # BFGS inverse Hessian
    B = np.eye(n)             # SR1 Hessian
    first = True
    fallbacks = 0
    hist = []
    for k in range(maxiter):
        gnorm = float(np.linalg.norm(gx, np.inf))
        if record:
            hist.append((k, fx, gnorm))
        if gnorm <= gtol:
            return UncResult(x, fx, gnorm, k, True, fallbacks, hist)

        if method == "gd":
            d = -gx
        elif method in ("newton", "modnewton"):
            H = fd_hessian(grad, x, cnt)
            if method == "modnewton":
                H = _eig_modify(H)
            try:
                d = -np.linalg.solve(H, gx)
            except np.linalg.LinAlgError:
                d = -gx
            if not np.all(np.isfinite(d)) or gx @ d >= 0:   # pure Newton: not a descent direction
                d = -gx
                fallbacks += 1
        elif method == "bfgs":
            d = -Hinv @ gx
        else:  # sr1
            try:
                w = np.linalg.eigvalsh(B)
                Bm = B if w.min() > 1e-10 * max(1, abs(w).max()) else _eig_modify(B)
                d = -np.linalg.solve(Bm, gx)
            except np.linalg.LinAlgError:
                d = -gx
            if gx @ d >= 0:
                d = -gx
                fallbacks += 1

        alpha, fn = armijo(fun, x, fx, gx, d, cnt)
        if alpha == 0.0:
            if method in ("bfgs", "sr1") and not first:   # reset quasi-Newton matrix and retry
                Hinv, B, first = np.eye(n), np.eye(n), True
                fallbacks += 1
                continue
            return UncResult(x, fx, gnorm, k, False, fallbacks, hist)
        s = alpha * d
        xn = x + s
        gn = grad(xn); cnt.ng += 1
        yv = gn - gx
        sy = float(s @ yv)
        if method == "bfgs":
            if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(yv):
                if first:
                    Hinv = (sy / float(yv @ yv)) * np.eye(n)
                rho = 1.0 / sy
                I = np.eye(n)
                Hinv = (I - rho * np.outer(s, yv)) @ Hinv @ (I - rho * np.outer(yv, s)) + rho * np.outer(s, s)
                first = False
        elif method == "sr1":
            if first and sy > 0:
                B = (float(yv @ yv) / sy) * np.eye(n)
                first = False
            r = yv - B @ s
            den = float(r @ s)
            if abs(den) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
                B = B + np.outer(r, r) / den
        x, fx, gx = xn, fn, gn
    gnorm = float(np.linalg.norm(gx, np.inf))
    return UncResult(x, fx, gnorm, maxiter, gnorm <= gtol, fallbacks, hist)
