"""SciPy SLSQP reference solver and an independent KKT / second-order check."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize as sp_minimize

from .problems import FullProblem


@dataclass
class SLSQPResult:
    x: np.ndarray
    f: float
    success: bool
    nit: int
    nfev: int
    njev: int
    time_s: float
    message: str


def solve_slsqp(p, x0, maxiter=500, ftol=1e-14):
    prob = FullProblem(p)
    cons = [dict(type="eq", fun=prob.eq, jac=prob.jac_eq),
            dict(type="ineq", fun=lambda z: -prob.ineq(z), jac=lambda z: -prob.jac_ineq(z))]
    t0 = time.perf_counter()
    r = sp_minimize(prob.f, prob.z(x0), jac=prob.grad, method="SLSQP", constraints=cons,
                    bounds=prob.bounds_z(), options=dict(maxiter=maxiter, ftol=ftol))
    return SLSQPResult(prob.x(r.x), float(r.fun), bool(r.success), int(r.nit), int(r.nfev),
                       int(r.njev), time.perf_counter() - t0, str(r.message))


def _fd_hess(fun_grad, z, h=1e-6):
    n = z.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        H[:, i] = (fun_grad(z + e) - fun_grad(z - e)) / (2 * h)
    return 0.5 * (H + H.T)


def kkt_check(p, x, act_tol=1e-6):
    """Independent KKT and second-order check of a design x (least-squares multipliers)."""
    prob = FullProblem(p, include_bounds_as_ineq=True)
    z = prob.z(x)
    gf = prob.grad(z)
    cI = prob.ineq(z)
    JI = prob.jac_ineq(z)
    JE = prob.jac_eq(z)
    act = np.where(cI > -act_tol)[0]
    A = np.vstack([JE, JI[act]])                      # rows = active constraint gradients
    lam_all, *_ = np.linalg.lstsq(A.T, -gf, rcond=None)
    stat = gf + A.T @ lam_all
    nu, lam_act = lam_all[:1], lam_all[1:]
    lam = np.zeros(len(cI)); lam[act] = lam_act

    def gradL(zz):
        return prob.grad(zz) + prob.jac_eq(zz).T @ nu + prob.jac_ineq(zz).T @ lam

    H = _fd_hess(gradL, z)
    strong = [0] + [1 + i for i, li in enumerate(lam_act) if li > 1e-8]
    As = A[strong]
    U, sv, Vt = np.linalg.svd(As)
    rank = int((sv > 1e-10 * sv.max()).sum())
    Z = Vt[rank:].T
    red_eigs = np.linalg.eigvalsh(Z.T @ H @ Z) if Z.shape[1] else np.array([])
    return dict(
        active=[prob.ineq_names[i] for i in act],
        multipliers={"weight_buildup": float(nu[0]),
                     **{prob.ineq_names[i]: float(lam[i]) for i in act}},
        stationarity=float(np.linalg.norm(stat, np.inf)),
        min_multiplier=float(lam_act.min()) if lam_act.size else 0.0,
        primal_violation=float(max(abs(prob.eq(z)[0]), max(cI.max(), 0.0))),
        complementarity=float(np.abs(lam * cI).max()),
        null_space_dim=int(Z.shape[1]),
        reduced_hessian_eigs=[float(e) for e in red_eigs],
        lam_vector=lam, nu=nu,
    )


def lagrangian_param_sensitivity(p, x, kkt, param, rel_step=1e-4):
    """Envelope theorem: dR/dp = -R dL/dp at fixed x* and multipliers."""
    from dataclasses import replace

    v = getattr(p, param)
    h = rel_step * abs(v)

    def L(pp):
        prob = FullProblem(pp, include_bounds_as_ineq=True)
        z = prob.z(x)
        return prob.f(z) + kkt["nu"] @ prob.eq(z) + kkt["lam_vector"] @ prob.ineq(z)

    dLdp = (L(replace(p, **{param: v + h})) - L(replace(p, **{param: v - h}))) / (2 * h)
    R = np.exp(-FullProblem(p).f(FullProblem(p).z(x)))
    return -R * dLdp
