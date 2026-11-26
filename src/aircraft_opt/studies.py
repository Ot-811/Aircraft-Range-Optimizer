"""Study drivers: optimisation with cross-check, method comparison, multi-start,
KKT sensitivities, payload-range and parameter sweeps."""
from __future__ import annotations

import math
import numpy as np

from .params import Params, baseline_x, B737, FUEL_DENSITY
from .model import AircraftModel
from .problems import ReducedProblem
from .alm import solve_alm
from .unconstrained import METHODS
from .reference import solve_slsqp, kkt_check, lagrangian_param_sensitivity


def optimize(p: Params, x0=None, method="bfgs", check=True):
    """Solve with the custom ALM (reduced formulation) and cross-check with SLSQP (full)."""
    x0 = baseline_x(p) if x0 is None else x0
    rp = ReducedProblem(p)
    r = solve_alm(rp, rp.z_from_x(x0), method=method)
    x = rp.x(r.z)
    out = dict(x=x, alm=r, design=AircraftModel(p).describe(x), active=_active(p, x))
    if check:
        s = solve_slsqp(p, x0)
        out["slsqp"] = s
        out["rel_diff_vs_slsqp"] = float(np.abs(x / s.x - 1).max())
    return out


def _active(p, x, tol=1e-6):
    m = AircraftModel(p)
    act = [n for n, g in zip(m.ineq_names, m.g(x)) if g > -tol]
    act += [f"{v}_lb" for v, xi, lo in zip(("S", "AR", "V", "W_fuel", "W_i"), x, m.lb) if xi <= lo * (1 + tol)]
    act += [f"{v}_ub" for v, xi, hi in zip(("S", "AR", "V", "W_fuel", "W_i"), x, m.ub) if xi >= hi * (1 - tol)]
    return act


def method_comparison(p: Params, x0=None):
    x0 = baseline_x(p) if x0 is None else x0
    rp = ReducedProblem(p)
    s = solve_slsqp(p, x0)
    rows = []
    for mth in METHODS:
        r = solve_alm(rp, rp.z_from_x(x0), method=mth)
        x = rp.x(r.z)
        rows.append(dict(method=mth, converged=r.converged, outer_it=r.outer_it, inner_it=r.inner_it,
                         f_evals=r.nf, grad_evals=r.ng, hess_evals=r.nh, fallbacks=r.fallbacks,
                         time_s=r.time_s, range_km=AircraftModel(p).range_km(x),
                         max_rel_err_vs_slsqp=float(np.abs(x / s.x - 1).max()),
                         max_violation=r.max_violation, history=r.history))
    rows.append(dict(method="SLSQP (SciPy)", converged=s.success, outer_it=s.nit, inner_it=np.nan,
                     f_evals=s.nfev, grad_evals=s.njev, hess_evals=0, fallbacks=0, time_s=s.time_s,
                     range_km=AircraftModel(p).range_km(s.x), max_rel_err_vs_slsqp=0.0,
                     max_violation=float(max(abs(AircraftModel(p).h(s.x)), AircraftModel(p).g(s.x).max(), 0)),
                     history=[]))
    return rows


def inner_histories(p: Params, x0=None, mu=10.0):
    """Convergence of each unconstrained method on the SAME first ALM subproblem (lam=0, mu)."""
    from .unconstrained import minimize
    x0 = baseline_x(p) if x0 is None else x0
    rp = ReducedProblem(p)
    lam = np.zeros(len(rp.ineq(rp.z_from_x(x0))))

    def LA(z):
        fz = rp.f(z)
        if not np.isfinite(fz):
            return np.inf
        return fz + (np.maximum(0.0, lam + mu * rp.ineq(z)) ** 2 - lam ** 2).sum() / (2 * mu)

    def gLA(z):
        return rp.grad(z) + rp.jac_ineq(z).T @ np.maximum(0.0, lam + mu * rp.ineq(z))

    out = {}
    for mth in METHODS:
        r = minimize(LA, gLA, rp.z_from_x(x0), method=mth, gtol=1e-10, maxiter=4000, record=True)
        out[mth] = np.array(r.history)
    return out


def latin_hypercube_starts(p: Params, n, seed=0):
    """Random starting points (S, AR, V, W_i) by Latin hypercube, kept only if fuel > 0."""
    rng = np.random.default_rng(seed)
    rp = ReducedProblem(p)
    lo = np.array([p.lb[0], p.lb[1], p.lb[2], 60_000.0])
    hi = np.array([p.ub[0], p.ub[1], p.ub[2], p.ub[4]])
    pts = []
    while len(pts) < n:
        u = (np.argsort(rng.random((n, 4)), axis=0) + rng.random((n, 4))) / n
        for y in lo + u * (hi - lo):
            if rp.fuel(y) > p.lb[3] and len(pts) < n:
                pts.append(rp.x_of_y(y))
    return pts


def multistart(p: Params, n=40, seed=0):
    rows = []
    rp = ReducedProblem(p)
    for i, x0 in enumerate(latin_hypercube_starts(p, n, seed)):
        r = solve_alm(rp, rp.z_from_x(x0), method="bfgs")
        xa = rp.x(r.z)
        s = solve_slsqp(p, x0)
        rows.append(dict(start=i, **{f"x0_{k}": v for k, v in zip(("S", "AR", "V", "W_fuel", "W_i"), x0)},
                         alm_converged=r.converged, alm_range_km=AircraftModel(p).range_km(xa),
                         alm_S=xa[0], alm_AR=xa[1], alm_V=xa[2],
                         slsqp_success=s.success, slsqp_range_km=AircraftModel(p).range_km(s.x) if s.success else np.nan,
                         slsqp_S=s.x[0], slsqp_AR=s.x[1], slsqp_V=s.x[2]))
    return rows


SENS_PARAMS = [("MTOW", "kg"), ("M_max", "-"), ("k_f", "fuel-volume coeff."), ("W_pay", "kg"),
               ("V_stall_max", "m/s"), ("b_max", "m")]


def sensitivities(p: Params, opt, rel=0.01):
    """KKT-multiplier sensitivities d(range)/d(param), verified by re-solving with a +rel change."""
    x = opt["x"]
    k = kkt_check(p, x)
    R0 = opt["design"]["range_km"]
    rows = []
    for name, unit in SENS_PARAMS:
        if getattr(p, name) is None:
            continue
        v = getattr(p, name)
        dRdp = lagrangian_param_sensitivity(p, x, k, name)
        dp = rel * v
        R1 = optimize(p.with_(**{name: v + dp}), x0=x, check=False)["design"]["range_km"]
        rows.append(dict(parameter=name, unit=unit, value=v, dR_dp_km_per_unit=dRdp,
                         predicted_dR_km=dRdp * dp, resolved_dR_km=R1 - R0,
                         elasticity=dRdp * v / R0))
    return k, rows


def payload_range(p: Params, x_opt, payloads):
    """Three payload-range curves:
    * fixed 737-800 geometry, speed, wing mass (classic diagram: MTOW-, tank- and payload-limited),
    * fixed optimised geometry (designed for the nominal payload),
    * design re-optimised for each payload (envelope of best designs).
    Payload is capped at the 737-800 structural limit MZFW - OEW."""
    m = AircraftModel(p)
    xb = baseline_x(p)

    def fixed_geometry_range(x_design, pay):
        S, AR, V = x_design[:3]
        Wwing = m.wing_mass(x_design)
        OEW = p.W_fixed + Wwing
        cap = p.k_f * S ** 1.5 / math.sqrt(AR)
        fuel_tot = min(cap, p.MTOW - OEW - pay)
        Wf = fuel_tot - p.W_res
        if Wf <= 0:
            return np.nan
        return m.range_km((S, AR, V, Wf, OEW + pay + fuel_tot))

    rows = []
    for pay in payloads:
        ropt = optimize(p.with_(W_pay=pay), x0=x_opt, check=True)
        rows.append(dict(payload_kg=pay,
                         range_737_km=fixed_geometry_range(xb, pay),
                         range_opt_fixed_km=fixed_geometry_range(x_opt, pay),
                         range_reopt_km=ropt["design"]["range_km"],
                         reopt_S=ropt["x"][0], reopt_AR=ropt["x"][1],
                         reopt_rel_diff_vs_slsqp=ropt["rel_diff_vs_slsqp"],
                         reopt_active=";".join(ropt["active"])))
    return rows


def sweep(p: Params, name, values):
    rows = []
    x0 = None
    for v in values:
        r = optimize(p.with_(**{name: v}), x0=x0)
        d = r["design"]
        rows.append(dict(param=name, value=v, range_km=d["range_km"], S=d["S"], AR=d["AR"], span=d["span"],
                         Mach=d["Mach"], V=d["V"], W_fuel=d["W_fuel"], W_wing=d["W_wing"], LD=d["LD"],
                         CL=d["CL"], V_stall=d["V_stall"], active=";".join(r["active"]),
                         rel_diff_vs_slsqp=r["rel_diff_vs_slsqp"], converged=r["alm"].converged))
        x0 = r["x"]
    return rows
