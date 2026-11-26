"""Study drivers: optimisation with cross-check, method comparison, multi-start,
KKT sensitivities, payload-range and parameter sweeps."""
from __future__ import annotations

import math
import numpy as np

from .params import Params, baseline_x, B737, FUEL_DENSITY
from .model import AircraftModel
from .problems import ReducedProblem
from .alm import solve_alm, augmented_lagrangian
from .unconstrained import METHODS, minimize
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
        out["slsqp_success"] = s.success
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
                         max_violation=r.max_violation,
                         inner_cap_hits=sum(not h["inner_converged"] for h in r.history),
                         history=r.history))
    rows.append(dict(method="SLSQP (SciPy)", converged=s.success, outer_it=s.nit, inner_it=np.nan,
                     f_evals=s.nfev, grad_evals=s.njev, hess_evals=0, fallbacks=0, time_s=s.time_s,
                     range_km=AircraftModel(p).range_km(s.x), max_rel_err_vs_slsqp=0.0,
                     max_violation=float(max(abs(AircraftModel(p).h(s.x)), AircraftModel(p).g(s.x).max(), 0)),
                     inner_cap_hits=0, history=[]))
    return rows


def inner_histories(p: Params, x0=None, mu=10.0):
    """Convergence of each unconstrained method on the SAME first ALM subproblem (lam = 0, mu)."""
    x0 = baseline_x(p) if x0 is None else x0
    rp = ReducedProblem(p)
    z0 = rp.z_from_x(x0)
    LA, grad_LA = augmented_lagrangian(rp, np.zeros(len(rp.ineq(z0))), np.zeros(0), mu)
    out = {}
    for mth in METHODS:
        r = minimize(LA, grad_LA, z0, method=mth, gtol=1e-10, maxiter=4000, record=True)
        out[mth] = np.array(r.history)
    return out


def latin_hypercube_starts(p: Params, n, seed=0, max_batches=50):
    """Random starting points (S, AR, V, W_i) by Latin hypercube, kept only if the fuel left by the
    weight build-up is above its lower bound. W_i is drawn from 60 t (lighter aircraft cannot carry
    the 737 payload plus fuel) to its upper bound. Raises if no such point exists."""
    rng = np.random.default_rng(seed)
    rp = ReducedProblem(p)
    lo = np.array([p.lb[0], p.lb[1], p.lb[2], 60_000.0])
    hi = np.array([p.ub[0], p.ub[1], p.ub[2], p.ub[4]])
    pts = []
    for _ in range(max_batches):
        if len(pts) >= n:
            break
        u = (np.argsort(rng.random((n, 4)), axis=0) + rng.random((n, 4))) / n
        for y in lo + u * (hi - lo):
            if rp.fuel(y) > p.lb[3] and len(pts) < n:
                pts.append(rp.x_of_y(y))
    if len(pts) < n:
        raise ValueError("no starting point with positive fuel: the payload is too heavy for the bounds")
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


SENS_PARAMS = [("MTOW", "kg"), ("M_max", "unit of Mach"), ("k_f", "kg/m³ of k_f"), ("W_pay", "kg"),
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
        dRdp += 0.0                                   # turns -0.0 into 0.0
        rows.append(dict(parameter=name, unit=unit, value=v, dR_dp_km_per_unit=dRdp,
                         predicted_dR_km=dRdp * dp, resolved_dR_km=R1 - R0,
                         elasticity=dRdp * v / R0))
    return k, rows


def fly_fixed_geometry(p: Params, x_design, payload):
    """Fly an existing wing (S, AR, V of x_design, its wing mass) with a given payload.
    Fuel = as much as the tanks hold and the MTOW allows. Returns the design x, or None if
    no cruise fuel is left (payload too heavy for this MTOW)."""
    m = AircraftModel(p)
    S, AR, V = x_design[:3]
    OEW = p.W_fixed + m.wing_mass(x_design)
    tank = p.k_f * S ** 1.5 / math.sqrt(AR)
    fuel_total = min(tank, p.MTOW - OEW - payload)
    if fuel_total - p.W_res <= 0:
        return None
    return np.array([S, AR, V, fuel_total - p.W_res, OEW + payload + fuel_total])


def payload_range(p: Params, x_opt, payloads):
    """Three payload-range curves:
    * fixed 737-800 geometry, speed, wing mass (classic diagram: MTOW-, tank- and payload-limited),
    * fixed optimised geometry (designed for the nominal payload),
    * design re-optimised for each payload (envelope of best designs).
    Payload is capped at the 737-800 structural limit MZFW - OEW."""
    m = AircraftModel(p)
    xb = baseline_x(p)

    def fixed_range(x_design, pay):
        x = fly_fixed_geometry(p, x_design, pay)
        return np.nan if x is None else m.range_km(x)

    rows = []
    for pay in payloads:
        ropt = optimize(p.with_(W_pay=pay), x0=x_opt, check=True)
        rows.append(dict(payload_kg=pay,
                         range_737_km=fixed_range(xb, pay),
                         range_opt_fixed_km=fixed_range(x_opt, pay),
                         range_reopt_km=ropt["design"]["range_km"],
                         reopt_S=ropt["x"][0], reopt_AR=ropt["x"][1],
                         reopt_rel_diff_vs_slsqp=ropt["rel_diff_vs_slsqp"],
                         converged=ropt["alm"].converged, slsqp_success=ropt["slsqp_success"],
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
                         rel_diff_vs_slsqp=r["rel_diff_vs_slsqp"], converged=r["alm"].converged,
                         slsqp_success=r["slsqp_success"]))
        x0 = r["x"]
    return rows


def convergence_stats(gnorms):
    """Measured convergence of one unconstrained run from its gradient norms e_k = ||grad||.

    * order p: if e_{k+1} ~ e_k^p (scaled problem, constant ~1) then p = ln e_{k+1} / ln e_k.
      Averaged over the last two decreasing steps with e_k < 1e-2.
      Theory: 1 = linear (gradient descent), between 1 and 2 = superlinear (BFGS, SR1),
      2 = quadratic (Newton near the solution).
    * factor: average contraction e_{k+1}/e_k while e goes from 1e-3 to 1e-6 (close to 1 = slow).
    * iterations until e <= 1e-6.
    """
    e = np.asarray(gnorms, float)
    steps = [k for k in range(len(e) - 1) if e[k] < 1e-2 and 1e-13 < e[k + 1] < e[k]]
    order = float(np.mean([math.log(e[k + 1]) / math.log(e[k]) for k in steps[-2:]])) if steps else math.nan
    below = lambda t: next((k for k in range(len(e)) if e[k] <= t), None)  # noqa: E731
    a, b = below(1e-3), below(1e-6)
    factor = float((e[b] / e[a]) ** (1 / (b - a))) if a is not None and b is not None and b > a else math.nan
    return dict(order=order, factor=factor, iterations_to_1e_6=b)


def design_space(p: Params, x_opt, n=150, S_lim=(80.0, 200.0), AR_lim=(6.0, 16.0)):
    """Range over a grid of (S, AR), with V and W_i held at their optimal values (both sit on
    their limits: Mach and MTOW). W_fuel follows from the weight build-up. Returns the grid,
    the range [km] (NaN where there is no fuel) and the constraint values g (<= 0 is allowed)."""
    m = AircraftModel(p)
    V, Wi = x_opt[2], x_opt[4]
    S, AR = np.meshgrid(np.linspace(*S_lim, n), np.linspace(*AR_lim, n))
    Wf = Wi - p.W_fixed - p.k_w * S ** p.a * AR ** p.b * Wi ** p.c - p.W_res - p.W_pay
    R = np.full(S.shape, np.nan)
    G = {name: np.full(S.shape, np.nan) for name in m.ineq_names}
    for idx in np.ndindex(S.shape):
        x = (S[idx], AR[idx], V, Wf[idx], Wi)
        if Wf[idx] > 0:
            R[idx] = m.range_km(x)
            for name, g in zip(m.ineq_names, m.g(x)):
                G[name][idx] = g
    return dict(S=S, AR=AR, range_km=R, g=G)


# Modelling assumptions varied in the tornado study: (Params attribute, label, relative change)
TORNADO_PARAMS = [("CD0", "Zero-lift drag C_D0"), ("e", "Oswald efficiency e"),
                  ("CLmax", "Landing C_L,max"), ("W_res", "Reserve fuel"),
                  ("W_pay", "Payload (passenger mass)")]


def tornado(rel=0.10):
    """Change each assumed value by -rel and +rel and recompute the range GAIN of the optimum over
    the 737-800 (both flown with the same assumptions). Model constants (C_D0, e, reserve) are
    re-calibrated, exactly as the nominal model is; the payload is a mission input."""
    from .params import calibrated_params
    nominal = calibrated_params()
    rows = []
    for name, label in TORNADO_PARAMS:
        out = dict(parameter=name, label=label)
        for side, f in (("low", 1 - rel), ("high", 1 + rel)):
            p = calibrated_params(**{name: getattr(nominal, name) * f})
            m = AircraftModel(p)
            x737 = fly_fixed_geometry(p, baseline_x(nominal), p.W_pay)   # the real 737 wing and tanks
            r = optimize(p, check=True)
            out[f"{side}_gain_pct"] = 100 * (r["design"]["range_km"] / m.range_km(x737) - 1)
            out[f"{side}_range_km"] = r["design"]["range_km"]
            out[f"{side}_737_feasible"] = bool(np.all(m.g(x737) <= 1e-9))
            out[f"{side}_active"] = ";".join(r["active"])
            out[f"{side}_ok"] = bool(r["alm"].converged and r["slsqp_success"] and r["rel_diff_vs_slsqp"] < 1e-5)
        rows.append(out)
    return rows
