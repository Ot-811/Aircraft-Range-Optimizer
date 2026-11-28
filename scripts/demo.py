"""Interactive demo: your requirements and starting design -> optimum, checks and report."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aircraft_opt.params import calibrated_params, baseline_x, B737
from aircraft_opt.model import AircraftModel
from aircraft_opt.problems import ReducedProblem
from aircraft_opt.alm import solve_alm
from aircraft_opt.unconstrained import METHODS
from aircraft_opt.reference import solve_slsqp, kkt_check
from aircraft_opt.studies import fly_fixed_geometry, latin_hypercube_starts
from aircraft_opt.atmosphere import FT
from aircraft_opt.report_writer import json_default
from demo_report import build_demo_report, DEFAULT_START_FUEL

REPORTS = ROOT / "demo_reports"
PAX_MASS = 100.0                     # kg per passenger including bags
N_RANDOM = 20_000  # random designs for the check
CONSTRAINT_NAMES = {"mtow": "MTOW", "stall": "stall speed", "fuel_volume": "fuel volume",
                    "mach": "Mach limit", "span": "gate span"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ask(question, default, lo=None, hi=None, kind=float, use_defaults=False):
    """Ask for a value; Enter keeps the default; asks again on bad input."""
    if use_defaults:
        print(f"  {question} [{default}]: {default}")
        return default
    while True:
        try:
            s = input(f"  {question} [{default}]: ").strip()
        except EOFError:
            s = ""
        if s == "":
            return default
        try:
            v = kind(s)
        except ValueError:
            print("    -> please type a valid value")
            continue
        if kind is not str and ((lo is not None and v < lo) or (hi is not None and v > hi)):
            print(f"    -> please type a value between {lo} and {hi}")
            continue
        return v


def folder_name(name):
    """Safe folder name: letters, digits, '-' and '_' only."""
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()).strip("_")[:60]
    return s or "demo"


VAR_LABELS = ("wing area", "aspect ratio", "cruise speed", "cruise fuel", "initial mass")


def violated(m, x, tol=1e-6):
    """{constraint: % by which the design breaks it}, for the physical constraints and the bounds."""
    out = {CONSTRAINT_NAMES[n]: 100 * float(g) for n, g in zip(m.ineq_names, m.g(x)) if g > tol}
    for label, xi, lo, hi in zip(VAR_LABELS, x, m.lb, m.ub):
        if xi < lo * (1 - tol):
            out[f"{label} below its lower bound {lo:g}"] = 100 * (lo / xi - 1)
        elif xi > hi * (1 + tol):
            out[f"{label} above its upper bound {hi:g}"] = 100 * (xi / hi - 1)
    return out


def random_feasible_designs(p, n, seed=0):
    """Random designs inside the bounds that satisfy every constraint; returns (spans, ranges)."""
    rng = np.random.default_rng(seed)
    rp = ReducedProblem(p)
    m = rp.m
    lo = np.array([p.lb[0], p.lb[1], p.lb[2], p.lb[4]])
    hi = np.array([p.ub[0], p.ub[1], p.ub[2], p.ub[4]])
    spans, ranges = [], []
    for y in lo + rng.random((n, 4)) * (hi - lo):
        x = rp.x_of_y(y)
        if x[3] > 0 and np.all(rp.ineq(rp.z_from_x(x)) <= 0):
            spans.append(math.sqrt(x[0] * x[1]))
            ranges.append(m.range_km(x))
    return np.array(spans), np.array(ranges)


def main(use_defaults=False, name=None):
    print("\n" + "=" * 70)
    print(" AIRCRAFT RANGE OPTIMISER - DEMO MODE")
    print(" Press Enter to keep the value in [brackets] (Boeing 737-800).")
    print("=" * 70)
    saved = sorted(d.name for d in REPORTS.glob("*") if (d / "results.json").is_file()) if REPORTS.is_dir() else []
    if saved:
        print("\nSaved runs: " + ", ".join(saved))

    name = name or ask("name for this run", "demo", kind=str, use_defaults=use_defaults)
    out = REPORTS / folder_name(name)
    if (out / "results.json").is_file():
        print(f"\nThe run '{out.name}' already exists. Its report: {out / 'report.md'}")
        if (out / "report.md").is_file():
            text = (out / "report.md").read_text(encoding="utf-8")
            start = text.find("## Result")
            section = text[start:text.find("\n## ", start + 5)] if start >= 0 else ""
            print("\n".join(l for l in section.splitlines() if not l.startswith(("![", "_"))))
        again = (not use_defaults) and ask("run it again and OVERWRITE it? (y/n)", "n", kind=str).lower() in ("y", "yes")
        if not again:
            print("Nothing changed. (Use a new name to save a new run.)")
            return
        shutil.rmtree(out)

    p737 = calibrated_params()  # 737 defaults
    print("\nA. Requirements and technology")
    pax = ask("number of passengers", 162, 0, 213, int, use_defaults)
    mtow = ask("maximum take-off mass [kg]", p737.MTOW, 60_000, 100_000, use_defaults=use_defaults)
    alt_ft = ask("cruise altitude [ft]", 35_000.0, 25_000, 43_000, use_defaults=use_defaults)
    engine = ask("engine fuel burn factor (1.0 = CFM56-7B, 0.85 = new-generation engine)", 1.0, 0.5, 1.5,
                 use_defaults=use_defaults)
    vstall = ask("maximum allowed stall speed [m/s]", p737.V_stall_max, 45, 80, use_defaults=use_defaults)
    mmax = ask("maximum cruise Mach number", p737.M_max, 0.6, 0.9, use_defaults=use_defaults)
    kw = ask("wing structure factor (1.0 = aluminium 737 wing, 0.8 = composite)", 1.0, 0.5, 1.5,
             use_defaults=use_defaults)
    bmax = ask("maximum wingspan for airport gates [m] (0 = no limit, 36 = ICAO code C)", 0.0, 0, 80,
               use_defaults=use_defaults)

    p = p737.with_(W_pay=pax * PAX_MASS, MTOW=mtow, h_cruise=alt_ft * FT, cT=engine * p737.cT,
                   V_stall_max=vstall, M_max=mmax, k_w=kw * p737.k_w, b_max=bmax if bmax > 0 else None)
    m = AircraftModel(p)

    print("\nB. YOUR starting design (the optimiser will try to improve it)")
    span = ask("wingspan [m]", B737["span"], 20, 70, use_defaults=use_defaults)
    S = ask("wing area [m²]", B737["S"], p.lb[0], p.ub[0], use_defaults=use_defaults)
    mach = ask("cruise Mach number", B737["M_cruise"], 0.5, 0.9, use_defaults=use_defaults)
    fuel = ask(f"cruise fuel [kg] (the {p.W_res:.0f} kg reserve comes on top)", DEFAULT_START_FUEL, 1_000, 40_000,
               use_defaults=use_defaults)

    print("\nC. Solver")
    solver = ask("inner solver: " + " / ".join(METHODS), "bfgs", kind=str, use_defaults=use_defaults).lower()
    if solver not in METHODS:
        print(f"    unknown solver '{solver}', using bfgs")
        solver = "bfgs"

    t0 = time.perf_counter()
    AR = span ** 2 / S
    x_user = np.array([S, AR, mach * p.a_sound, fuel, m.initial_mass(S, AR, fuel)])
    p_737wing = p.with_(k_w=p737.k_w)
    x_737 = fly_fixed_geometry(p_737wing, baseline_x(p737), p.W_pay)
    if x_737 is not None:
        x_737[2] = B737["M_cruise"] * p.a_sound

    print("\nOptimising ...")
    rp = ReducedProblem(p)
    res = solve_alm(rp, rp.z_from_x(x_user), method=solver)
    x_opt = rp.x(res.z)
    feasible = res.max_violation < 1e-6

    checks = {}
    if feasible:
        print("Checking the result (SLSQP, random starts, random designs, KKT) ...")
        s = solve_slsqp(p, x_user)
        checks["slsqp_range_km"] = m.range_km(s.x) if s.success else None
        starts = []
        for x0 in latin_hypercube_starts(p, 5, seed=1):
            r = solve_alm(rp, rp.z_from_x(x0), method=solver)
            starts.append(m.range_km(rp.x(r.z)) if r.converged else None)
        checks["random_start_ranges_km"] = starts
        spans, ranges = random_feasible_designs(p, N_RANDOM)
        checks["n_random"], checks["n_random_feasible"] = N_RANDOM, int(len(ranges))
        checks["best_random_km"] = float(ranges.max()) if len(ranges) else None
        k = kkt_check(p, x_opt)
        checks["kkt"] = {kk: v for kk, v in k.items() if kk not in ("lam_vector", "nu")}

    out.mkdir(parents=True, exist_ok=True)
    inputs = dict(name=name, passengers=pax, MTOW_kg=mtow, altitude_ft=alt_ft, engine_factor=engine,
                  max_stall_speed=vstall, max_Mach=mmax, wing_structure_factor=kw,
                  max_wingspan_m=bmax if bmax > 0 else None, start_wingspan_m=span, start_wing_area_m2=S,
                  start_Mach=mach, start_cruise_fuel_kg=fuel, solver=solver)
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    results = dict(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"), inputs=inputs,
        W_res=p.W_res, payload_kg=p.W_pay,
        start=m.describe(x_user), start_violations=violated(m, x_user),
        b737=({**m.describe(x_737), "W_wing": AircraftModel(p737).wing_mass(baseline_x(p737))}
              if x_737 is not None else None),
        b737_violations=violated(m, x_737) if x_737 is not None else None,
        optimum=m.describe(x_opt), feasible=feasible, max_violation=res.max_violation,
        solver=dict(name=solver, converged=res.converged, outer_it=res.outer_it, inner_it=res.inner_it,
                    time_s=res.time_s),
        history=[dict(outer=0, range_km=m.range_km(x_user),
                      violation=max(0.0, float(rp.ineq(rp.z_from_x(x_user)).max())))]
                + [dict(outer=h["outer"], range_km=math.exp(-h["f"]), violation=h["viol"]) for h in res.history],
        checks=checks, run_time_s=time.perf_counter() - t0,
    )
    (out / "results.json").write_text(json.dumps(results, indent=1, default=json_default), encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    hist = results["history"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot([h["outer"] for h in hist], [h["range_km"] for h in hist], "o-", color="#257de0", label="optimiser")
    if results["b737"]:
        axs[0].axhline(results["b737"]["range_km"], color="#ec4647", ls="--", label="737-800")
    axs[0].set_xlabel("outer iteration (0 = your start)"); axs[0].set_ylabel("range [km]")
    axs[0].set_title("Range during the optimisation"); axs[0].legend()
    axs[1].semilogy([h["outer"] for h in hist], [max(h["violation"], 1e-12) for h in hist], "o-", color="#f47230")
    axs[1].set_xlabel("outer iteration (0 = your start)"); axs[1].set_ylabel("max constraint violation")
    axs[1].set_title("Constraint violation (must go to ~0)")
    fig.tight_layout(); fig.savefig(out / "fig_history.png", dpi=150); plt.close(fig)

    if feasible and len(ranges):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(spans, ranges, s=3, c="#c8c8c8", label=f"{len(ranges):,} random allowed designs")
        if results["b737"] and not results["b737_violations"]:
            ax.plot(results["b737"]["span"], results["b737"]["range_km"], "s", color="#ec4647", ms=9, label="737-800")
        st = results["start"]
        ax.plot(st["span"], st["range_km"], "^", color="#257de0", ms=10,
                mfc="none" if results["start_violations"] else "#257de0",
                label="your start" + (" (not allowed)" if results["start_violations"] else ""))
        ax.plot(results["optimum"]["span"], results["optimum"]["range_km"], "*", color="gold", mec="k", ms=18,
                label="optimum")
        ax.set_xlabel("wingspan [m]"); ax.set_ylabel("range [km]")
        ax.set_title("No random allowed design flies further than the optimum")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_random_designs.png", dpi=150); plt.close(fig)

    status = build_demo_report(out)
    o = results["optimum"]
    print("\n" + "-" * 70)
    print(f"Your start:   {results['start']['range_km']:,.0f} km"
          + (f"  (NOT allowed: breaks {', '.join(results['start_violations'])})" if results["start_violations"] else ""))
    if results["b737"]:
        print(f"737-800:      {results['b737']['range_km']:,.0f} km")
    if feasible:
        print(f"OPTIMUM:      {o['range_km']:,.0f} km   (span {o['span']:.1f} m, area {o['S']:.1f} m², "
              f"AR {o['AR']:.2f}, Mach {o['Mach']:.3f})")
    else:
        print(f"NO allowed design exists for these requirements (smallest violation {res.max_violation:.1e}).")
    print(f"Report: {status}\nSaved in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defaults", action="store_true", help="no questions, 737-800 values")
    ap.add_argument("--name", help="run name (skips the name question)")
    a = ap.parse_args()
    main(use_defaults=a.defaults, name=a.name)
