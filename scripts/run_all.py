"""Run every study in the proposal and write tables (results/tables), figures (results/figures)
and a one-page summary (results/summary.md).

    python scripts/run_all.py            # full run (~1-2 min)
    python scripts/run_all.py --quick    # fewer sweep points / starts
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aircraft_opt.params import calibrated_params, baseline_x, B737, NMI  # noqa: E402
from aircraft_opt.model import AircraftModel  # noqa: E402
from aircraft_opt.atmosphere import FT  # noqa: E402
from aircraft_opt import studies  # noqa: E402

try:  # Windows consoles default to cp1252; keep printing Unicode safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (fixed order), text inks
C = ["#257de0", "#f47230", "#00b894", "#d28e05", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "axes.titlesize": 10,
    "axes.titleweight": "bold",
})


def write_csv(name, rows):
    rows = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    with open(TAB / name, "w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def fmt(v, nd=1):
    return f"{v:,.{nd}f}"


def main(quick=False):
    t0 = time.perf_counter()
    log = []
    say = lambda s="": (print(s), log.append(s))  # noqa: E731

    # ------------------------------------------------------------ calibration
    p_lit = calibrated_params(calibrate_cT=False)
    p = calibrated_params()
    xb = baseline_x(p)
    m = AircraftModel(p)
    R_lit = AircraftModel(p_lit).range_km(xb)
    calib = dict(published_range_km=B737["range_km"], predicted_range_lit_cT_km=R_lit,
                 prediction_error_pct=100 * (R_lit / B737["range_km"] - 1),
                 cT_literature_lb_per_lbf_h=p_lit.cT * 3600, cT_effective_lb_per_lbf_h=p.cT * 3600,
                 k_w=p.k_w, W_fixed_kg=p.W_fixed, W_wing_737_kg=m.wing_mass(xb), k_f=p.k_f,
                 rho_cruise=p.rho, a_cruise=p.a_sound)
    json.dump(calib, open(TAB / "calibration.json", "w", encoding="utf-8"), indent=2)
    base = m.describe(xb)

    # --------------------------------------------------------- main optimum
    opt = studies.optimize(p)
    p36 = p.with_(b_max=36.0)
    opt36 = studies.optimize(p36)
    designs = [dict(case="737-800 baseline (model)", **base),
               dict(case="Optimum, no span limit", **opt["design"]),
               dict(case="Optimum, 36 m span (ICAO C)", **opt36["design"])]
    write_csv("designs.csv", designs)

    # ---------------------------------------------------- method comparison
    mc = studies.method_comparison(p)
    mc36 = studies.method_comparison(p36)
    write_csv("method_comparison_nospan.csv", mc)
    write_csv("method_comparison_span36.csv", mc36)
    hist = studies.inner_histories(p)

    # -------------------------------------------------- KKT + sensitivities
    kkt, sens = studies.sensitivities(p, opt)
    kkt36, sens36 = studies.sensitivities(p36, opt36)
    write_csv("sensitivities_nospan.csv", sens)
    write_csv("sensitivities_span36.csv", sens36)
    clean = lambda k: {kk: v for kk, v in k.items() if kk not in ("lam_vector", "nu")}  # noqa: E731
    json.dump(dict(nospan=clean(kkt), span36=clean(kkt36)), open(TAB / "kkt.json", "w", encoding="utf-8"), indent=2)

    # ------------------------------------------------------------ multistart
    ms = studies.multistart(p, n=10 if quick else 40)
    write_csv("multistart.csv", ms)

    # --------------------------------------------------------- payload-range
    pays = np.linspace(0, B737["MZFW"] - B737["OEW"], 7 if quick else 15)
    pr = studies.payload_range(p, opt["x"], pays)
    write_csv("payload_range.csv", pr)

    # ------------------------------------------------------------- sweeps
    n = 5 if quick else 9
    sweeps = {
        "k_w": ("Wing-weight coefficient  k_w / k_w,737", p.k_w, np.linspace(0.6, 1.2, n)),
        "cT": ("Engine TSFC  c_T / c_T,737", p.cT, np.linspace(0.75, 1.1, n)),
        "V_stall_max": ("Max. stall speed [m/s]", 1.0, np.linspace(52, 68, n)),
        "h_cruise": ("Cruise altitude [ft]", FT, np.linspace(27_000, 41_000, n)),
    }
    sweep_rows = {}
    for key, (label, scale, vals) in sweeps.items():
        rows = studies.sweep(p, key, vals * scale)
        for r, v in zip(rows, vals):
            r["x_plot"] = v
        sweep_rows[key] = rows
        write_csv(f"sweep_{key}.csv", rows)

    # ================================================================ figures
    # 1. baseline vs optimum bar comparison (indexed to baseline = 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    qty = [("range_km", "Range"), ("S", "Wing area"), ("AR", "Aspect ratio"), ("span", "Span"),
           ("V", "Cruise speed"), ("LD", "L/D"), ("W_wing", "Wing mass"), ("W_fuel", "Trip fuel")]
    w = 0.38
    xs = np.arange(len(qty))
    for j, (d, lab) in enumerate([(designs[1], "Optimum, no span limit"), (designs[2], "Optimum, 36 m span")]):
        vals = [100 * d[q] / base[q] for q, _ in qty]
        ax.bar(xs + (j - 0.5) * w, vals, w * 0.92, color=C[j], label=lab, zorder=3)
    ax.axhline(100, color=INK2, lw=1, zorder=2)
    ax.set_xticks(xs, [l for _, l in qty])
    ax.set_ylabel("% of 737-800 baseline")
    ax.set_ylim(80, 130)
    ax.set_title("Optimised designs relative to the 737-800 baseline (= 100)")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout(); fig.savefig(FIG / "fig1_design_comparison.png"); plt.close(fig)

    # 2. convergence of inner methods on the same subproblem
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = {"gd": "Gradient descent", "newton": "Newton", "modnewton": "Modified Newton",
             "bfgs": "BFGS", "sr1": "SR1"}
    for i, (k, h) in enumerate(hist.items()):
        ax.semilogy(h[:, 0], np.maximum(h[:, 2], 1e-16), color=C[i], label=names[k],
                    ls="--" if k == "newton" else "-", zorder=4 if k == "newton" else 3)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Iteration (first ALM subproblem, μ = 10)")
    ax.set_ylabel("‖∇L_A‖∞")
    ax.set_title("Unconstrained methods on the same subproblem\n(Newton and modified Newton coincide here: Hessian is positive definite)", fontsize=9)
    ax.legend(ncols=2)
    fig.tight_layout(); fig.savefig(FIG / "fig2_inner_convergence.png"); plt.close(fig)

    # 3. ALM outer convergence (BFGS)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (rows, lab) in enumerate([(mc, "no span limit"), (mc36, "36 m span")]):
        hh = [r for r in rows if r["method"] == "bfgs"][0]["history"]
        k = [h["outer"] for h in hh]
        ax.semilogy(k, [max(h["viol"], 1e-16) for h in hh], color=C[i], marker="o", label=f"violation, {lab}")
        ax.semilogy(k, [max(h["stat"], 1e-16) for h in hh], color=C[i], ls="--", marker="s",
                    label=f"stationarity, {lab}")
    ax.set_xlabel("ALM outer iteration"); ax.set_ylabel("max-norm")
    ax.set_title("Augmented Lagrangian (BFGS inner) convergence")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_alm_convergence.png"); plt.close(fig)

    # 4. payload-range
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    pay_t = np.array([r["payload_kg"] for r in pr]) / 1000
    for i, (key, lab) in enumerate([("range_737_km", "737-800 geometry (model)"),
                                    ("range_opt_fixed_km", "Optimised geometry, fixed"),
                                    ("range_reopt_km", "Re-optimised for each payload")]):
        ax.plot([r[key] for r in pr], pay_t, color=C[i], marker="o" if i == 2 else None, label=lab)
    ax.scatter([B737["range_km"]], [p.W_pay / 1000], color=INK, zorder=5, s=30)
    ax.annotate("published 737-800\n(162 pax)", (B737["range_km"], p.W_pay / 1000), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Range [km]"); ax.set_ylabel("Payload [t]")
    ax.set_title("Payload-range diagram")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig4_payload_range.png"); plt.close(fig)

    # 5-8. sweeps
    for idx, (key, (label, _, _)) in enumerate(sweeps.items()):
        rows = sweep_rows[key]
        xv = [r["x_plot"] for r in rows]
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.7))
        for a_, (q, ql) in zip(axs, [("range_km", "Range [km]"), ("S", "Wing area [m²]"),
                                      ("AR", "Aspect ratio"), ("Mach", "Cruise Mach")]):
            a_.plot(xv, [r[q] for r in rows], color=C[0], marker="o")
            a_.set_title(ql); a_.set_xlabel(label, fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig{5 + idx}_sweep_{key}.png"); plt.close(fig)

    # 9. multistart
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ra = np.array([r["alm_range_km"] for r in ms]); rs = np.array([r["slsqp_range_km"] for r in ms])
    ax.scatter(range(len(ms)), ra, color=C[0], label="ALM + BFGS (custom)", s=28, zorder=3)
    ax.scatter(range(len(ms)), rs, color=C[1], label="SLSQP (SciPy)", s=12, marker="x", zorder=4)
    ax.axhline(opt["design"]["range_km"], color=INK2, lw=1)
    ax.set_xlabel("Random start #"); ax.set_ylabel("Final range [km]")
    ax.set_title("Multi-start: final range from Latin-hypercube starting points")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig9_multistart.png"); plt.close(fig)

    # ================================================================ summary
    say("# Results summary (auto-generated by scripts/run_all.py)\n")
    say("## Calibration against the Boeing 737-800\n")
    say(f"- Published range: {fmt(B737['range_km'],0)} km ({B737['range_nmi']:.0f} nmi, 162 pax).")
    say(f"- Model with literature cruise TSFC {calib['cT_literature_lb_per_lbf_h']:.3f} lb/lbf/h predicts "
        f"{fmt(R_lit,0)} km ({calib['prediction_error_pct']:+.1f} %).")
    say(f"- Effective c_T that reproduces the published range: {calib['cT_effective_lb_per_lbf_h']:.3f} lb/lbf/h "
        "(absorbs climb/descent, taxi and other effects not modelled). Used for all studies.")
    say(f"- Wing mass (Raymer): {fmt(calib['W_wing_737_kg'],0)} kg; W_fixed = {fmt(p.W_fixed,0)} kg; "
        f"k_f = {p.k_f:.2f} kg/m³.\n")
    say("## Optimum vs baseline\n")
    cols = [("range_km", "Range [km]", 0), ("S", "S [m²]", 1), ("AR", "AR", 2), ("span", "Span [m]", 2),
            ("Mach", "Mach", 3), ("LD", "L/D", 2), ("CL", "C_L", 3), ("W_wing", "Wing [kg]", 0),
            ("W_fuel", "Trip fuel [kg]", 0), ("W_i", "W_i [kg]", 0), ("V_stall", "V_stall [m/s]", 1)]
    say("| Quantity | " + " | ".join(d["case"] for d in designs) + " |")
    say("|---|" + "---|" * len(designs))
    for q, lab, nd in cols:
        say(f"| {lab} | " + " | ".join(fmt(d[q], nd) for d in designs) + " |")
    g1 = 100 * (opt["design"]["range_km"] / base["range_km"] - 1)
    g2 = 100 * (opt36["design"]["range_km"] / base["range_km"] - 1)
    say(f"\nModel-predicted range gain: **{g1:+.2f} %** without span limit, **{g2:+.2f} %** with the 36 m limit.")
    say(f"Active constraints (no span): {', '.join(opt['active'])}. With span limit: {', '.join(opt36['active'])}.")
    say(f"Custom ALM vs SLSQP max relative difference in x: {opt['rel_diff_vs_slsqp']:.1e} / {opt36['rel_diff_vs_slsqp']:.1e}.\n")

    for title, rows in [("Method comparison (no span limit)", mc), ("Method comparison (36 m span)", mc36)]:
        say(f"## {title}\n")
        say("| Method | Converged | Outer it. | Inner it. | f evals | ∇f evals | Hessians | Fallbacks | Time [s] | Max rel. err vs SLSQP |")
        say("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            inner = "-" if isinstance(r["inner_it"], float) and math.isnan(r["inner_it"]) else r["inner_it"]
            say(f"| {r['method']} | {r['converged']} | {r['outer_it']} | {inner} | {r['f_evals']} | {r['grad_evals']} | "
                f"{r['hess_evals']} | {r['fallbacks']} | {r['time_s']:.3f} | {r['max_rel_err_vs_slsqp']:.1e} |")
        say("")

    for title, k, sr in [("KKT check and sensitivities (no span limit)", kkt, sens),
                         ("KKT check and sensitivities (36 m span)", kkt36, sens36)]:
        say(f"## {title}\n")
        say(f"- Active: {', '.join(k['active'])}; stationarity residual {k['stationarity']:.1e}; "
            f"min multiplier {k['min_multiplier']:.3g} (≥ 0 required); complementarity {k['complementarity']:.1e}.")
        say(f"- Multipliers (scaled problem): " + ", ".join(f"{a} = {b:.4g}" for a, b in k["multipliers"].items()))
        if k["null_space_dim"]:
            say(f"- Second order: reduced Hessian on the {k['null_space_dim']}-dim tangent space has eigenvalues "
                f"{', '.join(f'{e:.3g}' for e in k['reduced_hessian_eigs'])} (> 0 ⇒ strict local minimum).")
        else:
            say("- Second order: active constraint gradients span R^5 (tangent space is {0}) with strictly positive "
                "multipliers, so the point is a strict local minimum (vertex solution).")
        say("\n| Parameter | Value | dR/dp (KKT) | Predicted ΔR for +1 % [km] | Re-solved ΔR [km] | Elasticity |")
        say("|---|---|---|---|---|---|")
        for r in sr:
            say(f"| {r['parameter']} | {r['value']:.4g} | {r['dR_dp_km_per_unit']:.4g} km/{r['unit']} | "
                f"{r['predicted_dR_km']:.2f} | {r['resolved_dR_km']:.2f} | {r['elasticity']:.3f} |")
        say("")

    say("## Multi-start\n")
    ok = ra[np.isfinite(ra)]
    distinct = np.unique(np.round(ok, 1))
    say(f"- {len(ms)} Latin-hypercube starts. ALM converged in {sum(r['alm_converged'] for r in ms)}; "
        f"SLSQP succeeded in {sum(bool(r['slsqp_success']) for r in ms)}.")
    say(f"- Distinct final ranges (ALM, 0.1 km rounding): {', '.join(fmt(v,1) for v in distinct)} km. "
        "This is numerical evidence, not proof, of global optimality.\n")

    say("## Payload-range\n")
    say("| Payload [kg] | 737-800 geometry [km] | Optimised geometry fixed [km] | Re-optimised [km] | Re-opt. S | Re-opt. AR | Active (re-opt.) |")
    say("|---|---|---|---|---|---|---|")
    for r in pr:
        say(f"| {fmt(r['payload_kg'],0)} | {fmt(r['range_737_km'],0)} | {fmt(r['range_opt_fixed_km'],0)} | "
            f"{fmt(r['range_reopt_km'],0)} | {r['reopt_S']:.1f} | {r['reopt_AR']:.2f} | {r['reopt_active']} |")
    say("")
    say("## Parameter studies\n")
    for key, (label, _, _) in sweeps.items():
        say(f"### {label}\n")
        say("| Value | Range [km] | S [m²] | AR | Span [m] | Mach | L/D | Active | ALM vs SLSQP |")
        say("|---|---|---|---|---|---|---|---|---|")
        for r in sweep_rows[key]:
            say(f"| {r['x_plot']:.4g} | {fmt(r['range_km'],0)} | {r['S']:.1f} | {r['AR']:.2f} | {r['span']:.2f} | "
                f"{r['Mach']:.3f} | {r['LD']:.2f} | {r['active']} | {r['rel_diff_vs_slsqp']:.0e} |")
        say("")
    say(f"_Total run time: {time.perf_counter() - t0:.1f} s._")
    (ROOT / "results" / "summary.md").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
