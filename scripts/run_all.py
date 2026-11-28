"""Main study: every study in the proposal -> results/ tables, figures, results.json and report."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aircraft_opt.params import calibrated_params, baseline_x, B737, PARAMETER_SOURCES
from aircraft_opt.model import AircraftModel
from aircraft_opt.atmosphere import FT
from aircraft_opt.report_writer import json_default
from aircraft_opt import studies
from main_report import build_main_report

try:  # UTF-8 output on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"

C = ["#257de0", "#f47230", "#00b894", "#d28e05", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "axes.titlesize": 10,
    "axes.titleweight": "bold",
})
METHOD_NAMES = {"gd": "Gradient descent", "newton": "Newton", "modnewton": "Modified Newton",
                "bfgs": "BFGS", "sr1": "SR1"}


def write_csv(name, rows):
    rows = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    with open(TAB / name, "w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def no_history(rows):
    return [{k: v for k, v in r.items() if k != "history"} for r in rows]


def main(quick=False):
    t0 = time.perf_counter()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    step = lambda s: print(f"[{time.perf_counter() - t0:5.1f} s] {s}")

    step("calibration")
    p_lit = calibrated_params(calibrate_cT=False)
    p = calibrated_params()
    xb = baseline_x(p)
    m = AircraftModel(p)
    R_lit = AircraftModel(p_lit).range_km(xb)
    calib = dict(published_range_km=B737["range_km"], range_nmi=B737["range_nmi"],
                 predicted_range_lit_cT_km=R_lit,
                 prediction_error_pct=100 * (R_lit / B737["range_km"] - 1),
                 cT_literature=p_lit.cT * 3600, cT_effective=p.cT * 3600,
                 k_w=p.k_w, W_fixed_kg=p.W_fixed, W_wing_737_kg=m.wing_mass(xb), k_f=p.k_f,
                 rho_cruise=p.rho, a_cruise=p.a_sound)
    (TAB / "calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    base = m.describe(xb)
    params = [dict(name=n, description=d, unit=u, value=getattr(p, n) * k, source=s)
              for n, d, u, k, s in PARAMETER_SOURCES]

    step("optimum with and without the 36 m span limit")
    opt = studies.optimize(p)
    p36 = p.with_(b_max=36.0)
    opt36 = studies.optimize(p36)
    designs = [dict(case="737-800 baseline (model)", **base),
               dict(case="Optimum, no span limit", **opt["design"]),
               dict(case="Optimum, 36 m span (ICAO C)", **opt36["design"])]
    write_csv("designs.csv", designs)

    step("method comparison (5 inner solvers + SLSQP, two cases)")
    mc = studies.method_comparison(p)
    mc36 = studies.method_comparison(p36)
    write_csv("method_comparison_nospan.csv", mc)
    write_csv("method_comparison_span36.csv", mc36)
    hist = studies.inner_histories(p)

    step("KKT check and sensitivities")
    kkt, sens = studies.sensitivities(p, opt)
    kkt36, sens36 = studies.sensitivities(p36, opt36)
    write_csv("sensitivities_nospan.csv", sens)
    write_csv("sensitivities_span36.csv", sens36)
    clean = lambda k: {kk: v for kk, v in k.items() if kk not in ("lam_vector", "nu")}
    (TAB / "kkt.json").write_text(json.dumps(dict(nospan=clean(kkt), span36=clean(kkt36)), indent=2,
                                          default=json_default), encoding="utf-8")

    step("multi-start")
    ms = studies.multistart(p, n=10 if quick else 40)
    write_csv("multistart.csv", ms)

    step("payload-range")
    pays = np.linspace(0, B737["MZFW"] - B737["OEW"], 7 if quick else 15)
    pr = studies.payload_range(p, opt["x"], pays)
    write_csv("payload_range.csv", pr)

    step("parameter sweeps")
    n = 5 if quick else 9
    sweeps = {  # (label, scale to model value, plotted values)
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

    step("convergence rates, design space, assumption tornado")
    conv = [dict(method=k, **studies.convergence_stats(h[:, 2])) for k, h in hist.items()]
    write_csv("convergence_rates.csv", conv)
    ds = studies.design_space(p, opt["x"])
    torn = studies.tornado()
    write_csv("tornado.csv", torn)

    step("figures")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    qty = [("range_km", "Range"), ("S", "Wing area"), ("AR", "Aspect ratio"), ("span", "Span"),
           ("V", "Cruise speed"), ("LD", "L/D"), ("W_wing", "Wing mass"), ("W_fuel", "Trip fuel")]
    w = 0.38
    xs = np.arange(len(qty))
    for j, (d, lab) in enumerate([(designs[1], "Optimum, no span limit"), (designs[2], "Optimum, 36 m span")]):
        ax.bar(xs + (j - 0.5) * w, [100 * d[q] / base[q] for q, _ in qty], w * 0.92, color=C[j], label=lab, zorder=3)
    ax.axhline(100, color=INK2, lw=1, zorder=2)
    ax.set_xticks(xs, [l for _, l in qty])
    ax.set_ylabel("% of 737-800 baseline")
    pcts = [100 * d[q] / base[q] for d in designs[1:] for q, _ in qty]
    ax.set_ylim(min(80, min(pcts) - 5), max(130, max(pcts) + 5))      # never clip a bar
    ax.set_title("Optimised designs relative to the 737-800 baseline (= 100)")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout(); fig.savefig(FIG / "fig1_design_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (k, h) in enumerate(hist.items()):
        ax.semilogy(h[:, 0], np.maximum(h[:, 2], 1e-16), color=C[i], label=METHOD_NAMES[k],
                    ls="--" if k == "newton" else "-", zorder=4 if k == "newton" else 3)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Iteration (first ALM subproblem, μ = 10)")
    ax.set_ylabel("‖∇L_A‖∞")
    title = "Unconstrained methods on the same subproblem"
    if np.array_equal(hist["newton"], hist["modnewton"]):
        title += "\n(Newton and modified Newton coincide: the Hessian is positive definite here)"
    ax.set_title(title, fontsize=9)
    ax.legend(ncols=2)
    fig.tight_layout(); fig.savefig(FIG / "fig2_inner_convergence.png"); plt.close(fig)

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

    for idx, (key, (label, _, _)) in enumerate(sweeps.items()):
        rows = sweep_rows[key]
        xv = [r["x_plot"] for r in rows]
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.7))
        for a_, (q, ql) in zip(axs, [("range_km", "Range [km]"), ("S", "Wing area [m²]"),
                                      ("AR", "Aspect ratio"), ("Mach", "Cruise Mach")]):
            yv = np.array([r[q] for r in rows])
            a_.plot(xv, yv, color=C[0], marker="o")
            a_.ticklabel_format(axis="y", useOffset=False)
            if np.ptp(yv) < 1e-6 * abs(yv.mean()):        # constant: don't plot rounding noise
                a_.set_ylim(0.95 * yv.mean(), 1.05 * yv.mean())
            a_.set_title(ql); a_.set_xlabel(label, fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig{5 + idx}_sweep_{key}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ra = np.array([r["alm_range_km"] for r in ms]); rs = np.array([r["slsqp_range_km"] for r in ms])
    ax.scatter(range(len(ms)), ra, color=C[0], label="ALM + BFGS (custom)", s=28, zorder=3)
    ax.scatter(range(len(ms)), rs, color=C[1], label="SLSQP (SciPy)", s=12, marker="x", zorder=4)
    ax.axhline(opt["design"]["range_km"], color=INK2, lw=1)
    ax.set_xlabel("Random start #"); ax.set_ylabel("Final range [km]")
    ax.set_title("Multi-start: final range from Latin-hypercube starting points")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig9_multistart.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cs = ax.contourf(ds["S"], ds["AR"], ds["range_km"], levels=20, cmap="Blues")
    fig.colorbar(cs, ax=ax, label="Range [km]")
    ax.contour(ds["S"], ds["AR"], ds["range_km"], levels=20, colors="white", linewidths=0.4, alpha=0.6)
    bad = np.zeros(ds["S"].shape, bool)
    for name, col, lab in [("fuel_volume", C[1], "fuel volume limit"), ("stall", C[7], "stall limit")]:
        g = ds["g"][name]
        bad |= np.nan_to_num(g, nan=1.0) > 0
        if np.nanmin(g) < 0 < np.nanmax(g):
            ax.contour(ds["S"], ds["AR"], g, levels=[0], colors=col, linewidths=2)
            ax.plot([], [], color=col, lw=2, label=lab)
    ax.contourf(ds["S"], ds["AR"], bad.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
    s_line = np.linspace(ds["S"].min(), ds["S"].max(), 200)
    ax.plot(s_line, 36.0 ** 2 / s_line, color=INK, ls="--", lw=1.3, label="36 m span")
    ax.plot(opt["design"]["S"], opt["design"]["AR"], "*", ms=16, color="gold", mec=INK, label="optimum")
    ax.plot(opt36["design"]["S"], opt36["design"]["AR"], "*", ms=12, color=C[2], mec=INK, label="optimum, 36 m span")
    ax.set_ylim(ds["AR"].min(), ds["AR"].max())
    ax.set_xlabel("Wing area S [m²]"); ax.set_ylabel("Aspect ratio AR")
    ax.set_title("Design space at V = Mach limit and W_i = MTOW (hatched = not allowed)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, frameon=True)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "fig10_design_space.png"); plt.close(fig)

    g0 = 100 * (opt["design"]["range_km"] / base["range_km"] - 1)
    order = sorted(torn, key=lambda r: abs(r["high_gain_pct"] - r["low_gain_pct"]))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for i, r in enumerate(order):
        ax.barh(i, r["low_gain_pct"] - g0, left=g0, color=C[0], height=0.6, zorder=3, label="−10 %" if i == 0 else None)
        ax.barh(i, r["high_gain_pct"] - g0, left=g0, color=C[1], height=0.6, zorder=3, label="+10 %" if i == 0 else None)
    ax.axvline(g0, color=INK, lw=1, zorder=4)
    ax.set_yticks(range(len(order)), [r["label"] for r in order])
    ax.set_xlabel("Range gain of the optimum over the 737-800 [%]")
    fig.suptitle(f"Range gain (nominal {g0:+.1f} %) when each assumption changes by ±10 %",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig11_tornado.png"); plt.close(fig)

    ok = ra[np.isfinite(ra)]
    results = dict(
        meta=dict(date=datetime.now().strftime("%Y-%m-%d %H:%M"), quick=quick,
                  run_time_s=time.perf_counter() - t0, python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        params=params, calibration=calib, designs=designs,
        optimum=dict(active=opt["active"], rel_diff_vs_slsqp=opt["rel_diff_vs_slsqp"],
                     alm_converged=opt["alm"].converged, slsqp_success=opt["slsqp_success"]),
        optimum36=dict(active=opt36["active"], rel_diff_vs_slsqp=opt36["rel_diff_vs_slsqp"],
                       alm_converged=opt36["alm"].converged, slsqp_success=opt36["slsqp_success"]),
        methods=dict(nospan=no_history(mc), span36=no_history(mc36)),
        kkt=dict(nospan=clean(kkt), span36=clean(kkt36)),
        sensitivities=dict(nospan=sens, span36=sens36),
        multistart=dict(n=len(ms), alm_converged=sum(bool(r["alm_converged"]) for r in ms),
                        slsqp_success=sum(bool(r["slsqp_success"]) for r in ms),
                        distinct_ranges_km=sorted(set(np.round(ok, 1).tolist()))),
        payload_range=pr, convergence=conv, tornado=torn,
        sweeps={k: dict(label=sweeps[k][0], rows=sweep_rows[k]) for k in sweeps},
    )
    results["meta"]["run_time_s"] = time.perf_counter() - t0
    with open(RES / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=json_default)

    step("report")
    print(build_main_report(RES))
    step(f"done. Results in {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
