"""Regression tests for bugs found in review (each test names the bug it guards against)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aircraft_opt.params import calibrated_params
from aircraft_opt.model import AircraftModel
from aircraft_opt.studies import latin_hypercube_starts

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_mission_overrides_do_not_change_the_calibration():
    """Bug: calibrated_params(W_pay=...) used to re-fit c_T and k_w to the new payload."""
    p = calibrated_params()
    q = calibrated_params(W_pay=10_000.0, MTOW=70_000.0, h_cruise=9_000.0)
    assert (q.cT, q.k_w, q.k_f, q.W_fixed) == (p.cT, p.k_w, p.k_f, p.W_fixed)
    assert (q.W_pay, q.MTOW, q.h_cruise) == (10_000.0, 70_000.0, 9_000.0)


def test_random_starts_stop_when_no_start_exists():
    """Bug: latin_hypercube_starts looped forever when no start had positive fuel."""
    with pytest.raises(ValueError):
        latin_hypercube_starts(calibrated_params().with_(W_pay=60_000.0), 3)


def test_demo_flags_bound_violations():
    """Bug: the demo only checked the physical constraints, not the variable bounds (AR 25 > 16)."""
    import demo
    p = calibrated_params()
    m = AircraftModel(p)
    S, AR, fuel = 100.0, 25.0, 10_000.0
    v = demo.violated(m, np.array([S, AR, 230.0, fuel, m.initial_mass(S, AR, fuel)]))
    assert any("aspect ratio above" in k for k in v)


def test_demo_end_to_end(tmp_path, monkeypatch):
    """The demo with default inputs runs, finds the known optimum and passes its own checks."""
    import demo
    import aircraft_opt.report_writer as rw
    monkeypatch.setattr(demo, "REPORTS", tmp_path)
    monkeypatch.setattr(demo, "N_RANDOM", 2_000)
    monkeypatch.setattr(rw.shutil, "which", lambda name: None)       # skip pdflatex in tests
    demo.main(use_defaults=True, name="t")
    R = json.loads((tmp_path / "t" / "results.json").read_text(encoding="utf-8"))
    assert R["feasible"] and R["solver"]["converged"]
    assert R["optimum"]["range_km"] == pytest.approx(5705.2, abs=1.0)
    assert R["checks"]["slsqp_range_km"] == pytest.approx(R["optimum"]["range_km"], abs=0.1)
    assert R["b737"]["range_km"] == pytest.approx(5436.0, abs=1.0)              # the real 737-800
    assert (tmp_path / "t" / "report.md").is_file()


def test_convergence_order_on_known_sequences():
    from aircraft_opt.studies import convergence_stats
    linear = [0.5 ** k for k in range(40)]                 # e_{k+1} = e_k / 2
    quadratic = [1e-1, 1e-2, 1e-4, 1e-8]                  # e_{k+1} = e_k^2
    assert convergence_stats(linear)["order"] == pytest.approx(1.0, abs=0.1)
    assert convergence_stats(linear)["factor"] == pytest.approx(0.5)
    assert convergence_stats(quadratic)["order"] == pytest.approx(2.0)


def test_design_space_optimum_is_best_allowed_grid_point():
    from aircraft_opt.studies import design_space, optimize
    p = calibrated_params()
    opt = optimize(p, check=False)
    ds = design_space(p, opt["x"], n=60)
    allowed = np.all([np.nan_to_num(g, nan=1.0) <= 1e-6 for g in ds["g"].values()], axis=0)
    assert allowed.sum() > 100
    assert np.nanmax(np.where(allowed, ds["range_km"], np.nan)) <= opt["design"]["range_km"] + 1e-6


def test_cd0_and_e_are_equivalent_after_calibration():
    """Only C_D0*e matters once c_T is calibrated, so scaling either by 0.9 gives the same optimum."""
    from aircraft_opt.studies import optimize
    p = calibrated_params()
    r1 = optimize(calibrated_params(CD0=0.9 * p.CD0), check=False)["design"]["range_km"]
    r2 = optimize(calibrated_params(e=0.9 * p.e), check=False)["design"]["range_km"]
    assert r1 == pytest.approx(r2, rel=1e-8)
