"""Published data, ISA, solver convergence and LaTeX escaping."""
import math

import numpy as np
import pytest

from aircraft_opt.atmosphere import isa, FT
from aircraft_opt.params import B737, FUEL_DENSITY, LB, FT2, PARAMETER_SOURCES, calibrated_params, baseline_x
from aircraft_opt.model import AircraftModel
from aircraft_opt.problems import ReducedProblem
from aircraft_opt.alm import solve_alm
from aircraft_opt.report_writer import latex_inline


def test_737_published_data_unit_conversions():
    """Boeing publishes these in imperial units; check our metric values."""
    assert B737["MTOW"] == pytest.approx(174_200 * LB, abs=1)
    assert B737["OEW"] == pytest.approx(91_300 * LB, abs=1)
    assert B737["MZFW"] == pytest.approx(138_300 * LB, abs=1)
    assert B737["fuel_capacity_L"] == pytest.approx(6_875 * 3.785411784, rel=1e-3)
    assert B737["S"] == pytest.approx(1_341.2 * FT2, abs=0.05)
    assert B737["span"] == pytest.approx((112 + 7 / 12) * 0.3048, abs=0.01)
    assert B737["range_km"] == pytest.approx(2_935 * 1.852)
    assert B737["AR"] == pytest.approx(9.45, abs=0.01)
    assert 0.775 <= FUEL_DENSITY <= 0.840                     # Jet A-1 density range


def test_isa_at_cruise_altitude():
    T, p, rho, a = isa(35_000 * FT)
    assert T == pytest.approx(218.8, abs=0.1)
    assert rho == pytest.approx(0.3796, abs=1e-3)
    assert a == pytest.approx(296.5, abs=0.2)
    assert isa(0)[2] == pytest.approx(1.225, abs=1e-3)


def test_parameter_table_is_complete():
    p = calibrated_params()
    for name, *_ in PARAMETER_SOURCES:
        assert math.isfinite(getattr(p, name)), name


def test_initial_mass_closes_weight_buildup():
    p = calibrated_params()
    m = AircraftModel(p)
    S, AR, fuel = 130.0, 11.0, 17_000.0
    Wi = m.initial_mass(S, AR, fuel)
    assert abs(m.h((S, AR, 230.0, fuel, Wi))) < 1e-12


@pytest.mark.parametrize("span", [None, 36.0])
@pytest.mark.parametrize("method", ["newton", "modnewton", "bfgs", "sr1"])
def test_inner_methods_never_hit_iteration_cap(span, method):
    p = calibrated_params().with_(b_max=span)
    rp = ReducedProblem(p)
    r = solve_alm(rp, rp.z_from_x(baseline_x(p)), method=method)
    assert r.converged
    assert all(h["inner_converged"] for h in r.history)


def test_latex_escaping():
    s = latex_inline("**W_fuel** ≤ 50 % of μ·S²")
    assert s == r"\textbf{W\_fuel} $\le$ 50 \% of $\mu$$\cdot$S\textsuperscript{2}"
    assert "?" not in latex_inline("λ ← max(0, λ + μ g), ½ ρ₀ V², eₖ₊₁")
