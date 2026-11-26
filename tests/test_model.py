import numpy as np
import pytest

from aircraft_opt.params import calibrated_params, baseline_x, B737
from aircraft_opt.model import AircraftModel
from aircraft_opt.problems import FullProblem, ReducedProblem
from aircraft_opt.alm import solve_alm
from aircraft_opt.reference import solve_slsqp, kkt_check


@pytest.fixture(scope="module")
def p():
    return calibrated_params()


def fd_grad(f, z, h=1e-7):
    g = np.zeros_like(z)
    for i in range(z.size):
        e = np.zeros_like(z); e[i] = h
        g[i] = (f(z + e) - f(z - e)) / (2 * h)
    return g


def fd_jac(c, z, h=1e-7):
    return np.column_stack([(c(z + h * e) - c(z - h * e)) / (2 * h) for e in np.eye(z.size)])


@pytest.mark.parametrize("span", [None, 36.0])
def test_full_gradients(p, span):
    prob = FullProblem(p.with_(b_max=span), include_bounds_as_ineq=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = prob.z(baseline_x(p)) * (1 + 0.1 * rng.uniform(-1, 1, 5))
        assert np.allclose(prob.grad(z), fd_grad(prob.f, z), rtol=1e-5, atol=1e-7)
        assert np.allclose(prob.jac_eq(z), fd_jac(prob.eq, z), rtol=1e-5, atol=1e-7)
        assert np.allclose(prob.jac_ineq(z), fd_jac(prob.ineq, z), rtol=1e-5, atol=1e-7)


def test_reduced_gradients(p):
    prob = ReducedProblem(p.with_(b_max=36.0))
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = prob.z_from_x(baseline_x(p)) * (1 + 0.1 * rng.uniform(-1, 1, 4))
        assert np.allclose(prob.grad(z), fd_grad(prob.f, z), rtol=1e-5, atol=1e-7)
        assert np.allclose(prob.jac_ineq(z), fd_jac(prob.ineq, z), rtol=1e-5, atol=1e-7)


def test_calibration_reproduces_737(p):
    m = AircraftModel(p)
    xb = baseline_x(p)
    assert abs(m.range_km(xb) - B737["range_km"]) < 1e-6
    assert abs(m.h(xb)) < 1e-12                       # weight build-up closes at OEW
    assert abs(m.g(xb)[2]) < 1e-12                   # tanks exactly full
    assert np.all(m.g(xb) <= 1e-12)                  # baseline is feasible


@pytest.mark.parametrize("span", [None, 36.0])
def test_custom_solver_matches_slsqp(p, span):
    pp = p.with_(b_max=span)
    xb = baseline_x(p)
    ref = solve_slsqp(pp, xb)
    rp = ReducedProblem(pp)
    r = solve_alm(rp, rp.z_from_x(xb))
    assert ref.success and r.converged
    assert np.allclose(rp.x(r.z), ref.x, rtol=1e-5)
    k = kkt_check(pp, ref.x)
    assert k["stationarity"] < 1e-6 and k["min_multiplier"] > -1e-8
    assert all(e > 0 for e in k["reduced_hessian_eigs"])
