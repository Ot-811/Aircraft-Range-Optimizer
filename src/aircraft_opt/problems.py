"""Scaled problems: FullProblem (5 variables) and ReducedProblem (W_fuel eliminated, 4 variables)."""
from __future__ import annotations

import numpy as np

from .model import AircraftModel, VAR_NAMES
from .params import Params

X_REF = np.array([125.0, 10.0, 230.0, 20_000.0, 78_000.0])


class FullProblem:
    def __init__(self, p: Params, include_bounds_as_ineq: bool = False):
        self.m = AircraftModel(p)
        self.p = p
        self.xref = X_REF.copy()
        self.n = 5
        self.var_names = list(VAR_NAMES)
        self.include_bounds = include_bounds_as_ineq
        self.eq_names = ["weight_buildup"]
        names = list(self.m.ineq_names)
        if include_bounds_as_ineq:
            names += [f"{v}_lb" for v in VAR_NAMES] + [f"{v}_ub" for v in VAR_NAMES]
        self.ineq_names = names

    def x(self, z):
        return np.asarray(z, float) * self.xref

    def z(self, x):
        return np.asarray(x, float) / self.xref

    def f(self, z):
        return self.m.f(self.x(z))

    def grad(self, z):
        return self.m.grad_f(self.x(z)) * self.xref

    def eq(self, z):
        return np.array([self.m.h(self.x(z))])

    def jac_eq(self, z):
        return (self.m.jac_h(self.x(z)) * self.xref)[None, :]

    def ineq(self, z):
        x = self.x(z)
        g = self.m.g(x)
        if self.include_bounds:
            g = np.concatenate([g, (self.m.lb - x) / self.xref, (x - self.m.ub) / self.xref])
        return g

    def jac_ineq(self, z):
        x = self.x(z)
        J = self.m.jac_g(x) * self.xref[None, :]
        if self.include_bounds:
            J = np.vstack([J, -np.eye(5), np.eye(5)])
        return J

    def bounds_z(self):
        return list(zip(self.m.lb / self.xref, self.m.ub / self.xref))


class ReducedProblem:
    """y = (S, AR, V, W_i); W_fuel follows from the weight build-up, so every iterate is consistent."""
    IDX = [0, 1, 2, 4]   # positions of y inside x

    def __init__(self, p: Params):
        self.m = AircraftModel(p)
        self.p = p
        self.yref = X_REF[self.IDX].copy()
        self.n = 4
        self.var_names = ["S", "AR", "V", "W_i"]
        self.eq_names = []
        self.ineq_names = (list(self.m.ineq_names)
                           + [f"{v}_lb" for v in VAR_NAMES] + [f"{v}_ub" for v in VAR_NAMES])

    def fuel(self, y):
        S, AR, V, Wi = y
        p = self.p
        return Wi - p.W_fixed - p.k_w * S ** p.a * AR ** p.b * Wi ** p.c - p.W_res - p.W_pay

    def x_of_y(self, y):
        S, AR, V, Wi = y
        return np.array([S, AR, V, self.fuel(y), Wi])

    def dx_dy(self, y):
        S, AR, V, Wi = y
        p = self.p
        Ww = p.k_w * S ** p.a * AR ** p.b * Wi ** p.c
        T = np.zeros((5, 4))
        T[0, 0] = T[1, 1] = T[2, 2] = T[4, 3] = 1.0
        T[3] = [-p.a * Ww / S, -p.b * Ww / AR, 0.0, 1.0 - p.c * Ww / Wi]
        return T

    def x(self, z):
        return self.x_of_y(np.asarray(z, float) * self.yref)

    def z_from_x(self, x):
        return np.asarray(x, float)[self.IDX] / self.yref

    def f(self, z):
        y = np.asarray(z, float) * self.yref
        if np.any(y <= 0):
            return np.inf
        return self.m.f(self.x_of_y(y))

    def grad(self, z):
        y = np.asarray(z, float) * self.yref
        return (self.m.grad_f(self.x_of_y(y)) @ self.dx_dy(y)) * self.yref

    def eq(self, z):
        return np.zeros(0)

    def jac_eq(self, z):
        return np.zeros((0, self.n))

    def ineq(self, z):
        y = np.asarray(z, float) * self.yref
        x = self.x_of_y(y)
        return np.concatenate([self.m.g(x), (self.m.lb - x) / X_REF, (x - self.m.ub) / X_REF])

    def jac_ineq(self, z):
        y = np.asarray(z, float) * self.yref
        x = self.x_of_y(y)
        Jx = np.vstack([self.m.jac_g(x), -np.diag(1 / X_REF), np.diag(1 / X_REF)])
        return (Jx @ self.dx_dy(y)) * self.yref[None, :]
