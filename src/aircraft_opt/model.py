"""Aircraft model: Breguet range, drag polar, weight build-up, constraints and analytic gradients."""
from __future__ import annotations

import math
import numpy as np

from .atmosphere import G0
from .params import Params

VAR_NAMES = ("S", "AR", "V", "W_fuel", "W_i")


class AircraftModel:
    def __init__(self, p: Params):
        self.p = p
        self.rho = p.rho
        self.a_snd = p.a_sound
        self.q_stall = 0.5 * p.rho0 * p.V_stall_max ** 2 * p.CLmax
        self.ineq_names = ["mtow", "stall", "fuel_volume", "mach"] + (["span"] if p.b_max else [])
        self.lb = np.array(p.lb, float)
        self.ub = np.array(p.ub, float)

    def aero(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        q = 0.5 * self.rho * V ** 2
        CL = Wi * G0 / (q * S)
        k = 1.0 / (math.pi * p.e * AR)
        CD = p.CD0 + k * CL ** 2
        return CL, CD, k

    def wing_mass(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        return p.k_w * S ** p.a * AR ** p.b * Wi ** p.c

    def initial_mass(self, S, AR, W_fuel, iters=50):
        """W_i that closes the weight build-up for a given wing and fuel load (fixed-point iteration)."""
        p = self.p
        Wi = p.W_fixed + W_fuel + p.W_res + p.W_pay
        for _ in range(iters):
            Wi = p.W_fixed + p.k_w * S ** p.a * AR ** p.b * Wi ** p.c + W_fuel + p.W_res + p.W_pay
        return Wi

    def in_domain(self, x):
        S, AR, V, Wf, Wi = x
        return S > 0 and AR > 0 and V > 0 and 0 < Wf < Wi

    def range_km(self, x):
        S, AR, V, Wf, Wi = x
        CL, CD, _ = self.aero(x)
        return V / self.p.cT * (CL / CD) * math.log(Wi / (Wi - Wf)) / 1000.0

    def f(self, x):
        if not self.in_domain(x):
            return math.inf
        return -math.log(self.range_km(x))

    def grad_f(self, x):
        S, AR, V, Wf, Wi = x
        CL, CD, k = self.aero(x)
        B = math.log(Wi / (Wi - Wf))
        dlnE_dCL = 1.0 / CL - 2.0 * k * CL / CD
        dlnE_dk = -CL ** 2 / CD
        g = np.empty(5)
        g[0] = -dlnE_dCL * (-CL / S)
        g[1] = -dlnE_dk * (-k / AR)
        g[2] = -1.0 / V - dlnE_dCL * (-2.0 * CL / V)
        g[3] = -(1.0 / B) / (Wi - Wf)
        g[4] = -dlnE_dCL * (CL / Wi) - (1.0 / B) * (1.0 / Wi - 1.0 / (Wi - Wf))
        return g

    def h(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        return (p.W_fixed + self.wing_mass(x) + Wf + p.W_res + p.W_pay) / Wi - 1.0

    def jac_h(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        Ww = self.wing_mass(x)
        tot = p.W_fixed + Ww + Wf + p.W_res + p.W_pay
        return np.array([p.a * Ww / S / Wi, p.b * Ww / AR / Wi, 0.0, 1.0 / Wi,
                         p.c * Ww / Wi / Wi - tot / Wi ** 2])

    def g(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        out = [Wi / p.MTOW - 1.0,
               Wi * G0 / (S * self.q_stall) - 1.0,
               (Wf + p.W_res) * math.sqrt(AR) / (p.k_f * S ** 1.5) - 1.0,
               V / (p.M_max * self.a_snd) - 1.0]
        if p.b_max:
            out.append(math.sqrt(S * AR) / p.b_max - 1.0)
        return np.array(out)

    def jac_g(self, x):
        S, AR, V, Wf, Wi = x
        p = self.p
        J = np.zeros((len(self.ineq_names), 5))
        J[0, 4] = 1.0 / p.MTOW
        st = Wi * G0 / (S * self.q_stall)
        J[1, 0] = -st / S
        J[1, 4] = G0 / (S * self.q_stall)
        Q = (Wf + p.W_res) * math.sqrt(AR) / (p.k_f * S ** 1.5)
        J[2, 0] = -1.5 * Q / S
        J[2, 1] = 0.5 * Q / AR
        J[2, 3] = Q / (Wf + p.W_res)
        J[3, 2] = 1.0 / (p.M_max * self.a_snd)
        if p.b_max:
            s = math.sqrt(S * AR) / p.b_max
            J[4, 0] = 0.5 * s / S
            J[4, 1] = 0.5 * s / AR
        return J

    def describe(self, x):
        S, AR, V, Wf, Wi = map(float, x)
        CL, CD, _ = self.aero(x)
        p = self.p
        return dict(S=S, AR=AR, span=math.sqrt(S * AR), V=V, Mach=V / self.a_snd,
                    W_fuel=Wf, W_i=Wi, W_wing=self.wing_mass(x), CL=CL, CD=CD, LD=CL / CD,
                    WS=Wi * G0 / S, V_stall=math.sqrt(2 * Wi * G0 / (p.rho0 * S * p.CLmax)),
                    fuel_capacity=p.k_f * S ** 1.5 / math.sqrt(AR),
                    range_km=self.range_km(x), h_resid=self.h(x),
                    **{f"g_{n}": float(v) for n, v in zip(self.ineq_names, self.g(x))})
