"""Model parameters, Boeing 737-800 data, Raymer wing mass and calibration (masses in kg)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from .atmosphere import isa, FT, RHO0

LB = 0.45359237
FT2 = 0.09290304
NMI = 1.852  # km


B737 = dict(
    MTOW=79_016.0,  # kg (174,200 lb)
    MZFW=62_732.0,  # kg (138,300 lb), max zero-fuel weight
    OEW=41_413.0,  # kg (91,300 lb), operating empty weight
    fuel_capacity_L=26_020.0,  # L (6,875 US gal)
    S=124.6,  # m^2 (1,341 ft^2)
    span=34.32,  # m (112 ft 7 in), no winglets
    M_cruise=0.785,  # MMO is 0.82
    h_cruise_ft=35_000.0,
    range_nmi=2_935.0,  # with 162 passengers
    pax=162,
    TSFC_cruise=0.627,  # lb/(lbf h), literature value; c_T is calibrated
)
B737["AR"] = B737["span"] ** 2 / B737["S"]
B737["range_km"] = B737["range_nmi"] * NMI

FUEL_DENSITY = 0.80  # kg/L, Jet A-1


def raymer_transport_wing_mass(Wdg_kg, S_m2, AR, n_ult=3.75, tc=0.12, taper=0.278,
                               sweep_deg=25.0, Scs_frac=0.10):
    """Raymer eq. 15.25 transport wing mass in kg (S_cs = 0.1 S makes the area exponent 0.749)."""
    Wdg = Wdg_kg / LB
    Sw = S_m2 / FT2
    Scs = Scs_frac * Sw
    W = (0.0051 * (Wdg * n_ult) ** 0.557 * Sw ** 0.649 * AR ** 0.5 * tc ** -0.4
         * (1 + taper) ** 0.1 / math.cos(math.radians(sweep_deg)) * Scs ** 0.1)
    return W * LB


@dataclass(frozen=True)
class Params:
    W_pay: float = 162 * 100.0  # kg, 162 pax x 100 kg
    W_res: float = 2_500.0  # kg, carried, not burnt
    MTOW: float = B737["MTOW"]
    CD0: float = 0.020
    e: float = 0.80
    CLmax: float = 2.6  # landing flaps
    V_stall_max: float = 64.0  # m/s
    M_max: float = 0.80  # replaces wave drag
    h_cruise: float = B737["h_cruise_ft"] * FT  # m
    cT: float = B737["TSFC_cruise"] / 3600.0  # 1/s
    a: float = 0.749
    b: float = 0.5
    c: float = 0.557
    k_w: float = float("nan")  # calibrated
    W_fixed: float = float("nan")  # calibrated: OEW minus wing
    k_f: float = float("nan")  # calibrated: 737 tank capacity
    b_max: float | None = None  # 36 = ICAO code C gate; None = no limit
    lb: tuple = (60.0, 5.0, 150.0, 500.0, 50_000.0)  # bounds on (S, AR, V, W_fuel, W_i)
    ub: tuple = (250.0, 16.0, 300.0, 40_000.0, 100_000.0)

    @property
    def rho(self):
        return isa(self.h_cruise)[2]

    @property
    def a_sound(self):
        return isa(self.h_cruise)[3]

    @property
    def rho0(self):
        return RHO0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


def baseline_x(p: Params):
    """The 737-800 as a design point x = (S, AR, V, W_fuel, W_i), flown with full tanks."""
    cap = B737["fuel_capacity_L"] * FUEL_DENSITY
    Wi = B737["OEW"] + p.W_pay + cap
    return (B737["S"], B737["AR"], B737["M_cruise"] * p.a_sound, cap - p.W_res, Wi)


def calibrated_params(calibrate_cT: bool = True, **overrides) -> Params:
    """Params with k_w, W_fixed, k_f and c_T fitted to the 737-800 mission; mission overrides applied after."""
    p = Params(**overrides)
    ref = p.with_(W_pay=Params.W_pay, MTOW=B737["MTOW"], h_cruise=Params.h_cruise)   # 737 mission
    S, AR = B737["S"], B737["AR"]
    W_wing = raymer_transport_wing_mass(B737["MTOW"], S, AR)
    cap = B737["fuel_capacity_L"] * FUEL_DENSITY
    Wi0 = B737["OEW"] + ref.W_pay + cap
    consts = dict(k_w=W_wing / (S ** ref.a * AR ** ref.b * Wi0 ** ref.c),
                  W_fixed=B737["OEW"] - W_wing,
                  k_f=cap / (S ** 1.5 / math.sqrt(AR)))
    if calibrate_cT and "cT" not in overrides:
        from .model import AircraftModel
        ref = ref.with_(**consts)
        R_pred = AircraftModel(ref).range_km(baseline_x(ref))
        consts["cT"] = ref.cT * R_pred / B737["range_km"]
    return p.with_(**consts)


PARAMETER_SOURCES = [
    ("MTOW", "Maximum take-off mass", "kg", 1.0, "published (737-800)"),
    ("W_pay", "Payload: 162 pax x 100 kg", "kg", 1.0, "assumed (standard pax + bag mass)"),
    ("W_res", "Reserve fuel (carried, not burnt)", "kg", 1.0, "assumed"),
    ("CD0", "Zero-lift drag coefficient", "-", 1.0, "assumed (typical narrow-body)"),
    ("e", "Oswald efficiency", "-", 1.0, "assumed (typical swept wing)"),
    ("CLmax", "Max lift coefficient, landing flaps", "-", 1.0, "assumed (flapped narrow-body)"),
    ("V_stall_max", "Maximum allowed stall speed", "m/s", 1.0, "assumed (737 baseline = 62.3 m/s fits)"),
    ("M_max", "Mach limit (replaces wave drag)", "-", 1.0, "assumed (between cruise 0.785 and MMO 0.82)"),
    ("h_cruise", "Cruise altitude", "ft", 1 / FT, "typical 737 cruise"),
    ("cT", "Effective cruise TSFC", "lb/(lbf h)", 3600.0, "calibrated (literature 0.627)"),
    ("a", "Wing-mass exponent on S", "-", 1.0, "Raymer eq. 15.25 (0.649 + 0.1 from S_cs)"),
    ("b", "Wing-mass exponent on AR", "-", 1.0, "Raymer eq. 15.25"),
    ("c", "Wing-mass exponent on W_i", "-", 1.0, "Raymer eq. 15.25"),
    ("k_w", "Wing-mass coefficient", "-", 1.0, "calibrated (Raymer mass at 737 geometry)"),
    ("W_fixed", "Fixed mass (OEW minus wing)", "kg", 1.0, "calibrated (published OEW)"),
    ("k_f", "Fuel-volume coefficient", "kg/m^3", 1.0, "calibrated (published tank capacity)"),
]
