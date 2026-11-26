"""Fixed parameters of the conceptual model and the Boeing 737-800 reference data.

All masses are in kg (weights are mass x g where a force is needed), lengths in m,
speeds in m/s. Every number that is an assumption rather than a published
specification is marked ASSUMED; see docs/assumptions.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from .atmosphere import isa, FT, RHO0

LB = 0.45359237
FT2 = 0.09290304
NMI = 1.852  # km


# ---------------------------------------------------------------------------
# Published Boeing 737-800 data (Boeing airport-planning document / public specs).
# Checked Nov 2026 against the Boeing 737NG specification values quoted on Wikipedia
# ("Boeing 737 Next Generation", specifications + comparison table).
# ---------------------------------------------------------------------------
B737 = dict(
    MTOW=79_016.0,          # kg (174,200 lb), maximum take-off weight                     [published]
    MZFW=62_732.0,          # kg (138,300 lb), maximum zero-fuel weight                     [published]
    OEW=41_413.0,           # kg (91,300 lb), operating empty weight                        [published]
    fuel_capacity_L=26_020.0,  # L (6,875 US gal), usable fuel (wing + centre tanks)        [published]
    S=124.6,                # m^2 (1,341 ft^2), reference wing area                         [published]
    span=34.32,             # m (112 ft 7 in), span without winglets                        [published]
    M_cruise=0.785,         # typical cruise Mach (MMO is 0.82)                             [published]
    h_cruise_ft=35_000.0,   # typical cruise altitude                                       [typical]
    range_nmi=2_935.0,      # range with 162 passengers (two-class)                         [published]
    pax=162,
    TSFC_cruise=0.627,      # lb/(lbf h), CFM56-7B cruise TSFC. Literature range 0.60-0.65;
                            # only the starting point of the c_T calibration                [literature]
)
B737["AR"] = B737["span"] ** 2 / B737["S"]
B737["range_km"] = B737["range_nmi"] * NMI

FUEL_DENSITY = 0.80  # kg/L, Jet A-1 (ASSUMED typical value)


def raymer_transport_wing_mass(Wdg_kg, S_m2, AR, n_ult=3.75, tc=0.12, taper=0.278,
                               sweep_deg=25.0, Scs_frac=0.10):
    """Raymer (Aircraft Design, eq. 15.25) transport wing weight, returned in kg.

    W = 0.0051 (Wdg Nz)^0.557 Sw^0.649 A^0.5 (t/c)^-0.4 (1+lambda)^0.1 cos(L)^-1 Scs^0.1
    in lb / ft^2 units. Control-surface area Scs is taken proportional to Sw, which
    makes the overall area exponent 0.749.
    737-800 inputs: n_ult = 1.5 x 2.5 g limit load; taper 0.278 and 25 deg quarter-chord sweep
    (published 737NG planform); t/c = 0.12 is an ASSUMED average (root is thicker).
    """
    Wdg = Wdg_kg / LB
    Sw = S_m2 / FT2
    Scs = Scs_frac * Sw
    W = (0.0051 * (Wdg * n_ult) ** 0.557 * Sw ** 0.649 * AR ** 0.5 * tc ** -0.4
         * (1 + taper) ** 0.1 / math.cos(math.radians(sweep_deg)) * Scs ** 0.1)
    return W * LB


@dataclass(frozen=True)
class Params:
    # mission / operator requirements
    W_pay: float = 162 * 100.0        # kg, 162 pax x 100 kg incl. bags (ASSUMED standard mass)
    W_res: float = 2_500.0            # kg, reserve fuel, fixed (ASSUMED)
    MTOW: float = B737["MTOW"]
    # aerodynamics
    CD0: float = 0.020                # parasitic drag coefficient (ASSUMED, typical narrow-body)
    e: float = 0.80                   # Oswald efficiency (ASSUMED)
    CLmax: float = 2.6                # landing-configuration CLmax (ASSUMED, flapped)
    V_stall_max: float = 64.0         # m/s (~124 kt) maximum allowed stall speed (ASSUMED)
    M_max: float = 0.80               # Mach limit used in place of a wave-drag model (ASSUMED)
    h_cruise: float = B737["h_cruise_ft"] * FT   # m
    # propulsion: thrust-specific fuel consumption c_T in 1/s (weight of fuel / thrust / s)
    cT: float = B737["TSFC_cruise"] / 3600.0
    # structure: W_wing = k_w S^a AR^b W_i^c  (Raymer exponents)
    a: float = 0.749
    b: float = 0.5
    c: float = 0.557
    k_w: float = float("nan")         # calibrated (see calibrated_params)
    W_fixed: float = float("nan")     # calibrated: OEW minus wing
    # fuel volume: W_fuel + W_res <= k_f S^1.5 / sqrt(AR)
    k_f: float = float("nan")         # calibrated so the 737-800 tank capacity is reproduced
    # optional airport gate-span limit (ICAO code C = 36 m); None = not imposed
    b_max: float | None = None
    # variable bounds x = (S, AR, V, W_fuel, W_i)
    lb: tuple = (60.0, 5.0, 150.0, 500.0, 50_000.0)
    ub: tuple = (250.0, 16.0, 300.0, 40_000.0, 100_000.0)

    # derived atmospheric quantities -------------------------------------
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
    """The 737-800 reference design as a point x = (S, AR, V, W_fuel, W_i) in model variables.

    It is flown with full tanks: total fuel = tank capacity, of which W_res is reserve,
    and carries the payload p.W_pay (162 pax by default).
    """
    cap = B737["fuel_capacity_L"] * FUEL_DENSITY
    Wi = B737["OEW"] + p.W_pay + cap
    return (B737["S"], B737["AR"], B737["M_cruise"] * p.a_sound, cap - p.W_res, Wi)


def calibrated_params(calibrate_cT: bool = True, **overrides) -> Params:
    """Return Params with k_w, W_fixed, k_f (and optionally c_T) calibrated to the 737-800.

    * k_w : wing mass from Raymer's formula at the 737-800 geometry and MTOW, expressed
            in the model's k_w S^a AR^b W_i^c form at the baseline W_i.
    * W_fixed : OEW minus that wing mass.
    * k_f : chosen so that k_f S^1.5/sqrt(AR) equals the published usable fuel capacity.
    * c_T : (optional) effective cruise c_T that makes the Breguet range of the baseline
            equal the published range. It absorbs taxi/climb/descent fuel, winglets and
            all other effects the simplified model leaves out.

    The calibration always flies the 737-800's own mission (162 pax, published MTOW, FL350).
    Overrides of the mission (W_pay, MTOW, h_cruise, ...) are applied AFTER calibrating, so
    they ask "what if" questions instead of silently re-fitting the constants. Overrides of
    model constants (CD0, e, W_res, exponents) are used in the calibration.
    """
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


# ---------------------------------------------------------------------------
# Where every model parameter comes from (printed in the reports).
#   published  = Boeing / engine data          assumed    = our choice, see docs/assumptions.md
#   calibrated = fitted so the model reproduces the 737-800        physics = standard constant
# ---------------------------------------------------------------------------
PARAMETER_SOURCES = [
    # (attribute of Params, description, unit, factor to display unit, source)
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
