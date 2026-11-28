"""International Standard Atmosphere (ISA), troposphere + lower stratosphere (0-20 km)."""
import math

G0 = 9.80665          # m/s^2
R_AIR = 287.05287     # J/(kg K)
GAMMA = 1.4
T0, P0 = 288.15, 101325.0
RHO0 = P0 / (R_AIR * T0)   # 1.225 kg/m^3
LAPSE = 0.0065        # K/m
H_TROP = 11000.0      # m

FT = 0.3048


def isa(h_m: float):
    """Return (T [K], p [Pa], rho [kg/m^3], a [m/s]) at ISA altitude h_m (0-20 km)."""
    if not 0.0 <= h_m <= 20000.0:
        raise ValueError("ISA model valid for 0-20 km")
    if h_m <= H_TROP:
        T = T0 - LAPSE * h_m
        p = P0 * (T / T0) ** (G0 / (LAPSE * R_AIR))
    else:
        T11 = T0 - LAPSE * H_TROP
        p11 = P0 * (T11 / T0) ** (G0 / (LAPSE * R_AIR))
        T = T11
        p = p11 * math.exp(-G0 * (h_m - H_TROP) / (R_AIR * T11))
    rho = p / (R_AIR * T)
    a = math.sqrt(GAMMA * R_AIR * T)
    return T, p, rho, a
