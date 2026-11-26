# Assumptions and defaults

The proposal fixes the formulation and methods. The values below are the defaults chosen
where it gives no number. Everything is in `src/aircraft_opt/params.py` and can be changed with
`Params.with_(...)`.

## Modelling assumptions (from the proposal)

- Constant L/D and c_T during cruise, at a fixed altitude (Breguet jet range). C_L is
  evaluated at the initial cruise weight W_i.
- Fixed reserve fuel W_res. Take-off, climb and descent are not modelled. Their fuel is absorbed
  into the calibrated effective c_T.
- Drag polar C_D = C_D0 + C_L²/(π e AR). There is no wave drag; a Mach limit replaces it.
- Wing mass from Raymer's transport-wing relation W_wing = k_w S^a AR^b W_i^c.
- Fuel must fit in the wing: W_fuel + W_res ≤ k_f S^1.5 / √AR (volume of a wing of fixed t/c
  scales as S·c̄ ∝ S^1.5/√AR).
- The stall constraint uses W_i, as written in the proposal. This is conservative, because
  real landing weight is lower.

## Published 737-800 data used (checked November 2026)

Every value below was re-checked against Boeing's published 737-800 specification (as quoted in
the Wikipedia "Boeing 737 Next Generation" specification and variant-comparison tables, which cite
Boeing). `tests/test_parameters_and_reports.py` checks each lb/ft²/gal → SI conversion.

| Item | Value | Published (imperial) | Status |
|---|---|---|---|
| MTOW | 79,016 kg | 174,200 lb | ✓ matches |
| MZFW | 62,732 kg (caps payload in the payload-range diagram) | 138,300 lb | from Boeing airport-planning data; not re-checked online |
| OEW | 41,413 kg | 91,300 lb | ✓ matches |
| Usable fuel | 26,020 L | 6,875 US gal | ✓ matches |
| Wing area | 124.6 m² | 1,341.2 ft² | ✓ matches |
| Span (no winglets) | 34.32 m, so AR = 9.45 | 112 ft 7 in | ✓ matches (35.79 m with winglets) |
| Typical cruise | M 0.785 at FL350 | MMO = 0.82 | ✓ typical value; MMO matches |
| Published range | 2,935 nmi = 5,436 km with 162 passengers | 2,935 nmi, 162 pax (two-class) | ✓ matches. Boeing's current marketing page rounds to "up to 2,800 nmi" |
| CFM56-7B cruise TSFC (literature) | 0.627 lb/lbf/h | 0.60-0.65 in the literature | plausible, no single primary source. Only the starting point of the c_T calibration, so it does not affect the design |
| Wing planform (Raymer inputs) | taper 0.278, quarter-chord sweep 25° | 737NG data file | ✓ matches |

## Assumed values (defaults where the proposal is silent)

| Parameter | Value | Reason |
|---|---|---|
| Payload W_pay | 162 × 100 kg = 16,200 kg | standard passenger + bag mass |
| Reserve fuel W_res | 2,500 kg | typical narrow-body contingency + alternate + 30 min hold |
| Fuel density | 0.80 kg/L | Jet A-1 |
| C_D0 | 0.020 | typical narrow-body cruise |
| Oswald e | 0.80 | typical swept wing |
| C_L,max (landing) | 2.6 | flapped narrow-body, consistent with 737 approach speeds |
| V_stall,max | 64 m/s (124 kt) | keeps the 737 baseline (62.3 m/s) feasible; varied 52-68 m/s in the study |
| M_max | 0.80 | slightly above the 0.785 cruise, below MMO 0.82 |
| Cruise altitude | 35,000 ft (ISA ρ = 0.380 kg/m³, a = 296.5 m/s) | typical |
| Raymer wing inputs | N_ult = 3.75 (1.5 × 2.5 g), t/c = 0.12 (average; the root is thicker), S_cs = 0.1 S | 737-class values. t/c only changes the calibrated k_w; it is absorbed by the calibration |
| Exponents a, b, c | 0.749, 0.5, 0.557 | Raymer (S_cs ∝ S adds 0.1 to the area exponent) |
| Gate-span limit | 36 m (ICAO code C), separate study | from the proposal |
| Bounds | S 60-250 m², AR 5-16, V 150-300 m/s, W_fuel 0.5-40 t, W_i 50-100 t | wide; none active at the optimum |

## Calibration (consistency check, not validation)

- k_w: Raymer wing mass at the 737-800 geometry and MTOW is 5,691 kg. k_w is set so the model
  reproduces this at the baseline W_i. W_fixed = OEW − W_wing = 35,722 kg.
- k_f: set so the 737-800 wing holds the published 20,816 kg. The real aircraft also has a
  centre tank; the model treats total capacity as wing volume.
- c_T: with the literature TSFC the model predicts 6,068 km (+11.6 % vs published). The
  effective c_T = 0.700 lb/lbf/h reproduces 5,436 km and is used in all studies. It absorbs
  taxi, climb and descent fuel, winglets, and every other effect the model leaves out.
- The baseline is flown with full tanks (W_i = 78,429 kg < MTOW). In this condition the
  fuel-volume constraint is active.

## Numerical defaults

- Variables scaled by x_ref = (125 m², 10, 230 m/s, 20 t, 78 t).
- ALM: μ₀ = 10, ×10 when the violation does not fall by 4×, inner tolerance 10^-(k+1) down to
  1e-8. Stops when violation, complementarity ≤ 1e-8 and stationarity ≤ 1e-6.
- Armijo: c₁ = 1e-4, halving, α₀ = 1.
- Newton-type methods use the central-difference Hessian of the analytic gradient (h = 1e-6).
  Pure Newton falls back to steepest descent when its direction is not a descent direction,
  and the fallbacks are counted.
- Modified Newton (lecture 13): if λ_min(H) ≤ 0, use H + ηI with η = 10⁻⁶ − λ_min; otherwise H
  is used unchanged (so on convex subproblems it is identical to Newton).
- BFGS and SR1 both update the inverse Hessian B (d = −Bg). The first step scales B = (sᵀy/yᵀy) I.
  If B gives a non-descent direction it is reset to I and counted as a fallback (this happens to
  SR1, whose B is not guaranteed positive definite; BFGS keeps B positive definite when sᵀy > 0).
- Multi-start: 40 Latin-hypercube points, kept only when they give positive fuel.
