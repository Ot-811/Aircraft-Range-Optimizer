# Multivariable Constrained Optimisation of Aircraft Design for Maximum Flight Range

AIT 203 Optimisation project. The code chooses wing area `S`, aspect ratio `AR`,
cruise speed `V`, fuel weight `W_fuel` and initial cruise weight `W_i` to maximise
Breguet range under weight, stall, fuel-volume, Mach and (optionally) gate-span constraints.
It is calibrated to the Boeing 737-800.

**Headline result:** with the 737-800's payload, MTOW and fuselage, the model's optimum wing
(S = 132 m², AR = 11.8, M 0.80) flies **5,705 km vs 5,436 km (+5.0 %)**. Under the 36 m ICAO
code C span limit the gain is **+4.2 %** (S = 128 m², AR = 10.1). The custom augmented-Lagrangian
solver and SciPy SLSQP agree to about 2·10⁻⁸ relative. All 40 multi-start runs reach the same optimum.
See [`docs/report.md`](docs/report.md) for the full write-up and [`results/summary.md`](results/summary.md)
for every number.

## Layout

```
src/aircraft_opt/
  atmosphere.py     ISA atmosphere (rho, speed of sound vs altitude)
  params.py         parameters, 737-800 published data, Raymer wing weight, calibration
  model.py          Breguet range, drag polar, weight build-up, constraints + analytic gradients
  problems.py       scaled full (5-var) and reduced (W_fuel eliminated, 4-var) formulations
  unconstrained.py  own GD / Newton / modified Newton / BFGS / SR1 with Armijo backtracking
  alm.py            own augmented Lagrangian method (PHR) for the constraints
  reference.py      SciPy SLSQP reference solver, independent KKT + 2nd-order check, sensitivities
  studies.py        method comparison, multistart, payload-range, parameter sweeps
scripts/run_all.py  runs every study and writes results/ (about 90 s)
tests/              gradient checks, calibration check, custom-vs-SLSQP agreement, KKT
results/figures/    9 PNG figures
results/tables/     CSV / JSON tables
results/summary.md  auto-generated tables of all results
docs/report.md      project report (formulation, method, results, discussion)
docs/assumptions.md every assumed value and default chosen where the proposal was silent
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python -m pytest                     # 6 tests, < 1 s
python scripts/run_all.py            # full study, ~90 s (use --quick for ~20 s)
```

Quick use from Python:

```python
import sys; sys.path.insert(0, "src")
from aircraft_opt.params import calibrated_params
from aircraft_opt.studies import optimize

p = calibrated_params()                 # 737-800-calibrated parameters
r = optimize(p.with_(b_max=36.0))       # ALM + BFGS, cross-checked with SLSQP
print(r["design"]["range_km"], r["active"], r["rel_diff_vs_slsqp"])
```

`Params.with_(...)` changes any parameter (`MTOW`, `W_pay`, `cT`, `k_w`, `V_stall_max`,
`h_cruise`, `M_max`, `b_max`, bounds...). `optimize(..., method="sr1")` picks the inner solver
(`gd`, `newton`, `modnewton`, `bfgs`, `sr1`).
