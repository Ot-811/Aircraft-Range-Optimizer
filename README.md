# Multivariable Constrained Optimisation of Aircraft Design for Maximum Flight Range

AIT 203 Optimisation project. The code chooses wing area `S`, aspect ratio `AR`,
cruise speed `V`, fuel weight `W_fuel` and initial cruise weight `W_i` to maximise
Breguet range under weight, stall, fuel-volume, Mach and (optionally) gate-span constraints.
It is calibrated to the Boeing 737-800.

**Headline result:** with the 737-800's payload, MTOW and fuselage, the model's optimum wing
(S = 132 m², AR = 11.8, M 0.80) flies **5,705 km vs 5,436 km (+5.0 %)**. Under the 36 m ICAO
code C span limit the gain is **+4.2 %** (S = 128 m², AR = 10.1). Our own augmented-Lagrangian
solver and SciPy SLSQP agree to about 2·10⁻⁸ (relative). All 40 multi-start runs reach the same optimum.
The full, auto-generated report is [`results/report.md`](results/report.md) (also as PDF).

## Two separate programs

| | Main study | Demo (for the viva) |
|---|---|---|
| Run | `python scripts/run_all.py` | `python scripts/demo.py` |
| What it does | every study in the proposal: optimum, solver comparison with measured convergence rates, KKT + sensitivities, multi-start, payload-range, 4 parameter sweeps; plus a design-space picture and a ±10 % assumption (tornado) study | asks for your requirements and **your own starting design**, optimises it and checks the result |
| Output | `results/` (tables, figures, `results.json`) | `demo_reports/<run name>/` |
| Report | `results/report.md` + `.pdf`, built by `scripts/main_report.py` | `demo_reports/<name>/report.md` + `.pdf`, built by `scripts/demo_report.py` |
| Rebuild only the report | `python scripts/main_report.py` | `python scripts/demo_report.py demo_reports/<name>` |

Computing and reporting are separate. The report scripts only read the saved `results.json`,
and every number and every result-dependent sentence in a report is generated from it, so a
report can never disagree with the data. The PDF needs `pdflatex` (MiKTeX or TeX Live). Without
it you still get the Markdown report and the `.tex` file.

## Running

```bash
pip install -r requirements.txt
python -m pytest                     # 26 tests, a few seconds
python scripts/run_all.py            # full study + report, about 1-2 min (--quick: about 35 s)
python scripts/demo.py               # interactive demo; Enter keeps the 737-800 value
python scripts/demo.py --defaults    # demo with no questions (run name 'demo')
```

**Demo tip:** give it a start that breaks the rules, e.g. a 50 m span on 100 m² with 25,000 kg of
fuel (saved example: `demo_reports/bad_start`). The report shows that this design "flies" twice as
far only because it is not allowed: too heavy, it stalls too fast, and the fuel does not fit in
the wing. The history plot shows the range falling as the optimiser forces the violation to zero.
Requirements that contradict each other (e.g. 213 passengers with a 60 t MTOW, saved as
`demo_reports/impossible`) give a report that says no allowed design exists. `demo_reports/composite_wing`
shows a lighter (composite) wing structure: 6,062 km, while the real 737-800 still flies 5,436 km.

## Layout

```
src/aircraft_opt/
  atmosphere.py     ISA atmosphere (rho, speed of sound vs altitude)
  params.py         parameters, 737-800 published data, Raymer wing weight, calibration, parameter sources
  model.py          Breguet range, drag polar, weight build-up, constraints + analytic gradients
  problems.py       scaled full (5-var) and reduced (W_fuel eliminated, 4-var) formulations
  unconstrained.py  our own GD / Newton / modified Newton / BFGS / SR1 with Armijo backtracking
  alm.py            our own augmented Lagrangian method (PHR) for the constraints
  reference.py      SciPy SLSQP reference solver, independent KKT + second-order check, sensitivities
  studies.py        method comparison, multi-start, payload-range, parameter sweeps
  report_writer.py  small report builder: one list of blocks -> Markdown and LaTeX/PDF
scripts/
  run_all.py        main study -> results/
  main_report.py    main report <- results/results.json
  demo.py           interactive demo -> demo_reports/<name>/
  demo_report.py    demo report <- demo_reports/<name>/results.json
tests/              gradient checks, published-data conversions, ISA, calibration,
                    custom-vs-SLSQP agreement, KKT, solver convergence, LaTeX escaping,
                    regression tests for fixed bugs, end-to-end demo run, new analyses
docs/assumptions.md every assumed value, and how each published value was checked
docs/viva_guide.md  how the code fits together, and likely viva questions with answers
```

## Quick use from Python

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
