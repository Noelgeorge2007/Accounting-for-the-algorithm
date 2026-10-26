# Accounting for the algorithm

Code accompanying the manuscript:

> Kalathil Sahadevan, V., A, U., George, N., & V, D. *Accounting for the algorithm: AI-assisted multi-objective sustainability assessment of an imperfect four-echelon dairy supply chain with whey valorisation and wastewater control.* Submitted to *Sustainable Production and Consumption* (VSI: AI for Sustainability).

The repository contains an integrated production–inventory model of a four-echelon dairy supply chain (farm → milk processor → cheese plant → retailer) with random defect rates and five sustainability investments, coupled with life-cycle-based carbon and water footprints per kg of cheese. An AI layer optimises, approximates and explains the model, and measures its own computational footprint.

## Contents

| File | Purpose |
|---|---|
| `model.py` | Exact model. Returns expected average integrated total profit (renewal-reward), carbon footprint (with or without milk-solids allocation, `alloc=1/0`), water footprint (blue + grey) and a waste indicator for any decision vector. All base-case parameters are in `BASE`. |
| `run.py` | All experiments reported in the paper (see below). Writes `results.json`. |
| `figs.py` | Builds the figures from `results.json` into `figures/` and writes `cf_breakdown.json`. |
| `requirements.txt` | Package versions used for the reported results. |

Decision vector: `x = [Q, G0, G1, mu, gam, G2]`, i.e. cheese lot size (kg) and annualised investments (k$/yr) in farm methane mitigation, processor renewables, whey valorisation, wastewater treatment and plant energy efficiency.

## Experiments in `run.py`

| Stage | Method |
|---|---|
| S1 | Genetic algorithm (pymoo), 30 seeds, maximising expected integrated profit |
| S2 | Carbon price scenarios (0–150 $/t CO2e), GA re-optimisation |
| S2b | One-at-a-time ±20% sensitivity, GA re-optimisation |
| S3 | NSGA-II (profit, CF, WF), 5 seeds, hypervolume, entropy-weighted TOPSIS |
| S4 | Latin hypercube sampling (20,000 points) and XGBoost / MLP surrogates |
| S5 | SHAP on surrogates vs Sobol total-order indices on the exact model (SALib) |
| S6 | Robust NSGA-II under parameter uncertainty: exact model vs surrogate |
| S7 | Computational footprint of every stage (Green Algorithms method) |

## Reproducing the results

Python 3.10 or later (results were produced with Python 3.13).

```bash
pip install -r requirements.txt
python run.py      # ~3–5 min on 2 CPU cores; writes results.json
python figs.py     # writes figures/*.pdf and cf_breakdown.json
```

Quick check of the model alone:

```python
import numpy as np
from model import evaluate
r = evaluate(np.array([[111972, 7.665, 0, 95.389, 30.019, 0]]), detail=True)
print(r["AITP"], r["CF"], r["WF"])
```

Notes

- Expectations over the random defect fractions use tensor Gauss–Legendre quadrature (6 nodes per dimension by default; set the environment variable `GL_N` to change it).
- Optimisation results are seeded and reproducible. Runtime, energy and emission figures from stage S7 depend on the hardware and will differ on other machines. The footprint constants (per-core power, PUE, grid factor) are set at the end of `run.py`.
- Economic parameters in `BASE` are illustrative. Replace them with site data for applied use. Sources for the environmental factors are given in the manuscript.

## Citation

If you use this code, please cite the paper above (full reference to be added on publication).
