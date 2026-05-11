# ⬡ NEXUS

**Nonlinear Exploratory Framework for X-domain Unified Systems**

> *"Any bounded dynamical system has a statistical attractor. Regime stress is departure from that attractor."*

[![Beta](https://img.shields.io/badge/status-beta%20v0.1-orange)](https://github.com/SAMYAK-2006/nexus)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

NEXUS is a domain-agnostic engine for the analysis, characterization, and forecasting of **nonlinear stochastic time series**. Give it any time series — financial, physiological, physical, or synthetic — and it produces a comprehensive analysis report with trust-weighted ensemble forecasts.

Built on the **Statistical Manifold Attractor (SMA) Framework** ([Jain, 2026](docs/NEXUS_technical_reference.pdf)).

---

## What it does

NEXUS answers questions classical time series methods don't:

- **Where in statistical parameter space is this system right now?**
- **How fast is it moving? Is it in an escape phase (crisis)?**
- **How much should we trust historical patterns at this moment?**
- **What are the next *n* values, accounting for current regime?**

It does this through four layers:

```
Input series
    │
    ▼
Layer 1 — Geometric Engine (SMA)
    Computes θ(t), speed s(t), attractor θ*, trust score α(t), regimes
    │
    ▼
Layer 2 — Statistical Analysis
    Hurst exponent, nonlinearity tests, spectral analysis, changepoints, risk
    │
    ▼
Layer 3 — Trust-Weighted Ensemble Forecast
    ARIMA + OU-SDE + Koopman/DMD + SMA-GradBoost, weighted by α(t)
    │
    ▼
Layer 4 — Output
    Self-contained HTML report  OR  real-time per-tick snapshots
```

---

## Quick start

```bash
git clone https://github.com/SAMYAK-2006/nexus.git
cd nexus
pip install -r requirements.txt

# Run on synthetic demo data
python run_nexus.py --demo

# Run on your own data
python run_nexus.py --file data.csv --column close --horizon 30
```

Open `nexus_report.html` in any browser. That's your full output.

---

## The report contains

1. **SMA applicability diagnostics** — does the attractor framework apply to this series?
2. **Series overview** — raw series with regime coloring, escape phase markers, trust score α(t), attractor distance D(t), displacement persistence P(t,W)
3. **Geometric portrait** — statistical manifold trajectory (PC1 vs PC2), stress dimensionality, regime occupancy
4. **Statistical analysis** — distribution, ACF, volatility clustering, Q-Q plot, Hurst R/S analysis
5. **Key insights** — auto-generated interpretive statements from all test results
6. **Symmetry & spectral** — PSD, spectrogram, phase portrait, time reversibility, anomaly detection
7. **Forecast** — ensemble forecast + 80/95% CI + model weight table

---

## Real-time mode

```python
from nexus.realtime.engine import RealTimeEngine

engine = RealTimeEngine(window=60, horizon=20)
engine.initialize(historical_series)

for new_value in live_stream:
    snap = engine.update(new_value)
    print(snap.trust_score, snap.regime, snap.forecast)
```

Per-tick: regime label, trust score α(t), speed s(t), escape phase flag, n-step forecast.

---

## Mathematical core

The SMA framework represents the statistical state of any process as a trajectory θ(t) on a parameter manifold Θ ⊆ ℝᵏ. Key quantities:

| Quantity | Symbol | Meaning |
|---|---|---|
| Feature vector | θ(t) | 14-dimensional rolling statistical summary |
| Speed | s(t) = ‖θ(t) − θ(t−1)‖ | Rate of statistical change |
| Attractor | θ* | Long-run mean of θ(t) |
| Attractor distance | D(t) = ‖θ(t) − θ*‖ | Current stress level |
| Trust score | α(t) ∈ (0,1) | How much to trust historical patterns |
| Persistence | P(t,W) ∈ [−1,1] | Momentum (+) vs mean-reversion (−) |

See [`docs/NEXUS_technical_reference.pdf`](docs/NEXUS_technical_reference.pdf) for the full mathematical treatment.

---

## Prediction models

| Model | Key idea | Best when |
|---|---|---|
| ARIMA | Auto-selected Box-Jenkins | Stationary, linear |
| OU-SDE | Ornstein-Uhlenbeck analytical forecast | Mean-reverting, stressed (low α) |
| Koopman/DMD | Linear operator on delay-embedded space | Nonlinear, structured |
| SMA-GradBoost | Gradient boosting on geometric features | High α(t), data-rich |

All four are combined in a trust-weighted ensemble. When α(t) is low (crisis/escape), the OU structural prior is boosted automatically.

---

## Project status — Beta v0.1

**Works:**
- Full SMA geometric pipeline
- 14-test statistical analysis battery
- 4-model trust-weighted ensemble
- HTML report with 5 visualization dashboards
- Real-time streaming engine

**Pending / known limitations:**
- Univariate only (multivariate extension in v0.2)
- No GARCH volatility module yet
- Benchmark validation on real datasets pending
- Fisher-Rao metric not yet implemented (Euclidean used)

See [`ROADMAP.md`](ROADMAP.md) for the full v0.2 plan.

---

## Theoretical foundations

- **SMA Framework**: Jain (2026) — *Statistical Manifold Geometry of Regime Dynamics*
- **Koopman/DMD**: Schmid (2010), Pan et al. (2024)
- **OU Process**: Uhlenbeck & Ornstein (1930)
- **Changepoint detection**: Killick et al. (2012) via `ruptures`

---

## Repository structure

```
nexus/
├── run_nexus.py              # Entry point
├── requirements.txt
├── docs/
│   ├── NEXUS_technical_reference.pdf   # Full technical guide
│   └── NEXUS_technical_reference.tex  # LaTeX source
├── nexus/
│   ├── core/geometric.py     # SMA engine
│   ├── analysis/analyzer.py  # Statistical analysis
│   ├── prediction/engine.py  # Prediction models + ensemble
│   ├── report/generator.py   # HTML report
│   └── realtime/engine.py    # Real-time streaming
├── tests/                    # (pending)
└── experiments/              # (pending)
```

---

## Author

**Samyak Jain** — NIT Rourkela
Interests: nonlinear stochastic systems, geometric ML, control theory, financial modeling

[github.com/SAMYAK-2006](https://github.com/SAMYAK-2006)

---

*NEXUS is a research prototype. Production use requires validation on your specific domain.*
