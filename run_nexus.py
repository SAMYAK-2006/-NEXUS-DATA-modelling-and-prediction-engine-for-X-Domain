"""
NEXUS — Main Entry Point
=========================
Run this file. That's it.

Usage:
    python run_nexus.py --file your_data.csv --output report.html
    python run_nexus.py --file your_data.csv --column close --horizon 30
    python run_nexus.py --demo   # runs on synthetic data so you can test immediately
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def run_demo():
    """Generate synthetic nonlinear stochastic series and run full pipeline."""
    print("\n[NEXUS] Running on DEMO data (synthetic nonlinear regime-switching series)\n")
    np.random.seed(42)
    n = 800

    # Regime 0: calm OU process
    x = [100.0]
    regime_schedule = (
        [0] * 250 +   # calm
        [1] * 100 +   # crisis
        [0] * 200 +   # recovery
        [2] * 80  +   # trending
        [0] * 170     # calm again
    )
    kappa = [0.05, 0.01, 0.05, 0.0]
    theta_star = [100.0, 80.0, 100.0, 110.0]
    sigma = [0.5, 3.0, 0.8, 1.2]

    for t in range(1, n):
        r = regime_schedule[t]
        drift = kappa[r] * (theta_star[r] - x[-1])
        noise = sigma[r] * np.random.randn()
        # Jump in crisis
        jump = np.random.choice([0, -5, -8], p=[0.97, 0.02, 0.01]) if r == 1 else 0
        x.append(x[-1] + drift + noise + jump)

    return np.array(x), "NEXUS_Demo_Series"


def load_data(filepath: str, column: str = None) -> tuple[np.ndarray, str]:
    """Load CSV or Excel file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.csv', '.txt']:
        df = pd.read_csv(filepath)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV or Excel.")

    print(f"[NEXUS] Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[NEXUS] Columns: {list(df.columns)}")

    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        series = df[column].dropna().values.astype(float)
        name = column
    else:
        # Auto-select: first numeric column that isn't a date/index
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the file.")
        # Skip obvious date/id columns
        skip = ['date', 'time', 'timestamp', 'index', 'id', 'year', 'month', 'day']
        candidates = [c for c in numeric_cols if c.lower() not in skip]
        if not candidates:
            candidates = numeric_cols
        name = candidates[0]
        series = df[name].dropna().values.astype(float)
        print(f"[NEXUS] Auto-selected column: '{name}' ({len(series)} values)")
        if len(numeric_cols) > 1:
            print(f"[NEXUS] Other numeric columns available: {[c for c in numeric_cols if c != name]}")
            print(f"[NEXUS] Use --column <name> to select a different one.\n")

    return series, name


def run_pipeline(series: np.ndarray, name: str, horizon: int, window: int, output: str):
    """Run the full NEXUS pipeline."""

    print(f"\n{'='*60}")
    print(f"  NEXUS — Nonlinear Time Series Analysis Engine")
    print(f"{'='*60}")
    print(f"  Series  : {name}")
    print(f"  Length  : {len(series)}")
    print(f"  Window  : {window}")
    print(f"  Horizon : {horizon}")
    print(f"{'='*60}\n")

    # ── Step 1: Geometric engine (SMA) ─────────────────────────
    print("[1/4] Running geometric engine (SMA framework)...")
    from nexus.core.geometric import GeometricEngine
    geo = GeometricEngine(
        window=window,
        pca_components=3,
        regime_n_clusters=3,
    )
    sma_result = geo.fit_transform(series)
    print(f"      ✓ Applicability: {sma_result.applicability.applicable}")
    print(f"      ✓ Attractor exists: {sma_result.applicability.attractor_exists}")
    print(f"      ✓ Mean reverting: {sma_result.applicability.mean_reverting}")
    print(f"      ✓ E[P(t,W)] = {sma_result.applicability.persistence_score:.4f}")
    print(f"      ✓ Current trust α(t) = {sma_result.trust_score[-1]:.4f}")
    print(f"      ✓ Current regime = {int(sma_result.regime_labels[-1])}")
    for note in sma_result.applicability.notes:
        print(f"      ⚠ {note}")

    # ── Step 2: Statistical analysis ────────────────────────────
    print("\n[2/4] Running full statistical analysis...")
    from nexus.analysis.analyzer import TimeSeriesAnalyzer
    analyzer = TimeSeriesAnalyzer()
    analysis = analyzer.analyze(series)
    print(f"      ✓ Hurst H = {analysis.hurst:.4f}  ({analysis.hurst_interpretation})")
    print(f"      ✓ Stationarity: {'STATIONARY' if analysis.adf_stationary else 'NON-STATIONARY'} (ADF p={analysis.adf_pvalue:.4f})")
    print(f"      ✓ ARCH effects: {'YES' if analysis.arch_effect else 'NO'}")
    print(f"      ✓ Spectral type: {analysis.spectral_complexity}")
    print(f"      ✓ Changepoints: {analysis.n_changepoints}")
    print(f"      ✓ Distribution: {analysis.distribution_fit}")
    print(f"      ✓ Time reversibility: {analysis.time_reversibility:.4f}")
    print(f"\n      Key insights:")
    for ins in analysis.key_insights:
        print(f"        → {ins}")

    # ── Step 3: Forecasting ──────────────────────────────────────
    print(f"\n[3/4] Running trust-weighted ensemble forecast (horizon={horizon})...")
    from nexus.prediction.engine import EnsemblePredictor
    predictor = EnsemblePredictor()
    current_trust = float(sma_result.trust_score[-1])
    current_regime = int(sma_result.regime_labels[-1])
    ensemble = predictor.predict(
        series=series,
        horizon=horizon,
        sma_result=sma_result,
        current_trust=current_trust,
        current_regime=current_regime,
    )
    print(f"      ✓ Models run: {list(ensemble.individual_forecasts.keys())}")
    print(f"      ✓ Trust-weighted: {ensemble.trust_weighted}")
    print(f"      ✓ Model weights: { {k: f'{v:.3f}' for k,v in ensemble.model_weights.items()} }")
    print(f"\n      Forecast (next {min(5, horizon)} steps):")
    for i in range(min(5, horizon)):
        v = ensemble.point_forecast[i]
        l = ensemble.lower_95[i]
        u = ensemble.upper_95[i]
        print(f"        t+{i+1:2d}: {v:.4f}  [95% CI: {l:.4f}, {u:.4f}]")
    if horizon > 5:
        print(f"        ... (see full report for all {horizon} steps)")

    # ── Step 4: Generate report ──────────────────────────────────
    print(f"\n[4/4] Generating full HTML report → {output}")
    from nexus.report.generator import generate_report
    out = generate_report(
        series=series,
        timestamps=None,
        series_name=name,
        sma_result=sma_result,
        analysis_result=analysis,
        ensemble_forecast=ensemble,
        output_path=output,
    )
    print(f"\n{'='*60}")
    print(f"  ✓ DONE. Open your report:")
    print(f"  → {os.path.abspath(out)}")
    print(f"{'='*60}\n")
    return out


def main():
    parser = argparse.ArgumentParser(
        description='NEXUS — Nonlinear Time Series Analysis Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_nexus.py --demo
  python run_nexus.py --file data.csv
  python run_nexus.py --file data.csv --column price --horizon 30
  python run_nexus.py --file data.xlsx --column close --window 30 --output my_report.html
        """
    )
    parser.add_argument('--file', type=str, help='Path to CSV or Excel file')
    parser.add_argument('--column', type=str, default=None, help='Column name to analyze (auto-detected if not specified)')
    parser.add_argument('--horizon', type=int, default=20, help='Forecast horizon (default: 20)')
    parser.add_argument('--window', type=int, default=60, help='Rolling window size (default: 60)')
    parser.add_argument('--output', type=str, default='nexus_report.html', help='Output HTML report path')
    parser.add_argument('--demo', action='store_true', help='Run on synthetic demo data')

    args = parser.parse_args()

    if args.demo:
        series, name = run_demo()
    elif args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        series, name = load_data(args.file, args.column)
    else:
        print("[NEXUS] No input specified. Running demo...\n")
        series, name = run_demo()

    if len(series) < args.window + 20:
        print(f"[WARNING] Series length ({len(series)}) is short for window={args.window}.")
        args.window = max(10, len(series) // 5)
        print(f"[WARNING] Auto-adjusted window to {args.window}.\n")

    run_pipeline(
        series=series,
        name=name,
        horizon=args.horizon,
        window=args.window,
        output=args.output,
    )


if __name__ == '__main__':
    main()
