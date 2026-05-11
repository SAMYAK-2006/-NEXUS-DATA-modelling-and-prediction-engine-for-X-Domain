"""
NEXUS — Report Generator
=========================
Produces a comprehensive, self-contained HTML report from all analysis outputs.
The report is the 'output file' — everything a researcher or practitioner needs.
"""

from __future__ import annotations

import json
import base64
import io
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def generate_report(
    series: np.ndarray,
    timestamps: Optional[np.ndarray],
    series_name: str,
    sma_result,
    analysis_result,
    ensemble_forecast,
    output_path: str = "nexus_report.html",
) -> str:
    """
    Generate full HTML report. Returns the output path.
    """
    ts_label = "t" if timestamps is None else None
    t_axis = np.arange(len(series)) if timestamps is None else np.arange(len(series))

    # ── Figure 1: Overview ──────────────────────────────────────────
    fig1 = _plot_overview(series, t_axis, sma_result, series_name, timestamps)
    f1_b64 = _fig_to_b64(fig1)

    # ── Figure 2: Geometric portrait ──────────────────────────────
    fig2 = _plot_geometric(sma_result, series_name)
    f2_b64 = _fig_to_b64(fig2)

    # ── Figure 3: Analysis dashboard ──────────────────────────────
    fig3 = _plot_analysis(series, analysis_result, sma_result)
    f3_b64 = _fig_to_b64(fig3)

    # ── Figure 4: Forecast ─────────────────────────────────────────
    fig4 = _plot_forecast(series, ensemble_forecast, sma_result, series_name)
    f4_b64 = _fig_to_b64(fig4)

    # ── Figure 5: Symmetry and spectral ───────────────────────────
    fig5 = _plot_symmetry_spectral(series, analysis_result)
    f5_b64 = _fig_to_b64(fig5)

    # ── Assemble HTML ─────────────────────────────────────────────
    html = _build_html(
        series=series,
        series_name=series_name,
        sma_result=sma_result,
        analysis_result=analysis_result,
        ensemble_forecast=ensemble_forecast,
        figs=[f1_b64, f2_b64, f3_b64, f4_b64, f5_b64],
        timestamps=timestamps,
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


# ─────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────

BG = '#0a0e1a'
ACCENT = '#00d4ff'
ACCENT2 = '#ff6b35'
ACCENT3 = '#7fff7f'
ACCENT4 = '#bf7fff'
GRID_COLOR = '#1a2040'
TEXT_COLOR = '#c8d8e8'


def _styled_fig(figsize=(16, 9)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig


def _ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color=ACCENT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def _plot_overview(series, t_axis, sma, series_name, timestamps):
    fig = _styled_fig((16, 12))
    gs = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)

    offset = sma.window
    sma_t = t_axis[offset:offset + len(sma.speed)]

    # Panel 1: Series + regime coloring
    ax1 = fig.add_subplot(gs[0])
    regime_colors = {0: '#00d4ff22', 1: '#ff6b3522', 2: '#ff000033'}
    for r_id, color in regime_colors.items():
        mask = sma.regime_labels == r_id
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        for start, end in _contiguous_ranges(idxs):
            s_idx = min(start + offset, len(series) - 1)
            e_idx = min(end + offset + 1, len(series))
            ax1.axvspan(t_axis[s_idx], t_axis[min(e_idx - 1, len(t_axis) - 1)],
                        color=color, alpha=0.6)
    ax1.plot(t_axis, series, color=ACCENT, linewidth=0.8, alpha=0.9)
    # Mark escape phases
    esc_mask = sma.escape_phases
    esc_t = sma_t[esc_mask[:len(sma_t)]]
    esc_vals = series[offset:offset + len(sma.speed)][esc_mask[:len(sma_t)]]
    ax1.scatter(esc_t, esc_vals, color=ACCENT2, s=20, zorder=5, label='Escape phase', alpha=0.8)
    _ax_style(ax1, title=f'{series_name} — Series with Regime Structure', ylabel='Value')
    ax1.legend(fontsize=7, loc='upper right', framealpha=0.3)

    # Panel 2: Speed s(t) + trust α(t)
    ax2 = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    ax2.plot(sma_t[:len(sma.speed)], sma.speed[:len(sma_t)], color=ACCENT2, linewidth=0.7, alpha=0.8, label='Speed s(t)')
    ax2b.plot(sma_t[:len(sma.trust_score)], sma.trust_score[:len(sma_t)], color=ACCENT3, linewidth=0.7, alpha=0.8, label='Trust α(t)')
    ax2b.set_ylim(0, 1)
    _ax_style(ax2, title='Manifold Speed s(t) and Data-Trust Score α(t)', ylabel='Speed')
    ax2b.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax2b.set_ylabel('α(t)', color=ACCENT3, fontsize=8)
    ax2b.yaxis.label.set_color(ACCENT3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right', framealpha=0.3)

    # Panel 3: Attractor distance D(t)
    ax3 = fig.add_subplot(gs[2])
    dist = sma.attractor_dist[:len(sma_t)]
    ax3.fill_between(sma_t[:len(dist)], 0, dist, color=ACCENT4, alpha=0.3)
    ax3.plot(sma_t[:len(dist)], dist, color=ACCENT4, linewidth=0.8)
    q90 = np.percentile(dist, 90)
    ax3.axhline(q90, color=ACCENT2, linewidth=1, linestyle='--', alpha=0.7, label=f'90th percentile')
    _ax_style(ax3, title='Attractor Distance D(t) — Stress Index', ylabel='D(t)')
    ax3.legend(fontsize=7, framealpha=0.3)

    # Panel 4: Persistence P(t, W)
    ax4 = fig.add_subplot(gs[3])
    pers = sma.persistence[:len(sma_t)]
    colors_p = [ACCENT2 if p > 0 else ACCENT for p in pers]
    ax4.bar(sma_t[:len(pers)], pers, color=colors_p, alpha=0.6, width=max(1, len(t_axis) // len(sma_t)))
    ax4.axhline(0, color=TEXT_COLOR, linewidth=0.5)
    _ax_style(ax4, title='Displacement Persistence P(t,W) — Momentum (+) vs Mean-Reversion (−)', ylabel='P(t,W)', xlabel='Time')

    return fig


def _plot_geometric(sma, series_name):
    fig = _styled_fig((16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35, figure=fig)

    pca = sma.theta_pca
    dist = sma.attractor_dist
    regimes = sma.regime_labels
    trust = sma.trust_score
    t = np.arange(len(pca))

    # Panel 1: Manifold trajectory PC1 vs PC2
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(pca[:, 0], pca[:, 1] if pca.shape[1] > 1 else np.zeros(len(pca)),
                     c=dist, cmap='plasma', s=2, alpha=0.7)
    plt.colorbar(sc, ax=ax1, label='D(t)', fraction=0.03)
    # Mark attractor
    ax1.scatter([0], [0], color=ACCENT3, s=100, marker='*', zorder=10, label='θ*')
    # Mark escape phases
    esc = sma.escape_phases[:len(pca)]
    if np.any(esc):
        ax1.scatter(pca[esc, 0], pca[esc, 1] if pca.shape[1] > 1 else np.zeros(np.sum(esc)),
                    color=ACCENT2, s=15, zorder=8, label='Escape', alpha=0.8)
    _ax_style(ax1, title='Statistical Manifold — PC1 vs PC2', xlabel='PC1', ylabel='PC2')
    ax1.legend(fontsize=6, framealpha=0.3)

    # Panel 2: Manifold trajectory over time (3D if possible)
    ax2 = fig.add_subplot(gs[0, 1])
    if pca.shape[1] >= 3:
        ax2.scatter(pca[:, 1], pca[:, 2], c=t, cmap='viridis', s=2, alpha=0.6)
    else:
        ax2.plot(t, pca[:, 0], color=ACCENT, linewidth=0.7)
    _ax_style(ax2, title='Manifold: PC2 vs PC3 (temporal gradient)', xlabel='PC2', ylabel='PC3')

    # Panel 3: Stress dimensionality sdim(t)
    ax3 = fig.add_subplot(gs[0, 2])
    sdim = sma.stress_dim[:len(t)]
    ax3.fill_between(t, 1, sdim, color=ACCENT4, alpha=0.4)
    ax3.plot(t, sdim, color=ACCENT4, linewidth=0.8)
    ax3.axhline(1, color=TEXT_COLOR, linewidth=0.5, linestyle='--')
    _ax_style(ax3, title='Stress Dimensionality sdim(t)\n(→1 = dimensionality collapse under stress)', ylabel='sdim(t)')

    # Panel 4: PCA variance explained
    ax4 = fig.add_subplot(gs[1, 0])
    vr = sma.pca_variance_ratio
    bars = ax4.bar(range(1, len(vr) + 1), vr * 100, color=ACCENT, alpha=0.8)
    ax4.plot(range(1, len(vr) + 1), np.cumsum(vr) * 100, 'o-', color=ACCENT3, markersize=4)
    _ax_style(ax4, title='PCA Variance Explained', xlabel='Component', ylabel='% Variance')

    # Panel 5: Acceleration magnitude
    ax5 = fig.add_subplot(gs[1, 1])
    accel = sma.acceleration_mag[:len(t)]
    ax5.plot(t[:len(accel)], accel, color=ACCENT2, linewidth=0.7, alpha=0.8)
    ax5.fill_between(t[:len(accel)], 0, accel, color=ACCENT2, alpha=0.2)
    _ax_style(ax5, title='Acceleration ‖a(t)‖ — Curvature of Manifold Trajectory', ylabel='‖a(t)‖')

    # Panel 6: Regime distribution
    ax6 = fig.add_subplot(gs[1, 2])
    r_labels, r_counts = np.unique(regimes, return_counts=True)
    r_names = ['Calm (R₀)', 'Intermediate (R₁)', 'Stressed (R₂)']
    colors_r = [ACCENT, ACCENT4, ACCENT2]
    bars = ax6.bar([r_names[i] for i in r_labels],
                   r_counts / len(regimes) * 100,
                   color=[colors_r[i] for i in r_labels], alpha=0.8)
    _ax_style(ax6, title='Regime Occupancy (%)', ylabel='% Time')
    ax6.tick_params(axis='x', labelrotation=15, labelsize=7)

    fig.suptitle(f'NEXUS — Geometric Portrait: {series_name}', color=ACCENT, fontsize=13, fontweight='bold', y=1.01)
    return fig


def _plot_analysis(series, analysis, sma):
    fig = _styled_fig((16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35, figure=fig)

    # Panel 1: Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(series, bins=50, density=True, color=ACCENT, alpha=0.6, label='Empirical')
    from scipy import stats as st
    x = np.linspace(min(series), max(series), 200)
    mu, sigma = np.mean(series), np.std(series, ddof=1)
    ax1.plot(x, st.norm.pdf(x, mu, sigma), color=ACCENT3, linewidth=1.5, label='Normal fit')
    if analysis.distribution_fit == 'laplace':
        loc, scale = st.laplace.fit(series)
        ax1.plot(x, st.laplace.pdf(x, loc, scale), color=ACCENT2, linewidth=1.5, label='Laplace fit')
    _ax_style(ax1, title='Distribution', xlabel='Value', ylabel='Density')
    ax1.legend(fontsize=7, framealpha=0.3)

    # Panel 2: Returns autocorrelation
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(series, lags=min(40, len(series) // 3), ax=ax2, color=ACCENT,
                 alpha=0.05, zero=False)
        ax2.lines[0].set_color(ACCENT)
    except Exception:
        lags = min(40, len(series) // 3)
        acf_vals = [np.corrcoef(series[:-l], series[l:])[0, 1] for l in range(1, lags + 1)]
        ax2.bar(range(1, lags + 1), acf_vals, color=ACCENT, alpha=0.7)
        ax2.axhline(0, color=TEXT_COLOR, linewidth=0.5)
    _ax_style(ax2, title='Autocorrelation Function', xlabel='Lag', ylabel='ACF')

    # Panel 3: Squared returns ACF (ARCH test visualization)
    ax3 = fig.add_subplot(gs[0, 2])
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    sq_returns = returns ** 2
    lags = min(30, len(sq_returns) // 3)
    sq_acf = [np.corrcoef(sq_returns[:-l], sq_returns[l:])[0, 1] if l < len(sq_returns) else 0 for l in range(1, lags + 1)]
    ax3.bar(range(1, lags + 1), sq_acf, color=ACCENT2, alpha=0.7)
    ax3.axhline(0, color=TEXT_COLOR, linewidth=0.5)
    threshold = 1.96 / np.sqrt(len(sq_returns))
    ax3.axhline(threshold, color=ACCENT3, linestyle='--', linewidth=0.8)
    ax3.axhline(-threshold, color=ACCENT3, linestyle='--', linewidth=0.8)
    arch_label = "✓ ARCH effects" if analysis.arch_effect else "✗ No ARCH"
    _ax_style(ax3, title=f'Squared-value ACF (Volatility Clustering)\n{arch_label}', xlabel='Lag')

    # Panel 4: Rolling statistics
    ax4 = fig.add_subplot(gs[1, 0])
    w = max(20, len(series) // 10)
    roll_mean = pd.Series(series).rolling(w).mean()
    roll_std = pd.Series(series).rolling(w).std()
    t = np.arange(len(series))
    ax4.plot(t, series, color=ACCENT, linewidth=0.5, alpha=0.4)
    ax4.plot(t, roll_mean, color=ACCENT3, linewidth=1.2, label=f'Rolling mean (w={w})')
    ax4.fill_between(t, roll_mean - roll_std, roll_mean + roll_std,
                     color=ACCENT3, alpha=0.15, label='±1σ')
    _ax_style(ax4, title='Rolling Mean ± Std', ylabel='Value')
    ax4.legend(fontsize=7, framealpha=0.3)

    # Panel 5: Q-Q plot
    ax5 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r) = st.probplot(series, dist='norm')
    ax5.plot(osm, osr, 'o', markersize=2, color=ACCENT, alpha=0.5)
    ax5.plot(osm, slope * np.array(osm) + intercept, color=ACCENT2, linewidth=1.5)
    _ax_style(ax5, title='Q-Q Plot vs Normal', xlabel='Theoretical quantiles', ylabel='Sample quantiles')

    # Panel 6: Hurst analysis
    ax6 = fig.add_subplot(gs[1, 2])
    n = len(series)
    lags = np.unique(np.geomspace(10, n // 2, num=15).astype(int))
    rs_vals = []
    for lag in lags:
        if lag >= n:
            continue
        sub = series[:lag]
        m = np.mean(sub)
        dev = np.cumsum(sub - m)
        r_stat = np.max(dev) - np.min(dev)
        s_stat = np.std(sub, ddof=1)
        if s_stat > 1e-10:
            rs_vals.append((lag, r_stat / s_stat))
    if rs_vals:
        log_lags = np.log([x[0] for x in rs_vals])
        log_rs = np.log([x[1] for x in rs_vals])
        ax6.scatter(log_lags, log_rs, color=ACCENT, s=20, zorder=5)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        line = np.polyval(coeffs, log_lags)
        ax6.plot(log_lags, line, color=ACCENT2, linewidth=1.5,
                 label=f'H={coeffs[0]:.3f}')
        ax6.plot(log_lags, 0.5 * np.array(log_lags) + (log_rs[0] - 0.5 * log_lags[0]),
                 color=ACCENT3, linewidth=1, linestyle='--', label='H=0.5 (random walk)')
    _ax_style(ax6, title='Hurst Exponent — R/S Analysis', xlabel='log(lag)', ylabel='log(R/S)')
    ax6.legend(fontsize=7, framealpha=0.3)

    fig.suptitle('Statistical Analysis Dashboard', color=ACCENT, fontsize=13, fontweight='bold')
    return fig


def _plot_forecast(series, ensemble, sma, series_name):
    fig = _styled_fig((16, 8))
    gs = gridspec.GridSpec(1, 2, wspace=0.3, figure=fig)

    # Panel 1: Forecast plot
    ax1 = fig.add_subplot(gs[0])
    n = len(series)
    t_hist = np.arange(n)
    horizon = ensemble.horizon
    t_fc = np.arange(n, n + horizon)

    # Historical
    lookback = min(200, n)
    ax1.plot(t_hist[-lookback:], series[-lookback:], color=ACCENT, linewidth=1.0, label='Historical')
    ax1.axvline(n - 1, color=TEXT_COLOR, linewidth=0.8, linestyle='--', alpha=0.5)

    # Individual model forecasts
    model_colors = [ACCENT3, ACCENT4, '#ff9f50', '#ff50ff']
    for i, (name, r) in enumerate(ensemble.individual_forecasts.items()):
        pt = np.asarray(r.point_forecast)[:horizon]
        ax1.plot(t_fc[:len(pt)], pt, linewidth=0.7, alpha=0.5,
                 color=model_colors[i % len(model_colors)], linestyle='--', label=name)

    # Ensemble
    pt = ensemble.point_forecast
    ax1.plot(t_fc, pt, color=ACCENT2, linewidth=2.0, label='Ensemble', zorder=10)
    ax1.fill_between(t_fc, ensemble.lower_80, ensemble.upper_80, color=ACCENT2, alpha=0.2, label='80% CI')
    ax1.fill_between(t_fc, ensemble.lower_95, ensemble.upper_95, color=ACCENT2, alpha=0.1, label='95% CI')

    _ax_style(ax1, title=f'Forecast: {series_name}\nα(t)={ensemble.current_trust:.3f}  Regime={ensemble.regime_label}',
              xlabel='Time', ylabel='Value')
    ax1.legend(fontsize=7, framealpha=0.3, loc='upper left')

    # Panel 2: Model weights
    ax2 = fig.add_subplot(gs[1])
    names = list(ensemble.model_weights.keys())
    weights = [ensemble.model_weights[n] for n in names]
    rmses = [ensemble.individual_forecasts[n].in_sample_rmse for n in names]

    bars = ax2.barh(names, [w * 100 for w in weights], color=ACCENT, alpha=0.8)
    ax2b = ax2.twiny()
    ax2b.barh(names, rmses, color=ACCENT2, alpha=0.4, label='RMSE')
    ax2b.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax2b.set_xlabel('In-sample RMSE', color=ACCENT2, fontsize=8)

    _ax_style(ax2, title='Model Ensemble Weights\n(trust-adjusted)', xlabel='Weight (%)')
    ax2.set_xlim(0, 100)

    # Add trust annotation
    alpha_val = ensemble.current_trust
    trust_text = f"Current α(t) = {alpha_val:.3f}\n"
    if alpha_val > 0.7:
        trust_text += "HIGH TRUST — data-driven models weighted up"
    elif alpha_val > 0.4:
        trust_text += "MODERATE TRUST — balanced weighting"
    else:
        trust_text += "LOW TRUST — structural priors dominant"
    ax2.text(0.02, 0.02, trust_text, transform=ax2.transAxes,
             fontsize=7, color=ACCENT3, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0e2a', edgecolor=ACCENT3, alpha=0.8))

    fig.suptitle('Trust-Weighted Ensemble Forecast', color=ACCENT, fontsize=13, fontweight='bold')
    return fig


def _plot_symmetry_spectral(series, analysis):
    fig = _styled_fig((16, 8))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35, figure=fig)

    from scipy import signal as sig

    # Panel 1: Power spectral density
    ax1 = fig.add_subplot(gs[0, 0])
    freqs, psd = sig.periodogram(series - np.mean(series))
    ax1.semilogy(freqs[1:], psd[1:], color=ACCENT, linewidth=0.7)
    if analysis.dominant_freq > 0:
        ax1.axvline(analysis.dominant_freq, color=ACCENT2, linewidth=1.5,
                    linestyle='--', label=f'Dominant f={analysis.dominant_freq:.4f}')
    _ax_style(ax1, title='Power Spectral Density', xlabel='Frequency', ylabel='PSD (log)')
    ax1.legend(fontsize=7, framealpha=0.3)

    # Panel 2: Spectrogram
    ax2 = fig.add_subplot(gs[0, 1])
    n = len(series)
    nperseg = min(128, n // 4)
    if nperseg >= 8:
        f_spec, t_spec, Sxx = sig.spectrogram(series, nperseg=nperseg)
        ax2.pcolormesh(t_spec, f_spec, np.log1p(Sxx), cmap='inferno', shading='gouraud')
    _ax_style(ax2, title='Spectrogram (time-frequency)', xlabel='Time', ylabel='Frequency')

    # Panel 3: Phase portrait (x_t vs x_{t+1})
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(series[:-1], series[1:], c=np.arange(len(series) - 1),
                cmap='plasma', s=2, alpha=0.5)
    _ax_style(ax3, title='Phase Portrait x(t) vs x(t+1)\n(attractor geometry)', xlabel='x(t)', ylabel='x(t+1)')

    # Panel 4: Return map (nonlinearity)
    ax4 = fig.add_subplot(gs[1, 0])
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    ax4.scatter(returns[:-1], returns[1:], c=np.arange(len(returns) - 1),
                cmap='viridis', s=2, alpha=0.4)
    _ax_style(ax4, title='Return Map r(t) vs r(t+1)', xlabel='r(t)', ylabel='r(t+1)')

    # Panel 5: Time reversibility test
    ax5 = fig.add_subplot(gs[1, 1])
    h_vals = [1, 5, 10, 20, 50]
    rev_stats = []
    for h in h_vals:
        if h < len(series):
            diffs = series[h:] - series[:-h]
            rev = abs(np.mean(diffs ** 3)) / (np.std(series) ** 3 + 1e-10)
            rev_stats.append(rev)
        else:
            rev_stats.append(0.0)
    ax5.plot(h_vals[:len(rev_stats)], rev_stats, 'o-', color=ACCENT, markersize=6)
    ax5.axhline(0, color=TEXT_COLOR, linewidth=0.5, linestyle='--')
    nonlinear = "Nonlinear (irreversible)" if analysis.time_reversibility > 0.3 else "Possibly linear (reversible)"
    _ax_style(ax5, title=f'Time Reversibility\n({nonlinear})', xlabel='Lag h', ylabel='Asymmetry stat')

    # Panel 6: Anomaly/tail events
    ax6 = fig.add_subplot(gs[1, 2])
    rolling_z = (series - pd.Series(series).rolling(20, min_periods=5).mean()) / \
                 (pd.Series(series).rolling(20, min_periods=5).std() + 1e-10)
    t = np.arange(len(series))
    ax6.plot(t, rolling_z, color=ACCENT, linewidth=0.7, alpha=0.8)
    ax6.axhline(2, color=ACCENT2, linewidth=1, linestyle='--', label='+2σ')
    ax6.axhline(-2, color=ACCENT2, linewidth=1, linestyle='--', label='-2σ')
    extreme_up = rolling_z > 2
    extreme_dn = rolling_z < -2
    ax6.scatter(t[extreme_up], rolling_z.values[extreme_up], color=ACCENT2, s=15, zorder=5)
    ax6.scatter(t[extreme_dn], rolling_z.values[extreme_dn], color=ACCENT4, s=15, zorder=5)
    _ax_style(ax6, title='Rolling Z-Score (Anomaly Detection)', ylabel='Z-score')
    ax6.legend(fontsize=7, framealpha=0.3)

    fig.suptitle('Symmetry, Spectral Structure, and Nonlinearity', color=ACCENT, fontsize=13, fontweight='bold')
    return fig


# ─────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────

def _build_html(series, series_name, sma_result, analysis_result, ensemble_forecast, figs, timestamps):
    sma = sma_result
    ar = analysis_result
    ef = ensemble_forecast
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt(v, d=4):
        try:
            return f"{float(v):.{d}f}"
        except Exception:
            return str(v)

    regime_names = {0: 'Calm', 1: 'Intermediate', 2: 'Stressed'}
    current_regime = int(sma.regime_labels[-1]) if len(sma.regime_labels) > 0 else 0
    current_trust = float(sma.trust_score[-1]) if len(sma.trust_score) > 0 else 0.5

    # Insights HTML
    insight_items = ''.join(f'<li>{ins}</li>' for ins in ar.key_insights)

    # Applicability
    app = sma.applicability
    app_color = '#7fff7f' if app.applicable else '#ff6b35'
    att_color = '#7fff7f' if app.attractor_exists else '#ff6b35'

    # Model table
    model_rows = ''
    for name, r in ef.individual_forecasts.items():
        w = ef.model_weights.get(name, 0.0)
        model_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{fmt(r.in_sample_rmse)}</td>
          <td>{fmt(r.in_sample_mae)}</td>
          <td>{fmt(r.in_sample_mape, 2)}%</td>
          <td>{fmt(w * 100, 1)}%</td>
          <td>{fmt(r.point_forecast[0]) if len(r.point_forecast) > 0 else 'N/A'}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS Report — {series_name}</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --bg2: #0f1525;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --accent3: #7fff7f;
    --accent4: #bf7fff;
    --text: #c8d8e8;
    --border: #1a2040;
    --card: #111827;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #0f1525, #1a2040); padding: 32px 40px; border-bottom: 2px solid var(--accent); }}
  .header h1 {{ font-size: 28px; color: var(--accent); letter-spacing: 4px; font-weight: bold; }}
  .header .subtitle {{ color: var(--text); font-size: 14px; margin-top: 8px; opacity: 0.7; }}
  .header .meta {{ margin-top: 16px; display: flex; gap: 32px; }}
  .meta-item {{ display: flex; flex-direction: column; }}
  .meta-item .label {{ font-size: 10px; color: var(--accent); text-transform: uppercase; letter-spacing: 2px; }}
  .meta-item .value {{ font-size: 16px; color: var(--text); }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 40px; }}
  .section {{ margin: 40px 0; }}
  .section-title {{ font-size: 18px; color: var(--accent); border-left: 3px solid var(--accent); padding-left: 12px; margin-bottom: 20px; letter-spacing: 2px; text-transform: uppercase; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  .grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .card h3 {{ color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }}
  .stat-row {{ display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding: 6px 0; }}
  .stat-label {{ color: var(--text); opacity: 0.7; }}
  .stat-value {{ color: var(--text); font-weight: bold; }}
  .stat-value.green {{ color: var(--accent3); }}
  .stat-value.red {{ color: var(--accent2); }}
  .stat-value.blue {{ color: var(--accent); }}
  .stat-value.purple {{ color: var(--accent4); }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
  .badge-green {{ background: #1a3a1a; color: var(--accent3); border: 1px solid var(--accent3); }}
  .badge-red {{ background: #3a1a1a; color: var(--accent2); border: 1px solid var(--accent2); }}
  .badge-blue {{ background: #1a2a3a; color: var(--accent); border: 1px solid var(--accent); }}
  .badge-purple {{ background: #2a1a3a; color: var(--accent4); border: 1px solid var(--accent4); }}
  .insights {{ background: var(--card); border: 1px solid var(--accent4); border-radius: 8px; padding: 20px; }}
  .insights h3 {{ color: var(--accent4); margin-bottom: 12px; font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }}
  .insights li {{ margin: 8px 0; color: var(--text); padding-left: 8px; border-left: 2px solid var(--accent4); list-style: none; }}
  .plot-container {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 16px 0; }}
  .plot-container img {{ width: 100%; height: auto; border-radius: 4px; }}
  .plot-title {{ color: var(--accent); font-size: 12px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: var(--border); color: var(--accent); padding: 8px 12px; text-align: left; font-size: 11px; letter-spacing: 1px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: #1a2040; }}
  .forecast-banner {{ background: linear-gradient(135deg, #0f1525, #1a2040); border: 1px solid var(--accent2); border-radius: 8px; padding: 24px; margin: 16px 0; }}
  .fc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; margin-top: 16px; }}
  .fc-cell {{ text-align: center; }}
  .fc-cell .step {{ font-size: 10px; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; }}
  .fc-cell .val {{ font-size: 20px; color: var(--text); font-weight: bold; }}
  .fc-cell .ci {{ font-size: 10px; color: var(--text); opacity: 0.6; }}
  .app-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .app-item {{ background: var(--card); border-radius: 6px; padding: 12px; border: 1px solid var(--border); display: flex; align-items: center; gap: 10px; }}
  .app-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .footer {{ text-align: center; padding: 32px; color: var(--text); opacity: 0.4; border-top: 1px solid var(--border); margin-top: 60px; font-size: 11px; }}
</style>
</head>
<body>

<div class="header">
  <h1>⬡ NEXUS</h1>
  <div class="subtitle">Nonlinear Exploratory framework for X-domain Unified Systems</div>
  <div class="meta">
    <div class="meta-item">
      <span class="label">Series</span>
      <span class="value">{series_name}</span>
    </div>
    <div class="meta-item">
      <span class="label">Observations</span>
      <span class="value">{len(series):,}</span>
    </div>
    <div class="meta-item">
      <span class="label">Current Regime</span>
      <span class="value" style="color: {'#7fff7f' if current_regime == 0 else '#ff6b35' if current_regime == 2 else '#bf7fff'}">{regime_names.get(current_regime, '?')}</span>
    </div>
    <div class="meta-item">
      <span class="label">Trust Score α(t)</span>
      <span class="value" style="color: {'#7fff7f' if current_trust > 0.7 else '#ff6b35' if current_trust < 0.4 else '#bf7fff'}">{current_trust:.4f}</span>
    </div>
    <div class="meta-item">
      <span class="label">Generated</span>
      <span class="value">{now}</span>
    </div>
  </div>
</div>

<div class="container">

<!-- APPLICABILITY -->
<div class="section">
  <div class="section-title">▸ SMA Framework Applicability</div>
  <div class="app-grid">
    <div class="app-item">
      <div class="app-dot" style="background: {app_color}"></div>
      <div><div style="font-size:10px;color:var(--accent)">FRAMEWORK APPLICABLE</div>
      <div class="badge {'badge-green' if app.applicable else 'badge-red'}">{str(app.applicable).upper()}</div></div>
    </div>
    <div class="app-item">
      <div class="app-dot" style="background: {att_color}"></div>
      <div><div style="font-size:10px;color:var(--accent)">ATTRACTOR EXISTS</div>
      <div class="badge {'badge-green' if app.attractor_exists else 'badge-red'}">{str(app.attractor_exists).upper()}</div></div>
    </div>
    <div class="app-item">
      <div class="app-dot" style="background: {'#7fff7f' if app.mean_reverting else '#ff6b35'}"></div>
      <div><div style="font-size:10px;color:var(--accent)">MEAN REVERTING</div>
      <div class="badge {'badge-green' if app.mean_reverting else 'badge-red'}">{str(app.mean_reverting).upper()}</div></div>
    </div>
    <div class="app-item">
      <div class="app-dot" style="background: {'#7fff7f' if app.bounded else '#ff6b35'}"></div>
      <div><div style="font-size:10px;color:var(--accent)">BOUNDED</div>
      <div class="badge {'badge-green' if app.bounded else 'badge-red'}">{str(app.bounded).upper()}</div></div>
    </div>
    <div class="app-item">
      <div class="app-dot" style="background: #00d4ff"></div>
      <div><div style="font-size:10px;color:var(--accent)">E[P(t,W)]</div>
      <div style="color:{'#7fff7f' if app.persistence_score < 0 else '#ff6b35'};font-size:16px;font-weight:bold">{fmt(app.persistence_score)} (p={fmt(app.persistence_pvalue, 4)})</div></div>
    </div>
    <div class="app-item">
      <div class="app-dot" style="background: #bf7fff"></div>
      <div><div style="font-size:10px;color:var(--accent)">STATIONARITY</div>
      <div style="color:var(--text);font-size:14px">{app.stationarity_verdict.upper()} (ADF p={fmt(app.adf_pvalue, 4)})</div></div>
    </div>
  </div>
  {"".join(f'<p style="color:#ff6b35;margin-top:8px;font-size:11px">⚠ {n}</p>' for n in app.notes)}
</div>

<!-- SERIES OVERVIEW -->
<div class="section">
  <div class="section-title">▸ Series Overview</div>
  <div class="plot-container">
    <div class="plot-title">Time Series, Regime Structure, Speed, Trust Score, Attractor Distance, Persistence</div>
    <img src="data:image/png;base64,{figs[0]}" alt="Overview">
  </div>
</div>

<!-- GEOMETRIC PORTRAIT -->
<div class="section">
  <div class="section-title">▸ Geometric Portrait — Statistical Manifold</div>
  <div class="plot-container">
    <div class="plot-title">Manifold Trajectory, PCA, Stress Dimensionality, Regime Occupancy</div>
    <img src="data:image/png;base64,{figs[1]}" alt="Geometric">
  </div>
  <div class="grid-3" style="margin-top:16px">
    <div class="card">
      <h3>Geometric Quantities</h3>
      <div class="stat-row"><span class="stat-label">Mean speed E[s(t)]</span><span class="stat-value blue">{fmt(np.mean(sma.speed))}</span></div>
      <div class="stat-row"><span class="stat-label">Max speed</span><span class="stat-value red">{fmt(np.max(sma.speed))}</span></div>
      <div class="stat-row"><span class="stat-label">Mean D(t)</span><span class="stat-value purple">{fmt(np.mean(sma.attractor_dist))}</span></div>
      <div class="stat-row"><span class="stat-label">Max D(t)</span><span class="stat-value red">{fmt(np.max(sma.attractor_dist))}</span></div>
      <div class="stat-row"><span class="stat-label">E[P(t,W)]</span><span class="stat-value {'green' if app.persistence_score < 0 else 'red'}">{fmt(app.persistence_score)}</span></div>
      <div class="stat-row"><span class="stat-label">Escape phases</span><span class="stat-value red">{int(np.sum(sma.escape_phases))} timesteps</span></div>
    </div>
    <div class="card">
      <h3>Regime Distribution</h3>
      {''.join(f'<div class="stat-row"><span class="stat-label">{regime_names[r_id]}</span><span class="stat-value">{int(np.sum(sma.regime_labels == r_id))} ({int(np.mean(sma.regime_labels == r_id)*100)}%)</span></div>' for r_id in range(3))}
      <div class="stat-row"><span class="stat-label">Current regime</span><span class="stat-value {'green' if current_regime == 0 else 'red' if current_regime == 2 else 'purple'}">{regime_names.get(current_regime)}</span></div>
      <div class="stat-row"><span class="stat-label">Current trust α(t)</span><span class="stat-value {'green' if current_trust > 0.7 else 'red' if current_trust < 0.4 else 'purple'}">{fmt(current_trust)}</span></div>
    </div>
    <div class="card">
      <h3>PCA Structure</h3>
      {''.join(f'<div class="stat-row"><span class="stat-label">PC{i+1}</span><span class="stat-value blue">{fmt(v*100, 1)}% variance</span></div>' for i, v in enumerate(sma.pca_variance_ratio))}
      <div class="stat-row"><span class="stat-label">Cumulative (all PCs)</span><span class="stat-value green">{fmt(np.sum(sma.pca_variance_ratio)*100, 1)}%</span></div>
      <div class="stat-row"><span class="stat-label">Mean sdim(t)</span><span class="stat-value purple">{fmt(np.mean(sma.stress_dim))}</span></div>
    </div>
  </div>
</div>

<!-- STATISTICAL ANALYSIS -->
<div class="section">
  <div class="section-title">▸ Statistical Analysis</div>
  <div class="plot-container">
    <img src="data:image/png;base64,{figs[2]}" alt="Analysis">
  </div>
  <div class="grid-4" style="margin-top:16px">
    <div class="card">
      <h3>Descriptive</h3>
      <div class="stat-row"><span class="stat-label">N</span><span class="stat-value">{ar.n:,}</span></div>
      <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{fmt(ar.mean)}</span></div>
      <div class="stat-row"><span class="stat-label">Std</span><span class="stat-value">{fmt(ar.std)}</span></div>
      <div class="stat-row"><span class="stat-label">Skewness</span><span class="stat-value {'red' if abs(ar.skewness) > 1 else 'green'}">{fmt(ar.skewness)}</span></div>
      <div class="stat-row"><span class="stat-label">Kurtosis</span><span class="stat-value {'red' if ar.kurtosis > 3 else 'green'}">{fmt(ar.kurtosis)}</span></div>
      <div class="stat-row"><span class="stat-label">IQR</span><span class="stat-value">{fmt(ar.iqr)}</span></div>
    </div>
    <div class="card">
      <h3>Dynamics</h3>
      <div class="stat-row"><span class="stat-label">Hurst H</span><span class="stat-value {'green' if ar.hurst > 0.55 else 'red' if ar.hurst < 0.45 else 'blue'}">{fmt(ar.hurst)}</span></div>
      <div class="stat-row"><span class="stat-label">Lyapunov (approx)</span><span class="stat-value {'red' if ar.lyapunov_approx > 0.1 else 'green'}">{fmt(ar.lyapunov_approx)}</span></div>
      <div class="stat-row"><span class="stat-label">Sample entropy</span><span class="stat-value">{fmt(ar.sample_entropy)}</span></div>
      <div class="stat-row"><span class="stat-label">Time reversibility</span><span class="stat-value {'red' if ar.time_reversibility > 0.3 else 'green'}">{fmt(ar.time_reversibility)}</span></div>
      <div class="stat-row"><span class="stat-label">BDS stat</span><span class="stat-value">{fmt(ar.bds_statistic)}</span></div>
      <div class="stat-row"><span class="stat-label">Spectral type</span><span class="stat-value blue">{ar.spectral_complexity}</span></div>
    </div>
    <div class="card">
      <h3>Risk</h3>
      <div class="stat-row"><span class="stat-label">Realized vol</span><span class="stat-value">{fmt(ar.realized_vol)}</span></div>
      <div class="stat-row"><span class="stat-label">Vol-of-vol</span><span class="stat-value {'red' if ar.vol_of_vol > ar.realized_vol * 0.5 else 'green'}">{fmt(ar.vol_of_vol)}</span></div>
      <div class="stat-row"><span class="stat-label">Max drawdown</span><span class="stat-value red">{fmt(ar.max_drawdown * 100, 1)}%</span></div>
      <div class="stat-row"><span class="stat-label">VaR 95%</span><span class="stat-value red">{fmt(ar.var_95)}</span></div>
      <div class="stat-row"><span class="stat-label">CVaR 95%</span><span class="stat-value red">{fmt(ar.cvar_95)}</span></div>
      <div class="stat-row"><span class="stat-label">ARCH effects</span><span class="stat-value {'red' if ar.arch_effect else 'green'}">{'YES' if ar.arch_effect else 'NO'} (p={fmt(ar.arch_pvalue, 4)})</span></div>
    </div>
    <div class="card">
      <h3>Structure</h3>
      <div class="stat-row"><span class="stat-label">Stationarity</span><span class="stat-value {'green' if ar.adf_stationary else 'red'}">{'STATIONARY' if ar.adf_stationary else 'NON-STATIONARY'}</span></div>
      <div class="stat-row"><span class="stat-label">ADF p-value</span><span class="stat-value">{fmt(ar.adf_pvalue)}</span></div>
      <div class="stat-row"><span class="stat-label">Trend</span><span class="stat-value {'red' if ar.trend_direction != 'flat' else 'green'}">{ar.trend_direction.upper()}</span></div>
      <div class="stat-row"><span class="stat-label">Trend R²</span><span class="stat-value">{fmt(ar.trend_r2)}</span></div>
      <div class="stat-row"><span class="stat-label">Changepoints</span><span class="stat-value {'red' if ar.n_changepoints > 0 else 'green'}">{ar.n_changepoints}</span></div>
      <div class="stat-row"><span class="stat-label">Best fit dist.</span><span class="stat-value blue">{ar.distribution_fit}</span></div>
    </div>
  </div>
</div>

<!-- KEY INSIGHTS -->
<div class="section">
  <div class="section-title">▸ Key Insights</div>
  <div class="insights">
    <h3>Automated Interpretations</h3>
    <ul>{insight_items}</ul>
  </div>
</div>

<!-- SYMMETRY & SPECTRAL -->
<div class="section">
  <div class="section-title">▸ Symmetry, Spectral Structure & Nonlinearity</div>
  <div class="plot-container">
    <img src="data:image/png;base64,{figs[4]}" alt="Symmetry">
  </div>
</div>

<!-- FORECAST -->
<div class="section">
  <div class="section-title">▸ Trust-Weighted Ensemble Forecast</div>
  <div class="forecast-banner">
    <div style="color:var(--accent);font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
      Next {ef.horizon} steps — α(t)={current_trust:.4f} — Regime: {regime_names.get(current_regime)}
    </div>
    <div class="fc-grid">
      {chr(10).join(f'<div class="fc-cell"><div class="step">t+{i+1}</div><div class="val">{fmt(v, 3)}</div><div class="ci">[{fmt(l, 3)}, {fmt(u, 3)}]</div></div>' for i, (v, l, u) in enumerate(zip(ef.point_forecast, ef.lower_95, ef.upper_95)))}
    </div>
  </div>
  <div class="plot-container">
    <img src="data:image/png;base64,{figs[3]}" alt="Forecast">
  </div>
  <div style="margin-top:16px">
    <h3 style="color:var(--accent);font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Model Performance Table</h3>
    <table>
      <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Ensemble Weight</th><th>Next-step forecast</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table>
  </div>
</div>

<!-- FOOTER -->
<div class="footer">
  NEXUS — Built on Statistical Manifold Attractor Framework (Jain, 2026)<br>
  "Any bounded dynamical system has a statistical attractor. Regime stress is departure from that attractor."<br><br>
  Generated: {now}
</div>

</div>
</body>
</html>"""
    return html


def _contiguous_ranges(indices):
    """Convert array of indices to list of (start, end) contiguous ranges."""
    if len(indices) == 0:
        return []
    ranges = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = i
            prev = i
    ranges.append((start, prev))
    return ranges
