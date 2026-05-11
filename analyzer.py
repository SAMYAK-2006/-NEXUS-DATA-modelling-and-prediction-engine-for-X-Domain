"""
NEXUS — Analysis Module
========================
Full statistical, spectral, nonlinear, and symmetry analysis of time series.
Runs after the geometric engine and produces interpretable insights.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.stats import entropy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisResult:
    """Complete analysis of a time series."""

    # ── Basic statistics ──────────────────────────────────────────
    n: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    range_val: float
    iqr: float
    cv: float                   # coefficient of variation

    # ── Stationarity & memory ─────────────────────────────────────
    adf_pvalue: float
    adf_stationary: bool
    hurst: float                # 0.5=random, >0.5=persistent, <0.5=mean-reverting
    hurst_interpretation: str
    lyapunov_approx: float      # positive → chaotic tendency
    sample_entropy: float       # regularity measure

    # ── Autocorrelation structure ──────────────────────────────────
    acf_significant_lags: list[int]
    pacf_significant_lags: list[int]
    dominant_period: Optional[float]  # dominant cycle length
    seasonality_strength: float

    # ── Volatility & risk ─────────────────────────────────────────
    realized_vol: float
    vol_of_vol: float           # volatility clustering measure
    max_drawdown: float
    calmar_ratio: float
    sharpe_approx: float
    tail_ratio: float           # 95th / 5th percentile of abs returns
    var_95: float               # Value at Risk 95%
    cvar_95: float              # Conditional VaR

    # ── Distribution shape ────────────────────────────────────────
    is_normal: bool
    normality_pvalue: float
    tail_heaviness: str         # 'thin', 'normal', 'fat'
    distribution_fit: str       # best-fit distribution name

    # ── Symmetry & nonlinearity ───────────────────────────────────
    time_reversibility: float   # 0=reversible, >0=irreversible (nonlinear)
    bds_statistic: float        # nonlinearity test
    arch_effect: bool           # volatility clustering
    arch_pvalue: float
    teraesvirta_stat: float     # neural net nonlinearity test proxy

    # ── Spectral ──────────────────────────────────────────────────
    dominant_freq: float
    spectral_entropy: float     # 0=single freq, 1=pure noise
    spectral_complexity: str    # 'periodic', 'chaotic', 'noisy', 'mixed'

    # ── Change points ─────────────────────────────────────────────
    n_changepoints: int
    changepoint_indices: list[int]
    changepoint_method: str

    # ── Trend ─────────────────────────────────────────────────────
    trend_slope: float
    trend_pvalue: float
    trend_r2: float
    trend_direction: str        # 'upward', 'downward', 'flat'

    # ── Summary ───────────────────────────────────────────────────
    key_insights: list[str] = field(default_factory=list)


class TimeSeriesAnalyzer:
    """
    Comprehensive analysis engine for any time series.
    Applies every relevant statistical, spectral, and nonlinear test.
    """

    def __init__(self, freq: Optional[str] = None, returns_mode: bool = False):
        """
        freq: pandas frequency string ('D', 'H', 'M', etc.) for period interpretation
        returns_mode: if True, treat series as returns (not levels)
        """
        self.freq = freq
        self.returns_mode = returns_mode

    def analyze(self, series: np.ndarray) -> AnalysisResult:
        series = np.asarray(series, dtype=float)
        series = series[np.isfinite(series)]
        n = len(series)

        # Work on returns or levels
        if self.returns_mode:
            returns = series
        else:
            returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
            returns = returns[np.isfinite(returns)]

        # ── Basic statistics ──────────────────────────────────
        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1))
        skewness = float(stats.skew(series))
        kurtosis_val = float(stats.kurtosis(series))
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        range_val = max_val - min_val
        iqr = float(np.percentile(series, 75) - np.percentile(series, 25))
        cv = std / (abs(mean) + 1e-10)

        # ── Stationarity ──────────────────────────────────────
        adf_pval, adf_stat = self._adf_test(series)
        hurst, hurst_interp = self._hurst_exponent(series)
        lyap = self._lyapunov_approx(series)
        samp_ent = self._sample_entropy(series)

        # ── Autocorrelation ───────────────────────────────────
        acf_lags, pacf_lags = self._significant_lags(series)
        dom_period = self._dominant_period(series)
        seas_strength = self._seasonality_strength(series, dom_period)

        # ── Volatility & risk ─────────────────────────────────
        r_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        vol_series = pd.Series(returns).rolling(max(5, len(returns)//10)).std().dropna().values
        volvol = float(np.std(vol_series)) if len(vol_series) > 1 else 0.0
        mdd = self._max_drawdown(series)
        calmar = (mean / (mdd + 1e-10)) if mdd > 0 else 0.0
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0.0
        tail_ratio = self._tail_ratio(returns)
        var95 = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
        cvar95 = float(np.mean(returns[returns <= var95])) if len(returns[returns <= var95]) > 0 else var95

        # ── Distribution ──────────────────────────────────────
        is_normal, norm_pval = self._normality_test(series)
        tail_heavy = self._tail_heaviness(kurtosis_val)
        dist_fit = self._best_distribution(series)

        # ── Symmetry & nonlinearity ───────────────────────────
        time_rev = self._time_reversibility(series)
        bds_stat = self._bds_proxy(returns)
        arch_effect, arch_pval = self._arch_test(returns)
        tera_stat = self._teraesvirta_proxy(series)

        # ── Spectral ──────────────────────────────────────────
        dom_freq, spec_ent, spec_complex = self._spectral_analysis(series)

        # ── Change points ─────────────────────────────────────
        cps, cp_method = self._detect_changepoints(series)

        # ── Trend ─────────────────────────────────────────────
        slope, trend_pval, r2, trend_dir = self._trend_analysis(series)

        # ── Key insights ──────────────────────────────────────
        insights = self._generate_insights(
            hurst=hurst, adf_pval=adf_pval, skewness=skewness,
            kurtosis_val=kurtosis_val, arch_effect=arch_effect,
            time_rev=time_rev, spec_complex=spec_complex,
            n_cp=len(cps), mdd=mdd, lyap=lyap,
            seas_strength=seas_strength, trend_dir=trend_dir
        )

        return AnalysisResult(
            n=n, mean=mean, std=std, skewness=skewness, kurtosis=kurtosis_val,
            min_val=min_val, max_val=max_val, range_val=range_val, iqr=iqr, cv=cv,
            adf_pvalue=adf_pval, adf_stationary=(adf_pval < 0.05),
            hurst=hurst, hurst_interpretation=hurst_interp,
            lyapunov_approx=lyap, sample_entropy=samp_ent,
            acf_significant_lags=acf_lags, pacf_significant_lags=pacf_lags,
            dominant_period=dom_period, seasonality_strength=seas_strength,
            realized_vol=r_vol, vol_of_vol=volvol, max_drawdown=mdd,
            calmar_ratio=calmar, sharpe_approx=sharpe, tail_ratio=tail_ratio,
            var_95=var95, cvar_95=cvar95,
            is_normal=is_normal, normality_pvalue=norm_pval,
            tail_heaviness=tail_heavy, distribution_fit=dist_fit,
            time_reversibility=time_rev, bds_statistic=bds_stat,
            arch_effect=arch_effect, arch_pvalue=arch_pval,
            teraesvirta_stat=tera_stat,
            dominant_freq=dom_freq, spectral_entropy=spec_ent,
            spectral_complexity=spec_complex,
            n_changepoints=len(cps), changepoint_indices=cps,
            changepoint_method=cp_method,
            trend_slope=slope, trend_pvalue=trend_pval,
            trend_r2=r2, trend_direction=trend_dir,
            key_insights=insights,
        )

    # ─── Statistical helpers ────────────────────────────────────────

    def _adf_test(self, series):
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, autolag='AIC')
            return float(result[1]), float(result[0])
        except Exception:
            return 1.0, 0.0

    def _hurst_exponent(self, series):
        """R/S analysis for Hurst exponent."""
        n = len(series)
        if n < 20:
            return 0.5, "insufficient data"
        try:
            lags = np.unique(np.geomspace(10, n // 2, num=15).astype(int))
            rs_vals = []
            for lag in lags:
                sub = series[:lag]
                m = np.mean(sub)
                dev = np.cumsum(sub - m)
                r = np.max(dev) - np.min(dev)
                s = np.std(sub, ddof=1)
                if s > 1e-10:
                    rs_vals.append((lag, r / s))
            if len(rs_vals) < 2:
                return 0.5, "insufficient"
            log_lags = np.log([x[0] for x in rs_vals])
            log_rs = np.log([x[1] for x in rs_vals])
            h = np.polyfit(log_lags, log_rs, 1)[0]
            h = float(np.clip(h, 0, 1))
            if h > 0.55:
                interp = f"persistent/trending (H={h:.3f}) — momentum present"
            elif h < 0.45:
                interp = f"mean-reverting (H={h:.3f}) — oscillatory dynamics"
            else:
                interp = f"random walk (H={h:.3f}) — no memory structure"
            return h, interp
        except Exception:
            return 0.5, "computation failed"

    def _lyapunov_approx(self, series):
        """Approximate largest Lyapunov exponent via divergence of nearby trajectories."""
        try:
            n = len(series)
            if n < 50:
                return 0.0
            dim = 3
            lag = 1
            # Embed
            N_emb = n - (dim - 1) * lag
            embedded = np.array([series[i:i + N_emb] for i in range(0, dim * lag, lag)]).T
            # Find nearest neighbours and track divergence
            divergences = []
            sample = min(100, N_emb // 2)
            indices = np.random.choice(N_emb // 2, size=sample, replace=False)
            for i in indices:
                dists = np.linalg.norm(embedded - embedded[i], axis=1)
                dists[max(0, i-5):i+5] = np.inf  # exclude temporal neighbours
                j = np.argmin(dists)
                if dists[j] < 1e-10:
                    continue
                steps = min(10, N_emb - max(i, j) - 1)
                if steps < 1:
                    continue
                future_dist = np.linalg.norm(embedded[i + steps] - embedded[j + steps])
                if future_dist > 1e-10 and dists[j] > 1e-10:
                    divergences.append(np.log(future_dist / dists[j]) / steps)
            return float(np.mean(divergences)) if divergences else 0.0
        except Exception:
            return 0.0

    def _sample_entropy(self, series, m=2, r_factor=0.2):
        """Sample entropy: regularity measure. Lower = more regular."""
        try:
            n = len(series)
            if n < 30:
                return 0.0
            r = r_factor * np.std(series, ddof=1)
            series_norm = (series - np.mean(series)) / (np.std(series) + 1e-10)
            count_m = 0
            count_m1 = 0
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    if np.max(np.abs(series_norm[i:i+m] - series_norm[j:j+m])) < r:
                        count_m += 1
                        if np.abs(series_norm[i+m] - series_norm[j+m]) < r:
                            count_m1 += 1
            if count_m == 0 or count_m1 == 0:
                return 0.0
            return float(-np.log(count_m1 / count_m))
        except Exception:
            return 0.0

    def _significant_lags(self, series, max_lag=40, alpha=0.05):
        """Find significant ACF and PACF lags."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            n = len(series)
            max_lag = min(max_lag, n // 3)
            threshold = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
            acf_vals = acf(series, nlags=max_lag, fft=True)[1:]
            pacf_vals = pacf(series, nlags=max_lag, method='ols')[1:]
            acf_lags = [i + 1 for i, v in enumerate(acf_vals) if abs(v) > threshold]
            pacf_lags = [i + 1 for i, v in enumerate(pacf_vals) if abs(v) > threshold]
            return acf_lags[:10], pacf_lags[:10]
        except Exception:
            return [], []

    def _dominant_period(self, series):
        """Find dominant period via FFT."""
        try:
            n = len(series)
            if n < 10:
                return None
            fft_vals = np.abs(np.fft.rfft(series - np.mean(series)))
            freqs = np.fft.rfftfreq(n)
            if len(freqs) < 2:
                return None
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            dom_freq = freqs[dominant_idx]
            if dom_freq > 1e-10:
                return float(1.0 / dom_freq)
            return None
        except Exception:
            return None

    def _seasonality_strength(self, series, period):
        """STL-based seasonality strength [0, 1]."""
        if period is None or period < 2 or period > len(series) // 2:
            return 0.0
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            period_int = max(2, int(round(period)))
            if len(series) < 2 * period_int:
                return 0.0
            result = seasonal_decompose(series, period=period_int, model='additive', extrapolate_trend='freq')
            seasonal_var = np.var(result.seasonal)
            residual_var = np.var(result.resid[~np.isnan(result.resid)])
            total = seasonal_var + residual_var
            return float(seasonal_var / total) if total > 0 else 0.0
        except Exception:
            return 0.0

    def _max_drawdown(self, series):
        if len(series) < 2:
            return 0.0
        cummax = np.maximum.accumulate(series)
        dd = (cummax - series) / (np.abs(cummax) + 1e-10)
        return float(np.max(dd))

    def _tail_ratio(self, returns):
        if len(returns) < 10:
            return 1.0
        p95 = np.percentile(np.abs(returns), 95)
        p5 = np.percentile(np.abs(returns), 5)
        return float(p95 / (p5 + 1e-10))

    def _normality_test(self, series):
        try:
            if len(series) < 8:
                return True, 1.0
            if len(series) < 50:
                stat, pval = stats.shapiro(series[:50])
            else:
                stat, pval = stats.normaltest(series)
            return pval > 0.05, float(pval)
        except Exception:
            return True, 1.0

    def _tail_heaviness(self, kurtosis_val):
        if kurtosis_val > 3:
            return 'fat'
        elif kurtosis_val < -1:
            return 'thin'
        return 'normal'

    def _best_distribution(self, series):
        candidates = {
            'normal': stats.norm,
            'laplace': stats.laplace,
            't(5)': stats.t,
        }
        best, best_aic = 'normal', np.inf
        for name, dist in candidates.items():
            try:
                if name == 't(5)':
                    params = dist.fit(series, f0=5)
                else:
                    params = dist.fit(series)
                log_lik = np.sum(dist.logpdf(series, *params))
                k = len(params)
                aic = 2 * k - 2 * log_lik
                if aic < best_aic:
                    best_aic = aic
                    best = name
            except Exception:
                pass
        return best

    def _time_reversibility(self, series):
        """
        Time reversibility statistic.
        If E[(x_t+h - x_t)^3] ≈ 0, the series is reversible (linear).
        Nonzero → irreversible → nonlinear dynamics.
        """
        try:
            h = max(1, len(series) // 10)
            diffs = series[h:] - series[:-h]
            return float(abs(np.mean(diffs ** 3))) / (np.std(series) ** 3 + 1e-10)
        except Exception:
            return 0.0

    def _bds_proxy(self, returns):
        """BDS-like nonlinearity proxy via correlation integral."""
        try:
            if len(returns) < 20:
                return 0.0
            r_std = np.std(returns)
            eps = 0.5 * r_std
            n = min(200, len(returns))
            rs = returns[:n]
            c1 = np.mean([np.mean(np.abs(rs - rs[i]) < eps) for i in range(0, n, max(1, n // 20))])
            c2_pairs = [(rs[i], rs[j]) for i in range(n - 1) for j in range(i + 1, min(i + 5, n))]
            if not c2_pairs:
                return 0.0
            c2 = np.mean([1.0 if (abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps) else 0.0
                           for a, b in zip(c2_pairs[:100], c2_pairs[1:101])])
            return float(abs(c2 - c1 ** 2))
        except Exception:
            return 0.0

    def _arch_test(self, returns):
        """Test for ARCH effects (volatility clustering)."""
        try:
            from statsmodels.stats.diagnostic import het_arch
            if len(returns) < 20:
                return False, 1.0
            stat, pval, _, _ = het_arch(returns, nlags=min(5, len(returns) // 4))
            return bool(pval < 0.05), float(pval)
        except Exception:
            # Manual ARCH(1) test
            try:
                sq = returns ** 2
                r, p = stats.pearsonr(sq[:-1], sq[1:])
                return bool(p < 0.05), float(p)
            except Exception:
                return False, 1.0

    def _teraesvirta_proxy(self, series):
        """Proxy for neural network nonlinearity test."""
        try:
            n = min(len(series), 200)
            s = series[:n]
            x = s[:-2]
            y = s[2:]
            residuals_linear = np.polyfit(x, y, 1)
            pred_linear = np.polyval(residuals_linear, x)
            res_linear = y - pred_linear
            # Add cubic term
            coeffs_cubic = np.polyfit(x, y, 3)
            pred_cubic = np.polyval(coeffs_cubic, x)
            res_cubic = y - pred_cubic
            ssr_linear = np.sum(res_linear ** 2)
            ssr_cubic = np.sum(res_cubic ** 2)
            # F-like statistic
            return float((ssr_linear - ssr_cubic) / (ssr_cubic / max(len(x) - 4, 1)))
        except Exception:
            return 0.0

    def _spectral_analysis(self, series):
        try:
            n = len(series)
            freqs, psd = signal.periodogram(series - np.mean(series))
            if len(psd) < 2 or np.sum(psd) < 1e-10:
                return 0.0, 1.0, 'noisy'
            dom_idx = np.argmax(psd[1:]) + 1
            dom_freq = float(freqs[dom_idx])
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-10)
            psd_norm = psd_norm[psd_norm > 0]
            spec_ent = float(entropy(psd_norm) / np.log(len(psd_norm) + 1))
            # Classify complexity
            top_frac = np.sum(np.sort(psd)[-3:]) / np.sum(psd)
            if top_frac > 0.7:
                complexity = 'periodic'
            elif spec_ent > 0.85:
                complexity = 'noisy'
            elif spec_ent > 0.6:
                complexity = 'mixed'
            else:
                complexity = 'chaotic'
            return dom_freq, spec_ent, complexity
        except Exception:
            return 0.0, 1.0, 'unknown'

    def _detect_changepoints(self, series):
        """PELT-like changepoint detection via cumulative sum."""
        try:
            from ruptures import Pelt
            model = Pelt(model='rbf', min_size=max(5, len(series) // 20))
            result = model.fit_predict(series.reshape(-1, 1), pen=3)
            cps = [r for r in result if r < len(series)]
            return cps[:-1], 'PELT-RBF'
        except Exception:
            pass
        # Fallback: cusum-based detection
        try:
            n = len(series)
            mean = np.mean(series)
            std = np.std(series) + 1e-10
            cusum = np.cumsum((series - mean) / std)
            cps = []
            window = max(10, n // 10)
            for i in range(window, n - window, window // 2):
                before = np.mean(series[max(0, i-window):i])
                after = np.mean(series[i:min(n, i+window)])
                if abs(after - before) > 1.5 * std:
                    cps.append(i)
            # Deduplicate
            deduped = []
            for cp in cps:
                if not deduped or cp - deduped[-1] > window // 2:
                    deduped.append(cp)
            return deduped, 'CUSUM-fallback'
        except Exception:
            return [], 'none'

    def _trend_analysis(self, series):
        try:
            n = len(series)
            x = np.arange(n)
            slope, intercept, r, p, se = stats.linregress(x, series)
            r2 = r ** 2
            if p > 0.1 or r2 < 0.05:
                direction = 'flat'
            elif slope > 0:
                direction = 'upward'
            else:
                direction = 'downward'
            return float(slope), float(p), float(r2), direction
        except Exception:
            return 0.0, 1.0, 0.0, 'flat'

    def _generate_insights(self, **kw) -> list[str]:
        """Generate human-readable insights from analysis results."""
        insights = []

        h = kw['hurst']
        if h > 0.6:
            insights.append(f"Strong momentum/persistence (H={h:.2f}) — trend-following strategies may work.")
        elif h < 0.4:
            insights.append(f"Mean-reverting dynamics (H={h:.2f}) — mean-reversion strategies applicable.")
        else:
            insights.append(f"Near-random-walk behavior (H={h:.2f}) — difficult to predict directionally.")

        if kw['arch_effect']:
            insights.append("ARCH effects detected — volatility clusters in time. Model vol separately.")

        if kw['time_rev'] > 0.5:
            insights.append(f"Significant time irreversibility (stat={kw['time_rev']:.2f}) — nonlinear dynamics confirmed.")

        if kw['adf_pval'] > 0.1:
            insights.append(f"Non-stationary series (ADF p={kw['adf_pval']:.3f}) — differencing or regime-aware modeling needed.")

        if abs(kw['skewness']) > 1.0:
            direction = "left" if kw['skewness'] < 0 else "right"
            insights.append(f"Heavy {direction} tail (skew={kw['skewness']:.2f}) — asymmetric risk profile.")

        if kw['kurtosis_val'] > 3:
            insights.append(f"Leptokurtic distribution (excess kurtosis={kw['kurtosis_val']:.2f}) — fat tails, extreme events more likely.")

        if kw['n_cp'] > 0:
            insights.append(f"{kw['n_cp']} structural break(s) detected — regime-aware modeling strongly recommended.")

        if kw['lyap'] > 0.1:
            insights.append(f"Positive Lyapunov exponent ({kw['lyap']:.3f}) — chaotic sensitivity to initial conditions.")

        if kw['seas_strength'] > 0.3:
            insights.append(f"Significant seasonality (strength={kw['seas_strength']:.2f}) — periodic component should be modeled.")

        sc = kw['spec_complex']
        if sc == 'periodic':
            insights.append("Spectrally periodic — dominant frequency component drives dynamics.")
        elif sc == 'chaotic':
            insights.append("Spectrally complex/chaotic — broad-spectrum dynamics, ensemble forecasting recommended.")

        if kw['mdd'] > 0.3:
            insights.append(f"Large maximum drawdown ({kw['mdd']*100:.1f}%) — high tail risk in the series.")

        if kw['trend_dir'] != 'flat':
            insights.append(f"Significant {kw['trend_dir']} trend detected — detrending may improve residual modeling.")

        return insights
