"""
NEXUS — Core Geometric Engine
==============================
Implementation of the Statistical Manifold Attractor (SMA) framework.
Based on: "Statistical Manifold Geometry of Regime Dynamics" (Jain, 2026)

This module computes:
  - Statistical feature map Φ: θ(t) ∈ R^k
  - Trajectory kinematics: velocity v(t), speed s(t), acceleration a(t)
  - Displacement and persistence P(t, W)
  - Attractor θ* and distance D(t)
  - Data-trust score α(t)
  - Regime detection and escape phase identification
  - Applicability check (attractor may or may not exist)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, normaltest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class ApplicabilityReport:
    """Result of checking whether SMA framework is applicable."""
    applicable: bool
    attractor_exists: bool
    mean_reverting: bool
    bounded: bool
    sufficient_length: bool
    persistence_score: float      # E[P(t,W)] — negative = mean reverting
    persistence_pvalue: float
    directional_accuracy: float   # < 0.5 confirms mean reversion
    stationarity_verdict: str     # 'stationary' | 'non-stationary' | 'uncertain'
    adf_pvalue: float
    notes: list[str] = field(default_factory=list)

    def __str__(self):
        lines = [
            "── SMA Applicability Report ──────────────────",
            f"  Applicable         : {self.applicable}",
            f"  Attractor exists   : {self.attractor_exists}",
            f"  Mean reverting     : {self.mean_reverting}  (E[P]={self.persistence_score:.4f}, p={self.persistence_pvalue:.4f})",
            f"  Directional acc.   : {self.directional_accuracy:.3f}  (< 0.5 → mean reversion)",
            f"  Bounded            : {self.bounded}",
            f"  Stationarity       : {self.stationarity_verdict}  (ADF p={self.adf_pvalue:.4f})",
        ]
        for n in self.notes:
            lines.append(f"  ⚠ {n}")
        lines.append("──────────────────────────────────────────────")
        return "\n".join(lines)


@dataclass
class RegimeSegment:
    start: int
    end: int
    label: int
    mean_trust: float
    mean_stress: float
    is_escape: bool


@dataclass
class SMAResult:
    """Full output of the geometric engine."""
    theta: np.ndarray              # (T, k) — statistical feature vectors
    theta_pca: np.ndarray          # (T, m) — PCA-reduced
    velocity: np.ndarray           # (T, k)
    speed: np.ndarray              # (T,)
    acceleration: np.ndarray       # (T, k)
    acceleration_mag: np.ndarray   # (T,)
    displacement: np.ndarray       # (T, k) — d(t, W)
    persistence: np.ndarray        # (T,) — P(t, W)
    attractor: np.ndarray          # (k,) — θ*
    attractor_dist: np.ndarray     # (T,) — D(t)
    stress_norm: np.ndarray        # (T,) — D̂(t)
    stress_rate: np.ndarray        # (T,) — Ḋ(t)
    trust_score: np.ndarray        # (T,) — α(t) ∈ (0,1)
    regime_labels: np.ndarray      # (T,) — integer regime IDs
    escape_phases: np.ndarray      # (T,) — bool mask
    stress_dim: np.ndarray         # (T,) — effective dimensionality
    pca_variance_ratio: np.ndarray # (m,)
    applicability: ApplicabilityReport
    feature_names: list[str]
    window: int


# ─────────────────────────────────────────────
# Feature map Φ
# ─────────────────────────────────────────────

def _compute_feature_vector(window_data: np.ndarray) -> np.ndarray:
    """
    Compute statistical feature vector θ from a rolling window.
    Implements Eq. (1) from the SMA paper + additional nonlinear features.

    Features:
      0  μ    — mean
      1  σ²   — variance
      2  γ    — skewness
      3  κ    — excess kurtosis
      4  ρ¹   — lag-1 autocorrelation
      5  ρ²   — lag-2 autocorrelation
      6  ρ⁵   — lag-5 autocorrelation
      7  ν    — volatility of abs values
      8  σ²↓  — downside variance
      9  MAD  — mean absolute deviation
      10 DD   — max drawdown
      11 G    — gain/pain ratio
      12 H    — Hurst exponent (nonlinear memory)
      13 IQR  — interquartile range (robust spread)
    """
    r = window_data.astype(float)
    W = len(r)
    mu = np.mean(r)
    sigma2 = np.var(r, ddof=1) if W > 1 else 1e-10
    sigma = np.sqrt(sigma2) + 1e-10

    # Skewness and kurtosis
    gamma = np.mean((r - mu) ** 3) / (sigma ** 3) if sigma > 1e-10 else 0.0
    kappa = np.mean((r - mu) ** 4) / (sigma ** 4) - 3 if sigma > 1e-10 else 0.0

    # Autocorrelations
    def autocorr(lag):
        if W <= lag:
            return 0.0
        c = np.corrcoef(r[:-lag], r[lag:])[0, 1]
        return c if np.isfinite(c) else 0.0

    rho1 = autocorr(1)
    rho2 = autocorr(2)
    rho5 = autocorr(min(5, W // 2))

    # Volatility of absolute values
    nu = np.std(np.abs(r)) if W > 1 else 0.0

    # Downside variance
    neg = r[r < 0]
    sigma2_down = np.var(neg) if len(neg) > 1 else 0.0

    # MAD
    mad = np.mean(np.abs(r - mu))

    # Max drawdown
    cumulative = np.cumprod(1 + np.clip(r, -0.999, None))
    running_max = np.maximum.accumulate(cumulative)
    dd_series = (running_max - cumulative) / (running_max + 1e-10)
    dd = np.max(dd_series)

    # Gain/pain ratio
    gains = r[r > 0]
    losses = r[r < 0]
    g_mean = np.mean(gains) if len(gains) > 0 else 0.0
    l_mean = np.abs(np.mean(losses)) if len(losses) > 0 else 1e-10
    gain_pain = g_mean / (l_mean + 1e-10)

    # Hurst exponent via R/S analysis
    hurst = _hurst_rs(r)

    # IQR
    iqr = np.percentile(r, 75) - np.percentile(r, 25)

    features = np.array([
        mu, sigma2, gamma, kappa,
        rho1, rho2, rho5,
        nu, sigma2_down, mad, dd, gain_pain,
        hurst, iqr
    ], dtype=float)

    # Replace any NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


def _hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent via simplified R/S analysis."""
    n = len(series)
    if n < 20:
        return 0.5
    try:
        lags = [n // 4, n // 2]
        rs_vals = []
        for lag in lags:
            if lag < 4:
                continue
            sub = series[:lag]
            mean_sub = np.mean(sub)
            dev = np.cumsum(sub - mean_sub)
            r = np.max(dev) - np.min(dev)
            s = np.std(sub, ddof=1)
            if s > 1e-10:
                rs_vals.append((lag, r / s))
        if len(rs_vals) < 2:
            return 0.5
        log_lags = np.log([x[0] for x in rs_vals])
        log_rs = np.log([x[1] for x in rs_vals])
        h = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


FEATURE_NAMES = [
    "mean", "variance", "skewness", "kurtosis",
    "autocorr_1", "autocorr_2", "autocorr_5",
    "vol_abs", "downside_var", "MAD", "max_drawdown", "gain_pain",
    "hurst", "IQR"
]


# ─────────────────────────────────────────────
# Applicability check
# ─────────────────────────────────────────────

def check_applicability(
    series: np.ndarray,
    window: int,
    min_length: int = 100
) -> ApplicabilityReport:
    """
    Check whether the SMA framework is applicable to this time series.
    The attractor framework is NOT universal — this function diagnoses it.
    """
    from statsmodels.tsa.stattools import adfuller

    notes = []
    n = len(series)

    # 1. Length check
    sufficient_length = n >= min_length
    if not sufficient_length:
        notes.append(f"Series too short ({n} < {min_length}). SMA results may be unreliable.")

    # 2. Boundedness — check via empirical range stability
    half = n // 2
    range1 = np.ptp(series[:half])
    range2 = np.ptp(series[half:])
    bounded = (range2 / (range1 + 1e-10)) < 3.0
    if not bounded:
        notes.append("Series appears unbounded (explosive). Attractor may not exist.")

    # 3. ADF test for stationarity
    try:
        adf_result = adfuller(series, autolag='AIC')
        adf_pvalue = adf_result[1]
        if adf_pvalue < 0.05:
            stationarity_verdict = 'stationary'
        elif adf_pvalue < 0.15:
            stationarity_verdict = 'uncertain'
        else:
            stationarity_verdict = 'non-stationary'
            notes.append("Series is non-stationary (ADF p>{:.3f}). Attractor may migrate.".format(adf_pvalue))
    except Exception:
        adf_pvalue = 1.0
        stationarity_verdict = 'uncertain'
        notes.append("ADF test failed.")

    # 4. Compute θ(t) and test displacement persistence
    theta_matrix = _compute_theta_matrix(series, window)
    T_theta = len(theta_matrix)
    W_disp = max(window // 2, 5)

    persistence_scores = []
    for t in range(W_disp, T_theta - W_disp):
        d_past = theta_matrix[t] - theta_matrix[t - W_disp]
        d_future = theta_matrix[t + W_disp] - theta_matrix[t]
        n_past = np.linalg.norm(d_past)
        n_future = np.linalg.norm(d_future)
        if n_past > 1e-10 and n_future > 1e-10:
            p = np.dot(d_past, d_future) / (n_past * n_future)
            persistence_scores.append(p)

    if len(persistence_scores) > 10:
        persistence_arr = np.array(persistence_scores)
        t_stat, p_val = stats.ttest_1samp(persistence_arr, 0.0)
        mean_p = float(np.mean(persistence_arr))
        da = float(np.mean(persistence_arr > 0))
        mean_reverting = (mean_p < 0) and (p_val < 0.1)
        attractor_exists = mean_reverting and bounded
    else:
        mean_p = 0.0
        p_val = 1.0
        da = 0.5
        mean_reverting = False
        attractor_exists = False
        notes.append("Insufficient data to compute displacement persistence.")

    if not mean_reverting:
        notes.append("No significant mean reversion detected. SMA attractor framework may not apply, but geometric features still computed.")

    applicable = sufficient_length  # We always compute geometry; attractor is conditional
    if not attractor_exists:
        notes.append("Running in GEOMETRIC-ONLY mode. Trust score α(t) and attractor distance D(t) are descriptive, not guaranteed by theory.")

    return ApplicabilityReport(
        applicable=applicable,
        attractor_exists=attractor_exists,
        mean_reverting=mean_reverting,
        bounded=bounded,
        sufficient_length=sufficient_length,
        persistence_score=mean_p,
        persistence_pvalue=p_val,
        directional_accuracy=da,
        stationarity_verdict=stationarity_verdict,
        adf_pvalue=adf_pvalue,
        notes=notes
    )


def _compute_theta_matrix(series: np.ndarray, window: int) -> np.ndarray:
    """Compute full θ(t) matrix from series."""
    n = len(series)
    rows = []
    for t in range(window, n):
        w = series[t - window:t]
        rows.append(_compute_feature_vector(w))
    return np.array(rows) if rows else np.zeros((0, len(FEATURE_NAMES)))


# ─────────────────────────────────────────────
# Main SMA engine
# ─────────────────────────────────────────────

class GeometricEngine:
    """
    Core SMA geometric engine.
    Takes a 1D time series, returns full SMAResult.
    """

    def __init__(
        self,
        window: int = 60,
        pca_components: int = 3,
        displacement_window: Optional[int] = None,
        trust_beta1: float = 2.0,
        trust_beta2: float = 1.0,
        regime_n_clusters: int = 3,
        escape_percentile: float = 0.90,
    ):
        self.window = window
        self.pca_components = pca_components
        self.displacement_window = displacement_window or max(window // 3, 5)
        self.trust_beta1 = trust_beta1
        self.trust_beta2 = trust_beta2
        self.regime_n_clusters = regime_n_clusters
        self.escape_percentile = escape_percentile

    def fit_transform(self, series: np.ndarray) -> SMAResult:
        """Run full SMA pipeline on a 1D time series."""
        series = np.asarray(series, dtype=float)
        series = series[np.isfinite(series)]  # drop NaN/inf

        # Applicability check
        applicability = check_applicability(series, self.window)

        # Feature map: θ(t)
        theta = _compute_theta_matrix(series, self.window)
        T = len(theta)

        if T < 10:
            raise ValueError(f"Series too short after windowing. Got {T} θ vectors, need ≥ 10.")

        # Standardise
        scaler = StandardScaler()
        theta_scaled = scaler.fit_transform(theta)

        # PCA
        n_comp = min(self.pca_components, theta_scaled.shape[1], T)
        pca = PCA(n_components=n_comp)
        theta_pca = pca.fit_transform(theta_scaled)

        # Kinematics
        velocity = np.diff(theta_scaled, axis=0, prepend=theta_scaled[[0]])
        speed = np.linalg.norm(velocity, axis=1)
        acceleration = np.diff(velocity, axis=0, prepend=velocity[[0]])
        accel_mag = np.linalg.norm(acceleration, axis=1)

        # Displacement and persistence
        W_d = self.displacement_window
        displacement = np.zeros_like(theta_scaled)
        persistence = np.zeros(T)
        for t in range(W_d, T):
            d = theta_scaled[t] - theta_scaled[t - W_d]
            displacement[t] = d
        for t in range(W_d, T - W_d):
            d_past = theta_scaled[t] - theta_scaled[t - W_d]
            d_future = theta_scaled[t + W_d] - theta_scaled[t]
            n_p = np.linalg.norm(d_past)
            n_f = np.linalg.norm(d_future)
            if n_p > 1e-10 and n_f > 1e-10:
                persistence[t] = np.dot(d_past, d_future) / (n_p * n_f)

        # Attractor θ* (sample mean of standardised features)
        attractor = np.mean(theta_scaled, axis=0)

        # Attractor distance D(t)
        attractor_dist = np.linalg.norm(theta_scaled - attractor, axis=1)

        # Normalised stress index D̂(t)
        running_q95 = np.array([
            np.percentile(attractor_dist[:max(t+1, 2)], 95)
            for t in range(T)
        ])
        stress_norm = attractor_dist / (running_q95 + 1e-10)

        # Stress rate Ḋ(t)
        stress_rate = np.diff(attractor_dist, prepend=attractor_dist[0])

        # Trust score α(t) = σ(-β₁D̂(t) - β₂Ḋ(t))
        logit = -self.trust_beta1 * stress_norm - self.trust_beta2 * np.clip(stress_rate, 0, None)
        trust_score = 1.0 / (1.0 + np.exp(-logit))

        # Stress dimensionality (local covariance participation ratio)
        stress_dim = self._compute_stress_dim(theta_pca, local_window=max(10, self.window // 3))

        # Regime detection via k-means on PCA space
        regime_labels = self._detect_regimes(theta_pca, attractor_dist, speed)

        # Escape phase detection
        escape_phases = self._detect_escape(attractor_dist, stress_rate, persistence)

        return SMAResult(
            theta=theta_scaled,
            theta_pca=theta_pca,
            velocity=velocity,
            speed=speed,
            acceleration=acceleration,
            acceleration_mag=accel_mag,
            displacement=displacement,
            persistence=persistence,
            attractor=attractor,
            attractor_dist=attractor_dist,
            stress_norm=stress_norm,
            stress_rate=stress_rate,
            trust_score=trust_score,
            regime_labels=regime_labels,
            escape_phases=escape_phases,
            stress_dim=stress_dim,
            pca_variance_ratio=pca.explained_variance_ratio_,
            applicability=applicability,
            feature_names=FEATURE_NAMES,
            window=self.window,
        )

    def _compute_stress_dim(self, theta_pca: np.ndarray, local_window: int) -> np.ndarray:
        T = len(theta_pca)
        sdim = np.ones(T)
        for t in range(local_window, T):
            local = theta_pca[t - local_window:t]
            cov = np.cov(local.T)
            if cov.ndim == 0:
                sdim[t] = 1.0
                continue
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 0)
            s = np.sum(eigvals)
            if s > 1e-10:
                sdim[t] = s ** 2 / np.sum(eigvals ** 2)
            else:
                sdim[t] = 1.0
        return sdim

    def _detect_regimes(
        self,
        theta_pca: np.ndarray,
        attractor_dist: np.ndarray,
        speed: np.ndarray
    ) -> np.ndarray:
        """Regime detection via quantile-based stress clustering."""
        T = len(theta_pca)
        labels = np.zeros(T, dtype=int)

        # Use attractor distance to define regimes
        q33 = np.percentile(attractor_dist, 33)
        q66 = np.percentile(attractor_dist, 66)

        for t in range(T):
            d = attractor_dist[t]
            if d <= q33:
                labels[t] = 0   # near-attractor / calm
            elif d <= q66:
                labels[t] = 1   # intermediate
            else:
                labels[t] = 2   # high stress / far from attractor

        return labels

    def _detect_escape(
        self,
        attractor_dist: np.ndarray,
        stress_rate: np.ndarray,
        persistence: np.ndarray
    ) -> np.ndarray:
        """
        Escape phase: D(t) > q90 AND Ḋ(t) > 0 AND P(t,W) > 0 locally.
        Proposition 3.2 from SMA paper.
        """
        q90 = np.percentile(attractor_dist, self.escape_percentile * 100)
        escape = (
            (attractor_dist > q90) &
            (stress_rate > 0) &
            (persistence > 0)
        )
        return escape
