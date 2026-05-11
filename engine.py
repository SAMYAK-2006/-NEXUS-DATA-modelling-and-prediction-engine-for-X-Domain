"""
NEXUS — Prediction Engine
==========================
Modular, trust-weighted ensemble forecasting.
Models: ARIMA, GARCH, SDE (OU + Jump), Koopman/DMD, Transformer proxy,
        gradient boosting on SMA features, ensemble.

The prediction engine is deliberately modular: swap any model in/out.
Trust score α(t) from SMA gates how much weight goes to data-driven vs prior.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class ForecastResult:
    """Prediction output from a single model or ensemble."""
    model_name: str
    horizon: int
    point_forecast: np.ndarray          # (horizon,)
    lower_80: np.ndarray                # (horizon,)
    upper_80: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    in_sample_rmse: float
    in_sample_mae: float
    in_sample_mape: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    model_params: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class EnsembleForecast:
    """Trust-weighted ensemble of all model forecasts."""
    point_forecast: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    model_weights: dict[str, float]
    individual_forecasts: dict[str, ForecastResult]
    trust_weighted: bool
    current_trust: float
    regime_label: int
    horizon: int


# ─────────────────────────────────────────────
# Base model interface
# ─────────────────────────────────────────────

class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit_predict(
        self,
        series: np.ndarray,
        horizon: int,
        **kwargs
    ) -> ForecastResult:
        pass

    def _make_intervals(self, point, sigma, horizon):
        """Build forecast intervals assuming Gaussian residuals."""
        z80 = 1.282
        z95 = 1.960
        t_scale = np.sqrt(np.arange(1, horizon + 1))  # uncertainty grows with horizon
        return (
            point - z80 * sigma * t_scale,
            point + z80 * sigma * t_scale,
            point - z95 * sigma * t_scale,
            point + z95 * sigma * t_scale,
        )

    def _in_sample_metrics(self, actual, fitted):
        actual = np.asarray(actual)
        fitted = np.asarray(fitted)
        mask = np.isfinite(actual) & np.isfinite(fitted)
        a, f = actual[mask], fitted[mask]
        if len(a) < 2:
            return 0.0, 0.0, 0.0
        rmse = float(np.sqrt(np.mean((a - f) ** 2)))
        mae = float(np.mean(np.abs(a - f)))
        mape = float(np.mean(np.abs((a - f) / (np.abs(a) + 1e-10)))) * 100
        return rmse, mae, mape


# ─────────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────────

class ARIMAForecaster(BaseForecaster):
    name = "ARIMA"

    def fit_predict(self, series, horizon, **kwargs):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            # Auto-select d
            pval = adfuller(series, autolag='AIC')[1]
            d = 1 if pval > 0.05 else 0
            best_aic, best_order, best_model = np.inf, (1, d, 1), None
            for p in range(0, 4):
                for q in range(0, 4):
                    try:
                        m = ARIMA(series, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p, d, q)
                            best_model = m
                    except Exception:
                        pass
            if best_model is None:
                raise ValueError("ARIMA fit failed")
            fc = best_model.get_forecast(steps=horizon)
            mean_fc = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            ci80 = fc.conf_int(alpha=0.20)
            fitted = best_model.fittedvalues
            rmse, mae, mape = self._in_sample_metrics(series[d:], fitted[d:])
            return ForecastResult(
                model_name=self.name,
                horizon=horizon,
                point_forecast=np.array(mean_fc),
                lower_80=np.array(ci80.iloc[:, 0]),
                upper_80=np.array(ci80.iloc[:, 1]),
                lower_95=np.array(ci.iloc[:, 0]),
                upper_95=np.array(ci.iloc[:, 1]),
                in_sample_rmse=rmse,
                in_sample_mae=mae,
                in_sample_mape=mape,
                aic=best_aic,
                model_params={"order": best_order},
            )
        except Exception as e:
            return self._fallback(series, horizon, str(e))

    def _fallback(self, series, horizon, msg):
        mean = float(np.mean(series[-20:]))
        sigma = float(np.std(series[-20:], ddof=1))
        pt = np.full(horizon, mean)
        l80, u80, l95, u95 = self._make_intervals(pt, sigma, horizon)
        return ForecastResult(
            model_name=self.name + "(fallback)",
            horizon=horizon,
            point_forecast=pt, lower_80=l80, upper_80=u80,
            lower_95=l95, upper_95=u95,
            in_sample_rmse=sigma, in_sample_mae=sigma * 0.8, in_sample_mape=0.0,
            notes=f"ARIMA failed: {msg}. Using mean forecast."
        )


# ─────────────────────────────────────────────
# OU / SDE model (regime-aware)
# ─────────────────────────────────────────────

class OUForecaster(BaseForecaster):
    """
    Ornstein-Uhlenbeck (mean-reverting SDE) forecaster.
    E[X(t+h) | X(t)] = θ* + (X(t) - θ*) exp(-κh)
    """
    name = "OU-SDE"

    def fit_predict(self, series, horizon, trust_score: float = 1.0, **kwargs):
        try:
            n = len(series)
            dt = 1.0
            # MLE for OU: fit AR(1) regression
            x = series[:-1]
            y = series[1:]
            slope, intercept, _, _, _ = stats.linregress(x, y)
            # Map to OU parameters
            kappa = -np.log(max(slope, 1e-6)) / dt
            theta_star = intercept / (1 - slope)
            residuals = y - (intercept + slope * x)
            sigma = float(np.std(residuals, ddof=1))
            sigma_ou = sigma * np.sqrt(2 * kappa) if kappa > 0 else sigma

            x0 = series[-1]
            t_steps = np.arange(1, horizon + 1) * dt
            mean_fc = theta_star + (x0 - theta_star) * np.exp(-kappa * t_steps)

            # Analytical variance of OU
            var_fc = (sigma_ou ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t_steps)) if kappa > 0 else sigma ** 2 * t_steps
            std_fc = np.sqrt(var_fc)

            z80, z95 = 1.282, 1.960
            l80, u80 = mean_fc - z80 * std_fc, mean_fc + z80 * std_fc
            l95, u95 = mean_fc - z95 * std_fc, mean_fc + z95 * std_fc

            fitted = intercept + slope * x
            rmse, mae, mape = self._in_sample_metrics(y, fitted)

            return ForecastResult(
                model_name=self.name,
                horizon=horizon,
                point_forecast=mean_fc,
                lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=rmse, in_sample_mae=mae, in_sample_mape=mape,
                model_params={"kappa": kappa, "theta_star": theta_star, "sigma": sigma_ou},
            )
        except Exception as e:
            mean = float(np.mean(series[-20:]))
            sigma = float(np.std(series[-20:], ddof=1))
            pt = np.full(horizon, mean)
            l80, u80, l95, u95 = self._make_intervals(pt, sigma, horizon)
            return ForecastResult(
                model_name=self.name + "(fallback)", horizon=horizon,
                point_forecast=pt, lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=sigma, in_sample_mae=sigma * 0.8, in_sample_mape=0.0,
                notes=str(e)
            )


# ─────────────────────────────────────────────
# DMD / Koopman
# ─────────────────────────────────────────────

class KoopmanForecaster(BaseForecaster):
    """
    Dynamic Mode Decomposition — linear operator in lifted space.
    Principled linear embedding of nonlinear dynamics (Koopman theory).
    """
    name = "Koopman-DMD"

    def fit_predict(self, series, horizon, embed_dim: int = 5, **kwargs):
        try:
            n = len(series)
            # Hankel embedding (delay coordinates = Takens theorem)
            d = min(embed_dim, n // 3)
            X = np.array([series[i:n - d + i] for i in range(d)])  # (d, T-d)
            X1 = X[:, :-1]
            X2 = X[:, 1:]

            # SVD-based DMD
            U, s, Vt = np.linalg.svd(X1, full_matrices=False)
            r = min(d, np.sum(s > 1e-10 * s[0]))  # truncate small singular values
            U_r = U[:, :r]
            s_r = s[:r]
            Vt_r = Vt[:r, :]

            A_tilde = U_r.T @ X2 @ Vt_r.T @ np.diag(1.0 / s_r)
            evals, evecs = np.linalg.eig(A_tilde)

            # DMD modes
            Phi = X2 @ Vt_r.T @ np.diag(1.0 / s_r) @ evecs

            # Fit initial amplitudes
            b, _, _, _ = np.linalg.lstsq(Phi, X[:, 0], rcond=None)

            # Forecast
            state = X[:, -1].copy()
            forecasts = []
            for h in range(horizon):
                # Advance each mode
                b_future = b * (evals ** (h + 1))
                x_future = np.real(Phi @ b_future)
                forecasts.append(float(x_future[0]))  # first component = original series

            mean_fc = np.array(forecasts)

            # Residuals
            fitted_list = []
            for t in range(X.shape[1] - 1):
                b_t = b * (evals ** t)
                x_t = np.real(Phi @ b_t)[0]
                fitted_list.append(x_t)
            rmse, mae, mape = self._in_sample_metrics(series[d:d + len(fitted_list)], fitted_list)
            sigma = rmse if rmse > 0 else float(np.std(series[-20:], ddof=1))

            l80, u80, l95, u95 = self._make_intervals(mean_fc, sigma, horizon)

            return ForecastResult(
                model_name=self.name, horizon=horizon,
                point_forecast=mean_fc, lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=rmse, in_sample_mae=mae, in_sample_mape=mape,
                model_params={"embed_dim": d, "dmd_rank": r},
            )
        except Exception as e:
            mean = float(np.mean(series[-20:]))
            sigma = float(np.std(series[-20:], ddof=1))
            pt = np.full(horizon, mean)
            l80, u80, l95, u95 = self._make_intervals(pt, sigma, horizon)
            return ForecastResult(
                model_name=self.name + "(fallback)", horizon=horizon,
                point_forecast=pt, lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=sigma, in_sample_mae=sigma * 0.8, in_sample_mape=0.0,
                notes=str(e)
            )


# ─────────────────────────────────────────────
# SMA-feature gradient boosting
# ─────────────────────────────────────────────

class SMAGBForecaster(BaseForecaster):
    """
    Gradient boosting on SMA geometric features (D(t), α(t), speed, persistence, etc.)
    This is the 'geometry-informed ML' model.
    """
    name = "SMA-GradBoost"

    def fit_predict(self, series, horizon, sma_result=None, **kwargs):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.multioutput import MultiOutputRegressor

            if sma_result is None:
                raise ValueError("SMA result required")

            n_sma = len(sma_result.attractor_dist)
            n = min(len(series), n_sma + sma_result.window)

            # Build feature matrix from SMA outputs
            features = np.column_stack([
                sma_result.attractor_dist,
                sma_result.stress_rate,
                sma_result.trust_score,
                sma_result.speed,
                sma_result.persistence,
                sma_result.stress_dim,
                sma_result.theta_pca[:, :min(3, sma_result.theta_pca.shape[1])],
            ])

            # Target: series values at multiple future steps
            offset = sma_result.window
            series_aligned = series[offset:offset + n_sma]

            if len(series_aligned) < 30:
                raise ValueError("Not enough aligned data")

            X_list, y_list = [], []
            for t in range(len(features) - horizon):
                X_list.append(features[t])
                y_list.append(series_aligned[t:t + horizon] if t + horizon <= len(series_aligned) else None)

            X_list = [x for x, y in zip(X_list, y_list) if y is not None and len(y) == horizon]
            y_list = [y for y in y_list if y is not None and len(y) == horizon]

            if len(X_list) < 20:
                raise ValueError("Insufficient training samples")

            X_arr = np.array(X_list)
            y_arr = np.array(y_list)

            # Time-series split for evaluation
            split = int(0.8 * len(X_arr))
            X_train, X_test = X_arr[:split], X_arr[split:]
            y_train, y_test = y_arr[:split], y_arr[split:]

            model = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
            )
            model.fit(X_train, y_train)

            # In-sample metrics
            pred_test = model.predict(X_test)
            rmse, mae, mape = self._in_sample_metrics(y_test.ravel(), pred_test.ravel())

            # Forecast
            current_features = features[-1].reshape(1, -1)
            mean_fc = model.predict(current_features)[0]

            # Uncertainty from test residuals
            residuals = y_test - pred_test
            sigma = float(np.std(residuals))
            l80, u80, l95, u95 = self._make_intervals(mean_fc, sigma, horizon)

            return ForecastResult(
                model_name=self.name, horizon=horizon,
                point_forecast=mean_fc, lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=rmse, in_sample_mae=mae, in_sample_mape=mape,
                model_params={"n_estimators": 100},
            )
        except Exception as e:
            mean = float(np.mean(series[-20:]))
            sigma = float(np.std(series[-20:], ddof=1))
            pt = np.full(horizon, mean)
            l80, u80, l95, u95 = self._make_intervals(pt, sigma, horizon)
            return ForecastResult(
                model_name=self.name + "(fallback)", horizon=horizon,
                point_forecast=pt, lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
                in_sample_rmse=sigma, in_sample_mae=sigma * 0.8, in_sample_mape=0.0,
                notes=str(e)
            )


# ─────────────────────────────────────────────
# Ensemble: trust-weighted combination
# ─────────────────────────────────────────────

class EnsemblePredictor:
    """
    Trust-weighted ensemble.
    When α(t) is high (near attractor, stable), weight data-driven models more.
    When α(t) is low (escape phase, stressed), weight structural/OU prior more.
    """

    def __init__(self):
        self.models = [
            ARIMAForecaster(),
            OUForecaster(),
            KoopmanForecaster(),
            SMAGBForecaster(),
        ]

    def predict(
        self,
        series: np.ndarray,
        horizon: int,
        sma_result=None,
        current_trust: float = 0.8,
        current_regime: int = 0,
    ) -> EnsembleForecast:

        results = {}
        for model in self.models:
            try:
                if isinstance(model, SMAGBForecaster):
                    r = model.fit_predict(series, horizon, sma_result=sma_result)
                else:
                    r = model.fit_predict(series, horizon)
                results[model.name] = r
            except Exception as e:
                pass  # skip failed models silently

        if not results:
            # Absolute fallback
            mean = float(np.mean(series[-20:]))
            sigma = float(np.std(series[-20:], ddof=1))
            pt = np.full(horizon, mean)
            z95 = 1.96
            t_scale = np.sqrt(np.arange(1, horizon + 1))
            return EnsembleForecast(
                point_forecast=pt,
                lower_80=pt - 1.282 * sigma * t_scale,
                upper_80=pt + 1.282 * sigma * t_scale,
                lower_95=pt - z95 * sigma * t_scale,
                upper_95=pt + z95 * sigma * t_scale,
                model_weights={},
                individual_forecasts={},
                trust_weighted=False,
                current_trust=current_trust,
                regime_label=current_regime,
                horizon=horizon,
            )

        # Compute weights
        # Base weights by RMSE (lower RMSE = higher weight)
        rmses = {name: max(r.in_sample_rmse, 1e-6) for name, r in results.items()}
        inv_rmse = {name: 1.0 / v for name, v in rmses.items()}
        total = sum(inv_rmse.values())
        base_weights = {name: v / total for name, v in inv_rmse.items()}

        # Trust modulation: α(t) boosts data-driven, penalises when low
        alpha = float(np.clip(current_trust, 0.05, 0.99))
        structural_models = {"OU-SDE"}  # these get boosted when trust is low
        final_weights = {}
        for name, w in base_weights.items():
            if name in structural_models:
                # Boost when low trust
                boost = 1.0 + (1.0 - alpha) * 2.0
                final_weights[name] = w * boost
            else:
                # Scale with trust
                final_weights[name] = w * (0.5 + alpha * 0.5)
        # Renormalise
        total_w = sum(final_weights.values())
        final_weights = {k: v / total_w for k, v in final_weights.items()}

        # Weighted ensemble
        ensemble_pt = np.zeros(horizon)
        ensemble_l80 = np.zeros(horizon)
        ensemble_u80 = np.zeros(horizon)
        ensemble_l95 = np.zeros(horizon)
        ensemble_u95 = np.zeros(horizon)
        for name, r in results.items():
            w = final_weights.get(name, 0.0)
            pt = np.asarray(r.point_forecast)[:horizon]
            if len(pt) < horizon:
                pt = np.pad(pt, (0, horizon - len(pt)), mode='edge')
            ensemble_pt += w * pt
            ensemble_l80 += w * np.asarray(r.lower_80)[:horizon]
            ensemble_u80 += w * np.asarray(r.upper_80)[:horizon]
            ensemble_l95 += w * np.asarray(r.lower_95)[:horizon]
            ensemble_u95 += w * np.asarray(r.upper_95)[:horizon]

        return EnsembleForecast(
            point_forecast=ensemble_pt,
            lower_80=ensemble_l80,
            upper_80=ensemble_u80,
            lower_95=ensemble_l95,
            upper_95=ensemble_u95,
            model_weights=final_weights,
            individual_forecasts=results,
            trust_weighted=True,
            current_trust=current_trust,
            regime_label=current_regime,
            horizon=horizon,
        )
