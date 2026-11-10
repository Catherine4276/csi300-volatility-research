"""Financial time-series diagnostic tests used in the research report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, adfuller
import warnings


def adf_diagnostic(series: pd.Series, name: str) -> dict[str, float | str]:
    """Augmented Dickey-Fuller test with a constant."""

    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        statistic, p_value, lags, observations, critical, _ = adfuller(
            clean.to_numpy(),
            autolag="AIC",
            regression="c",
            result_object=False,
        )
    return {
        "name": name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": int(lags),
        "observations": int(observations),
        "critical_5pct": float(critical["5%"]),
    }


def volatility_clustering_diagnostic(
    returns: pd.Series,
    lags: int = 22,
) -> dict[str, object]:
    """Measure autocorrelation and ARCH effects in squared log returns."""

    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    squared = np.square(clean.to_numpy())
    autocorrelations = acf(squared, nlags=lags, fft=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        arch_lm, arch_pvalue, arch_f, arch_f_pvalue = het_arch(squared, nlags=lags)
    return {
        "observations": int(len(squared)),
        "squared_return_acf_lag1": float(autocorrelations[1]),
        "squared_return_acf_lag5": float(autocorrelations[min(5, lags)]),
        "arch_lm_statistic": float(arch_lm),
        "arch_lm_p_value": float(arch_pvalue),
        "arch_f_statistic": float(arch_f),
        "arch_f_p_value": float(arch_f_pvalue),
        "lags": int(lags),
    }


def return_descriptives(returns: pd.Series) -> dict[str, float]:
    """Summarize the return distribution and tail behavior."""

    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "observations": float(len(clean)),
        "mean": float(clean.mean()),
        "standard_deviation": float(clean.std()),
        "skewness": float(clean.skew()),
        "excess_kurtosis": float(clean.kurtosis()),
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
        "one_percent_quantile": float(clean.quantile(0.01)),
        "ninety_nine_percent_quantile": float(clean.quantile(0.99)),
    }


def build_diagnostic_report(features: pd.DataFrame) -> dict[str, object]:
    """Run core stationarity, autocorrelation, and ARCH diagnostics."""

    return {
        "log_return_adf": adf_diagnostic(features["log_return"], "daily_log_return"),
        "rv_proxy_adf": adf_diagnostic(features["rv_proxy"], "daily_variance_proxy"),
        "return_descriptives": return_descriptives(features["log_return"]),
        "volatility_clustering": volatility_clustering_diagnostic(
            features["log_return"], lags=22
        ),
    }

