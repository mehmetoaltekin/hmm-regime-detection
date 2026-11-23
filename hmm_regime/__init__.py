"""
hmm_regime: walk-forward Gaussian HMM regime detection with causal filtering.

    from hmm_regime import WalkForwardHMM, prepare_hmm_features, add_filtered_regime
"""

from hmm_regime.data_loader import (
    compute_log_range_variance,
    compute_log_returns,
    compute_parkinson_volatility,
    compute_realized_volatility,
    fetch_returns,
    fetch_sample_data,
    prepare_hmm_features,
    simulate_regime_ohlc,
)
from hmm_regime.hmm_model import MarketRegimeHMM, WalkForwardHMM, stationary_distribution
from hmm_regime.model_selection import (
    count_free_parameters,
    evaluate_regime_models,
    select_model,
    select_n_components,
)
from hmm_regime.whipsaw import (
    add_filtered_regime,
    calm_vs_stress,
    change_lags,
    detection_lag,
    filter_regimes,
    switch_stats,
)

__version__ = "0.2.0"

__all__ = [
    "MarketRegimeHMM",
    "WalkForwardHMM",
    "stationary_distribution",
    "prepare_hmm_features",
    "compute_log_returns",
    "compute_realized_volatility",
    "compute_parkinson_volatility",
    "compute_log_range_variance",
    "fetch_returns",
    "fetch_sample_data",
    "simulate_regime_ohlc",
    "count_free_parameters",
    "evaluate_regime_models",
    "select_model",
    "select_n_components",
    "filter_regimes",
    "add_filtered_regime",
    "calm_vs_stress",
    "switch_stats",
    "change_lags",
    "detection_lag",
    "__version__",
]
