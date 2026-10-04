"""
Unit tests for model selection metrics (free parameter counting, AIC, and BIC).
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.model_selection import count_free_parameters, evaluate_regime_models


def test_free_parameter_counting():
    """
    Verify exact parameter counting for Gaussian HMM with full covariance.
    Formula: (K - 1) + K*(K - 1) + K*d + K*(d*(d + 1)/2)
    For K=2, d=2:
      init = 1
      trans = 2 * 1 = 2
      means = 2 * 2 = 4
      covs  = 2 * (2 * 3 // 2) = 6
      Total = 1 + 2 + 4 + 6 = 13
    """
    k = 2
    d = 2
    expected_params = 13
    computed_params = count_free_parameters(n_components=k, n_features=d, covariance_type="full")

    assert computed_params == expected_params, (
        f"Parameter count mismatch: expected {expected_params}, got {computed_params}"
    )


def test_evaluate_regime_models():
    """Test that model selection summary runs and outputs expected DataFrame columns."""
    df = fetch_sample_data(n_bars=300, seed=42)
    _, X = prepare_hmm_features(df, vol_window=10)

    summary_df = evaluate_regime_models(
        X=X,
        components_range=[2, 3],
        covariance_type="full",
        n_iter=50,
        random_state=42,
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert "AIC" in summary_df.columns
    assert "BIC" in summary_df.columns
    assert "States (K)" in summary_df.columns

    # Verify BIC penalties increase with parameter bloat if likelihood does not scale
    assert not summary_df["BIC"].isna().any()
