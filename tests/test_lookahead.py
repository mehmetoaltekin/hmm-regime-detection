"""
Unit tests asserting the absence of look-ahead bias and verifying state alignment.
Validates that future observations do not contaminate past filtered probabilities.
"""

import pytest
import numpy as np

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import MarketRegimeHMM


@pytest.fixture
def sample_feature_matrix() -> np.ndarray:
    """Fixture providing clean stationary features from synthetic price history."""
    df = fetch_sample_data(n_bars=600, seed=42)
    _, X = prepare_hmm_features(df, vol_window=21)
    return X


def test_state_sorting_by_variance(sample_feature_matrix: np.ndarray) -> None:
    """
    Test that State 0 is strictly lower variance than State 1 (label alignment).
    Guarantees consistent economic interpretation across re-fits.
    """
    X = sample_feature_matrix[:300]
    model = MarketRegimeHMM(n_components=2, covariance_type="full", random_state=42)
    model.fit(X)

    # Compute trace of covariance matrices
    var_state_0 = np.trace(model.model.covars_[0])
    var_state_1 = np.trace(model.model.covars_[1])

    assert var_state_0 < var_state_1, (
        f"State ordering failure: State 0 var ({var_state_0:.6f}) "
        f"is not strictly less than State 1 var ({var_state_1:.6f})"
    )


def test_zero_lookahead_bias_on_future_truncation(sample_feature_matrix: np.ndarray) -> None:
    """
    Core Invariance Test:
    Filtered probabilities P(S_t | Y_{1:t}) computed up to t must be bitwise
    identical regardless of whether future data Y_{t+1:T} exists or not.
    """
    # 1. Fit parameters on an in-sample window
    train_size = 252
    X_train = sample_feature_matrix[:train_size]
    X_eval = sample_feature_matrix[train_size:train_size + 150]

    model = MarketRegimeHMM(n_components=2, random_state=42).fit(X_train)

    cutoff_t = 75  # Midpoint evaluation time

    # Probability computed knowing only up to cutoff_t
    probs_historical = model.predict_filtered_proba(X_eval[:cutoff_t])

    # Probability computed on the entire horizon (up to t=150)
    probs_full = model.predict_filtered_proba(X_eval)

    # Assertion: Historical predictions must not change with new future information
    np.testing.assert_allclose(
        probs_historical,
        probs_full[:cutoff_t],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Look-ahead leakage detected: Past filtered probabilities altered by future data existence.",
    )


def test_zero_lookahead_bias_on_future_mutation(sample_feature_matrix: np.ndarray) -> None:
    """
    Permutation Test:
    Modifying, spiking, or corrupting future data points (t > cutoff)
    must produce zero change in regime probabilities at or before cutoff.
    """
    X_train = sample_feature_matrix[:252]
    X_eval = sample_feature_matrix[252:400].copy()

    model = MarketRegimeHMM(n_components=2, random_state=42).fit(X_train)

    cutoff_t = 50
    probs_baseline = model.predict_filtered_proba(X_eval)

    # Corrupt future values with massive market crash / anomaly shocks
    X_corrupted = X_eval.copy()
    X_corrupted[cutoff_t:, 0] = -0.50  # 50% single-day crash simulation
    X_corrupted[cutoff_t:, 1] = 10.00  # Massive volatility spike

    probs_after_future_shock = model.predict_filtered_proba(X_corrupted)

    # Verify past remains immutable
    np.testing.assert_allclose(
        probs_baseline[:cutoff_t],
        probs_after_future_shock[:cutoff_t],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Future data shock leaked into past regime probabilities.",
    )

    # Verify future probabilities did react
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            probs_baseline[cutoff_t:],
            probs_after_future_shock[cutoff_t:],
            atol=1e-4,
        )


def test_probability_simplex_constraints(sample_feature_matrix: np.ndarray) -> None:
    """Assert filtered regime probabilities strictly sum to 1.0 across all bars."""
    model = MarketRegimeHMM(n_components=2, random_state=42).fit(sample_feature_matrix[:200])
    probs = model.predict_filtered_proba(sample_feature_matrix[200:300])

    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(
        row_sums,
        np.ones(len(probs)),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Filtered probabilities do not sum to unity on the probability simplex.",
    )
