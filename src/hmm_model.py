"""
Market Regime Detection using Gaussian Hidden Markov Models.
Implements out-of-sample forward filtering and variance-based state sorting.
"""

from typing import Optional
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class MarketRegimeHMM:
    """
    Gaussian Hidden Markov Model for financial regime detection.
    
    Addresses two critical quantitative pitfalls:
    1. Label Switching: States are dynamically sorted by variance.
       State 0 is strictly the lowest volatility regime.
    2. Look-Ahead Bias: Provides sequential forward filtering
       P(S_t | Y_{1:t}) instead of smoothed inference P(S_t | Y_{1:T}).
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_order_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MarketRegimeHMM":
        """
        Fit the Gaussian HMM and sort hidden states by emission variance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Stationary feature matrix (e.g., returns, rolling volatility).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        raw_model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        raw_model.fit(X)

        # Sort states by total variance (trace of covariance matrix)
        if self.covariance_type == "full":
            variances = np.array([np.trace(cov) for cov in raw_model.covars_])
        elif self.covariance_type == "diag":
            variances = np.array([np.sum(cov) for cov in raw_model.covars_])
        else:
            variances = raw_model.covars_.flatten()

        order = np.argsort(variances)
        self.state_order_ = order

        # Re-index parameters in-place to preserve internal attributes (n_features)
        raw_model.startprob_ = raw_model.startprob_[order]
        raw_model.transmat_ = raw_model.transmat_[order, :][:, order]
        raw_model.means_ = raw_model.means_[order]
        raw_model.covars_ = raw_model.covars_[order]

        self.model = raw_model
        return self

    def predict_filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sequential forward-filtered probabilities P(S_t | Y_{1:t}).
        Guarantees strictly zero look-ahead bias (no future data used).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Out-of-sample or in-sample feature series.

        Returns
        -------
        filtered_probs : np.ndarray of shape (n_samples, n_components)
            Sequential regime probabilities at each time step t.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before computing probabilities.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        filtered_probs = np.zeros((n_samples, self.n_components))

        # Forward algorithm pass
        framelogprob = self.model._compute_log_likelihood(X)
        log_transmat = np.log(self.model.transmat_)
        log_startprob = np.log(self.model.startprob_)

        curr_log_alpha = log_startprob + framelogprob[0]
        # Normalize in log space to obtain filtered prob at t=0
        log_norm = np.logaddexp.reduce(curr_log_alpha)
        filtered_probs[0] = np.exp(curr_log_alpha - log_norm)

        for t in range(1, n_samples):
            # Prior: P(S_t | Y_{1:t-1}) = sum_i P(S_{t-1} | Y_{1:t-1}) * A_{ij}
            log_prior = np.logaddexp.reduce(
                curr_log_alpha[:, np.newaxis] + log_transmat, axis=0
            )
            # Update: P(S_t | Y_{1:t}) ~ Emission * Prior
            curr_log_alpha = log_prior + framelogprob[t]
            log_norm = np.logaddexp.reduce(curr_log_alpha)
            filtered_probs[t] = np.exp(curr_log_alpha - log_norm)

        return filtered_probs

    def predict_regimes(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely regime index at each step using filtered probabilities."""
        probs = self.predict_filtered_proba(X)
        return np.argmax(probs, axis=1)

    @property
    def transition_matrix_(self) -> pd.DataFrame:
        """Return regime transition matrix as a labeled DataFrame."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        cols = [f"Regime_{i}" for i in range(self.n_components)]
        return pd.DataFrame(self.model.transmat_, index=cols, columns=cols)
