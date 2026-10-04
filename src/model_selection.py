"""
Model selection criteria for Gaussian Hidden Markov Models.
Computes AIC and BIC across multiple regime counts to determine optimal state space.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def count_free_parameters(
    n_components: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """
    Calculate the exact number of free parameters in a Gaussian HMM.

    Parameters
    ----------
    n_components : int
        Number of hidden states (K).
    n_features : int
        Number of observed dimensions (d).
    covariance_type : str, default "full"
        Type of covariance matrix ('full', 'diag').

    Returns
    -------
    int
        Total number of independent free parameters.
    """
    # Initial state probabilities sum to 1: K - 1 degrees of freedom
    init_params = n_components - 1

    # Transition probability matrix rows sum to 1: K * (K - 1)
    trans_params = n_components * (n_components - 1)

    # Emission means for each state: K * d
    mean_params = n_components * n_features

    # Emission covariances
    if covariance_type == "full":
        # Symmetric positive-definite matrix: d * (d + 1) / 2 per state
        cov_params = n_components * (n_features * (n_features + 1)) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    else:
        cov_params = n_components

    return init_params + trans_params + mean_params + cov_params


def evaluate_regime_models(
    X: np.ndarray,
    components_range: Optional[List[int]] = None,
    covariance_type: str = "full",
    n_iter: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit HMMs across different candidate state counts and score AIC / BIC.

    Formulas:
        AIC = -2 * ln(L) + 2 * p
        BIC = -2 * ln(L) + p * ln(N)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    components_range : Optional[List[int]], default None
        List of hidden state counts to benchmark. Defaults to [2, 3, 4].
    covariance_type : str, default "full"
        Covariance structure.
    n_iter : int, default 100
        Maximum iterations for EM algorithm.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Ranked summary table containing Log-Likelihood, AIC, and BIC.
    """
    if components_range is None:
        components_range = [2, 3, 4]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    results: List[Dict[str, float]] = []

    for k in components_range:
        model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X)

        log_likelihood = float(model.score(X))
        p = count_free_parameters(k, n_features, covariance_type)

        aic = -2.0 * log_likelihood + 2.0 * p
        bic = -2.0 * log_likelihood + p * np.log(n_samples)

        results.append({
            "States (K)": k,
            "Parameters (p)": p,
            "Log-Likelihood": round(log_likelihood, 2),
            "AIC": round(aic, 2),
            "BIC": round(bic, 2),
        })

    summary_df = pd.DataFrame(results).sort_values("BIC").reset_index(drop=True)
    return summary_df
