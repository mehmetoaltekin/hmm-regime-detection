"""
Model selection criteria for Gaussian Hidden Markov Models.

Computes AIC and BIC across candidate regime counts. To avoid look-ahead,
call these only on training data (``WalkForwardHMM`` does this per window).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.hmm_model import MarketRegimeHMM, _as_2d


def count_free_parameters(
    n_components: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """
    Number of free parameters in a Gaussian HMM.

    (K - 1) start probs + K(K - 1) transitions + K*d means + covariances,
    where covariances are K*d(d+1)/2 (full), K*d (diag) or K (spherical).
    """
    init_params = n_components - 1
    trans_params = n_components * (n_components - 1)
    mean_params = n_components * n_features

    if covariance_type == "full":
        cov_params = n_components * (n_features * (n_features + 1)) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "spherical":
        cov_params = n_components
    elif covariance_type == "tied":
        cov_params = (n_features * (n_features + 1)) // 2
    else:
        raise ValueError(f"Unknown covariance_type {covariance_type!r}")

    return init_params + trans_params + mean_params + cov_params


def evaluate_regime_models(
    X: np.ndarray,
    components_range: Optional[Sequence[int]] = None,
    covariance_type: str = "full",
    n_iter: int = 200,
    n_init: int = 10,
    scale: bool = True,
    min_covar: float = 1e-3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit HMMs for each candidate K and score AIC / BIC.

        AIC = -2 ln(L) + 2p
        BIC = -2 ln(L) + p ln(N)

    Likelihoods are computed on standardised features when ``scale=True``.
    The scaling Jacobian is the same for every K, so rankings are unaffected.
    A K whose fit fails on every EM start is reported with NaN scores
    instead of crashing the whole comparison.

    Returns
    -------
    DataFrame sorted by BIC (best first).
    """
    if components_range is None:
        components_range = [2, 3, 4]

    X = _as_2d(X)
    n_samples, n_features = X.shape
    results: List[Dict[str, float]] = []

    for k in components_range:
        p = count_free_parameters(k, n_features, covariance_type)
        try:
            model = MarketRegimeHMM(
                n_components=k,
                covariance_type=covariance_type,
                n_iter=n_iter,
                n_init=n_init,
                scale=scale,
                min_covar=min_covar,
                random_state=random_state,
            ).fit(X)
            ll = model.log_likelihood_
            aic = -2.0 * ll + 2.0 * p
            bic = -2.0 * ll + p * np.log(n_samples)
        except (RuntimeError, ValueError):
            ll = aic = bic = np.nan

        results.append({
            "States (K)": k,
            "Parameters (p)": p,
            "Log-Likelihood": round(ll, 2),
            "AIC": round(aic, 2),
            "BIC": round(bic, 2),
        })

    return (
        pd.DataFrame(results)
        .sort_values("BIC", na_position="last")
        .reset_index(drop=True)
    )


def select_n_components(
    X: np.ndarray,
    components_range: Sequence[int] = (2, 3, 4),
    criterion: str = "bic",
    **model_kwargs,
) -> int:
    """Return the K with the lowest AIC/BIC on X (use training data only)."""
    col = {"aic": "AIC", "bic": "BIC"}[criterion]
    table = evaluate_regime_models(X, components_range, **model_kwargs)
    table = table.dropna(subset=[col])
    if table.empty:
        raise RuntimeError("No candidate K could be fitted.")
    return int(table.sort_values(col).iloc[0]["States (K)"])
