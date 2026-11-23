"""
Model selection criteria for Gaussian Hidden Markov Models.

Computes AIC and BIC across candidate regime counts. To avoid look-ahead,
call these only on training data (``WalkForwardHMM`` does this per window).
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hmm_regime.hmm_model import MarketRegimeHMM, _as_2d

logger = logging.getLogger(__name__)


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


def _fit_candidates(
    X: np.ndarray,
    components_range: Sequence[int],
    warm_start: Optional[MarketRegimeHMM] = None,
    **model_kwargs,
) -> List[Tuple[int, Optional[MarketRegimeHMM]]]:
    """Fit one model per candidate K; a K whose fit fails gives ``None``."""
    fitted = []
    for k in components_range:
        try:
            model = MarketRegimeHMM(n_components=int(k), **model_kwargs)
            # every candidate is tried, so only the selected one's problems
            # are worth a log line: the caller reports them
            model._fit(X, warm_start=warm_start)
        except (RuntimeError, ValueError):
            model = None
        fitted.append((int(k), model))
    return fitted


def _table(fitted, n_features: int, covariance_type: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for k, model in fitted:
        row = {
            "States (K)": k,
            "Parameters (p)": count_free_parameters(k, n_features, covariance_type),
            "Log-Likelihood": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "Converged": False,
            "Min state obs": np.nan,
        }
        if model is not None:
            row.update({
                "Log-Likelihood": model.log_likelihood_,
                "AIC": model.aic_,
                "BIC": model.bic_,
                "Converged": model.converged_,
                "Min state obs": float(model.state_occupancy_.min()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_regime_models(
    X: np.ndarray,
    components_range: Optional[Sequence[int]] = None,
    covariance_type: str = "full",
    n_iter: int = 200,
    n_init: int = 10,
    scale: bool = True,
    min_covar: float = 1e-3,
    random_state: int = 42,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Fit HMMs for each candidate K and score AIC / BIC.

        AIC = -2 ln(L) + 2p
        BIC = -2 ln(L) + p ln(N)

    Likelihoods are computed on standardised features when ``scale=True``.
    The scaling Jacobian is the same for every K, so rankings are unaffected.
    A K whose fit fails on every EM start is reported with NaN scores
    instead of crashing the whole comparison. Extra keyword arguments
    (``tol``, ``sort_by``, ``covar_prior``) go to ``MarketRegimeHMM``.

    Values are not rounded (round when printing). ``Converged`` is the EM
    flag of the kept start and ``Min state obs`` the expected number of
    training bars in the rarest state: a K whose rarest state holds only a
    handful of bars is fitting noise, whatever its BIC says.

    Returns
    -------
    DataFrame sorted by BIC (best first).
    """
    if components_range is None:
        components_range = [2, 3, 4]

    X = _as_2d(X)
    fitted = _fit_candidates(
        X, components_range,
        covariance_type=covariance_type, n_iter=n_iter, n_init=n_init, scale=scale,
        min_covar=min_covar, random_state=random_state, **model_kwargs,
    )
    return (
        _table(fitted, X.shape[1], covariance_type)
        .sort_values("BIC", na_position="last", kind="stable")
        .reset_index(drop=True)
    )


def select_model(
    X: np.ndarray,
    components_range: Sequence[int] = (2, 3, 4),
    criterion: str = "bic",
    warm_start: Optional[MarketRegimeHMM] = None,
    _quiet: bool = False,
    **model_kwargs,
) -> MarketRegimeHMM:
    """
    Fit every candidate K on X and return the fitted model with the lowest
    AIC/BIC (use training data only). Ties go to the smaller K.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    X = _as_2d(X)
    fitted = _fit_candidates(X, sorted(set(int(k) for k in components_range)),
                             warm_start=warm_start, **model_kwargs)
    best, best_score = None, np.inf
    for _, model in fitted:
        if model is None:
            continue
        score = model.bic_ if criterion == "bic" else model.aic_
        if np.isfinite(score) and score < best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError("No candidate K could be fitted.")
    if not _quiet:
        for message in best.warnings_:
            logger.warning(message)
    return best


def select_n_components(
    X: np.ndarray,
    components_range: Sequence[int] = (2, 3, 4),
    criterion: str = "bic",
    **model_kwargs,
) -> int:
    """Return the K with the lowest AIC/BIC on X (use training data only)."""
    return select_model(X, components_range, criterion=criterion, **model_kwargs).n_components
