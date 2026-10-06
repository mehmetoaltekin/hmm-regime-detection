"""
Gaussian HMM regime detection with strictly causal inference.

* ``MarketRegimeHMM``  - one calibration: standardised inputs, multi-start EM,
  states sorted by variance, forward filtering with an optional prior state.
* ``WalkForwardHMM``   - rolling re-calibration. Every parameter (scaler, HMM
  parameters, number of states K) is estimated only on data strictly before
  the bars it is applied to.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

SUPPORTED_COVARIANCE_TYPES = ("full", "diag")
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_2d(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or inf; drop or fill them first.")
    return X


class MarketRegimeHMM:
    """
    Gaussian HMM for market regime detection.

    Parameters
    ----------
    n_components : int
        Number of hidden states K.
    covariance_type : {"full", "diag"}
        "tied" is not supported (all states share one covariance, so there is
        no variance to sort regimes by). "spherical" is not supported because
        hmmlearn 0.3.x stores it with an inconsistent shape.
    n_iter : int
        Maximum EM iterations per start.
    n_init : int
        Number of EM restarts with different seeds. The run with the highest
        log-likelihood is kept, which reduces the risk of a poor local optimum.
    scale : bool
        Standardise each feature with the mean / std of the *training* data.
        Raw features live on very different scales (returns ~0.01, annualised
        volatility ~0.2), which makes full covariances near-singular.
    min_covar : float
        Floor added to covariance diagonals (in standardised units).
    random_state : int
        Base seed; restart i uses ``random_state + i``.

    Notes
    -----
    States are reordered by total variance, so state 0 is always the
    calmest regime. Inference uses the forward filter P(S_t | Y_1:t) only.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 200,
        n_init: int = 10,
        scale: bool = True,
        min_covar: float = 1e-3,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}, "
                f"got {covariance_type!r}"
            )
        if n_init < 1:
            raise ValueError("n_init must be >= 1")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_init = n_init
        self.scale = scale
        self.min_covar = min_covar
        self.tol = tol
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.order: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------ #
    # scaling
    # ------------------------------------------------------------------ #
    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self.scale:
            return X
        return (X - self.mean_) / self.scale_

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def _fit_single(self, Xs: np.ndarray, seed: int) -> GaussianHMM:
        hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            random_state=seed,
        )
        # Individual starts may wobble; the best start is chosen by its final
        # likelihood, so per-start convergence chatter is muted here.
        hmm_logger = logging.getLogger("hmmlearn.base")
        old_level = hmm_logger.level
        hmm_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm.fit(Xs)
        finally:
            hmm_logger.setLevel(old_level)
        return hmm

    def fit(self, X: ArrayLike) -> "MarketRegimeHMM":
        X = _as_2d(X)
        n_samples, n_features = X.shape
        if n_samples <= self.n_components:
            raise ValueError("Not enough samples for the requested number of states.")

        self.n_features_ = n_features
        if self.scale:
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0)
            self.scale_ = np.where(std > 0, std, 1.0)
        Xs = self._transform(X)

        best, best_ll, errors = None, -np.inf, []
        for i in range(self.n_init):
            seed = self.random_state + i
            try:
                hmm = self._fit_single(Xs, seed)
                ll = float(hmm.score(Xs))
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append(f"seed {seed}: {exc}")
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = hmm, ll

        if best is None:
            raise RuntimeError(
                f"All {self.n_init} EM starts failed. Last errors: {errors[-3:]}"
            )
        if errors:
            logger.debug("%d/%d EM starts failed: %s", len(errors), self.n_init, errors)

        self._sort_states(best)
        self.model = best
        self.log_likelihood_ = best_ll
        return self

    def _sort_states(self, hmm: GaussianHMM) -> None:
        """Reorder states by total variance (state 0 = calmest)."""
        # ``covars_`` always returns full (K, d, d) matrices in hmmlearn,
        # whatever the covariance type, so the trace works for every type.
        variances = np.array([np.trace(c) for c in hmm.covars_])
        order = np.argsort(variances, kind="stable")
        self.order = order

        hmm.startprob_ = hmm.startprob_[order]
        hmm.transmat_ = hmm.transmat_[order][:, order]
        hmm.means_ = hmm.means_[order]
        # Reorder the *internal* storage, whose shape matches the covariance
        # type: (K, d) diag, (K, d, d) full. Assigning the
        # full matrices returned by ``covars_`` back to a diag model fails.
        hmm._covars_ = hmm._covars_[order]

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

    def predict_filtered_proba(
        self,
        X: ArrayLike,
        prior_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward filter: P(S_t | Y_1:t), computed step by step without
        future information.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        prior_proba : array of shape (n_components,), optional
            Filtered state distribution at the bar *before* X[0]. If given,
            the first step is propagated through the transition matrix
            instead of restarting from ``startprob_``. This keeps the filter
            continuous across walk-forward refits.
        """
        self._check_fitted()
        X = _as_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        n_samples = len(X)
        K = self.n_components

        with np.errstate(divide="ignore"):
            log_trans = np.log(self.model.transmat_)
            if prior_proba is None:
                log_prior = np.log(self.model.startprob_)
            else:
                p = np.asarray(prior_proba, dtype=float)
                if p.shape != (K,) or not np.isclose(p.sum(), 1.0):
                    raise ValueError("prior_proba must be a length-K probability vector.")
                log_prior = np.logaddexp.reduce(np.log(p)[:, None] + log_trans, axis=0)

        frame_log_prob = self.model._compute_log_likelihood(self._transform(X))
        filtered = np.zeros((n_samples, K))

        curr = log_prior + frame_log_prob[0]
        curr -= np.logaddexp.reduce(curr)
        filtered[0] = np.exp(curr)

        for t in range(1, n_samples):
            prior = np.logaddexp.reduce(curr[:, None] + log_trans, axis=0)
            curr = prior + frame_log_prob[t]
            curr -= np.logaddexp.reduce(curr)
            filtered[t] = np.exp(curr)

        return filtered

    def predict_regimes(self, X: ArrayLike, prior_proba: Optional[np.ndarray] = None) -> np.ndarray:
        return np.argmax(self.predict_filtered_proba(X, prior_proba), axis=1)

    def score(self, X: ArrayLike) -> float:
        """Log-likelihood of X in the *scaled* space (comparable across K)."""
        self._check_fitted()
        return float(self.model.score(self._transform(_as_2d(X))))

    @property
    def transition_matrix_(self) -> pd.DataFrame:
        self._check_fitted()
        cols = [f"regime_{i}" for i in range(self.n_components)]
        return pd.DataFrame(self.model.transmat_, index=cols, columns=cols)

    @property
    def means_(self) -> np.ndarray:
        """State means in the original (unscaled) feature units."""
        self._check_fitted()
        m = self.model.means_
        return m * self.scale_ + self.mean_ if self.scale else m

    @property
    def covars_(self) -> np.ndarray:
        """State covariances (K, d, d) in the original feature units."""
        self._check_fitted()
        c = self.model.covars_
        if not self.scale:
            return c
        S = np.diag(self.scale_)
        return np.array([S @ ci @ S for ci in c])


class WalkForwardHMM:
    """
    Walk-forward (rolling or expanding window) Gaussian HMM.

    At each refit time t (every ``step_size`` bars after the first
    ``window_size`` bars):

    1. The training window ends at bar t-1.
    2. If ``n_components`` is a list, K is chosen by AIC/BIC on that window
       only - never on the full sample.
    3. The model is fitted (scaler + multi-start EM) on that window.
    4. The forward filter is run through the training window to obtain the
       state belief at t-1, which is used as the prior for bar t. There is
       no reset to ``startprob_`` at the refit boundary.
    5. Filtered probabilities are produced for bars t ... t+step_size-1.

    Output rows before the first refit are NaN (no model exists yet).
    """

    def __init__(
        self,
        n_components: Union[int, Sequence[int]] = 2,
        window_size: int = 252,
        step_size: int = 21,
        expanding: bool = False,
        covariance_type: str = "full",
        criterion: str = "bic",
        n_iter: int = 200,
        n_init: int = 10,
        scale: bool = True,
        min_covar: float = 1e-3,
        random_state: int = 42,
    ):
        if criterion not in ("aic", "bic"):
            raise ValueError("criterion must be 'aic' or 'bic'")
        if step_size < 1 or window_size < 2:
            raise ValueError("window_size must be >= 2 and step_size >= 1")

        self.candidates = (
            [int(n_components)] if np.isscalar(n_components) else [int(k) for k in n_components]
        )
        self.window_size = window_size
        self.step_size = step_size
        self.expanding = expanding
        self.covariance_type = covariance_type
        self.criterion = criterion
        self.model_kwargs = dict(
            covariance_type=covariance_type,
            n_iter=n_iter,
            n_init=n_init,
            scale=scale,
            min_covar=min_covar,
            random_state=random_state,
        )
        self.max_k = max(self.candidates)
        self.history_: list = []

    def _fit_window(self, X_train: np.ndarray) -> MarketRegimeHMM:
        # local import avoids a circular import
        from src.model_selection import select_n_components

        if len(self.candidates) == 1:
            k = self.candidates[0]
        else:
            k = select_n_components(
                X_train, self.candidates, criterion=self.criterion, **self.model_kwargs
            )
        return MarketRegimeHMM(n_components=k, **self.model_kwargs).fit(X_train)

    def fit_predict_filtered(self, X: ArrayLike) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
            Feature matrix, e.g. the ``feature_df`` from ``prepare_hmm_features``
            or the output of ``fetch_returns``. A DataFrame index is kept.

        Returns
        -------
        DataFrame with columns ``prob_regime_0 .. prob_regime_{maxK-1}``,
        ``regime`` (argmax, state 0 = calmest), ``n_states`` (K used) and
        ``refit`` (True on bars where a new model was calibrated).
        """
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number])
        X = _as_2d(X)
        n = len(X)
        if n <= self.window_size:
            raise ValueError(f"Need more than window_size={self.window_size} rows, got {n}.")

        probs = np.full((n, self.max_k), np.nan)
        n_states = np.full(n, np.nan)
        refit = np.zeros(n, dtype=bool)
        self.history_ = []

        for t in range(self.window_size, n, self.step_size):
            start = 0 if self.expanding else t - self.window_size
            X_train = X[start:t]
            model = self._fit_window(X_train)
            k = model.n_components

            # belief at t-1 from the new model, using data up to t-1 only
            belief = model.predict_filtered_proba(X_train)[-1]

            end = min(t + self.step_size, n)
            p = model.predict_filtered_proba(X[t:end], prior_proba=belief)

            probs[t:end, :k] = p
            if k < self.max_k:
                probs[t:end, k:] = 0.0
            n_states[t:end] = k
            refit[t] = True
            self.history_.append(
                {
                    "refit_pos": t,
                    "refit_time": index[t] if index is not None else t,
                    "train_start": start,
                    "n_states": k,
                    "log_likelihood": model.log_likelihood_,
                    "model": model,
                }
            )

        cols = [f"prob_regime_{i}" for i in range(self.max_k)]
        out = pd.DataFrame(probs, columns=cols, index=index)
        regime = np.full(n, np.nan)
        valid = ~np.isnan(probs[:, 0])
        regime[valid] = np.nanargmax(probs[valid], axis=1)
        out["regime"] = pd.Series(regime, index=out.index).astype("Int64")
        out["n_states"] = n_states
        out["refit"] = refit
        return out
