"""
Gaussian HMM regime detection with strictly causal inference.

* ``MarketRegimeHMM``  - one calibration: standardised inputs, multi-start EM,
  states sorted by volatility, forward filtering with an optional prior state.
* ``WalkForwardHMM``   - rolling re-calibration. Every parameter (scaler, HMM
  parameters, number of states K) is estimated only on data strictly before
  the bars it is applied to.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple, Union

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
    if X.ndim != 2:
        raise ValueError("Input must be 1D or 2D (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or inf; drop or fill them first.")
    return X


def _parse_sort_by(sort_by: str) -> Tuple[str, Optional[int]]:
    """'trace' -> ('trace', None); 'variance:0' -> ('variance', 0); 'mean:1' -> ('mean', 1)."""
    if sort_by == "trace":
        return "trace", None
    stat, sep, col = str(sort_by).partition(":")
    if stat in ("variance", "mean") and sep and col.isdigit():
        return stat, int(col)
    raise ValueError(
        f"sort_by must be 'trace', 'variance:<feature index>' or "
        f"'mean:<feature index>', got {sort_by!r}"
    )


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """
    Stationary distribution pi of a row-stochastic matrix: pi @ A = pi.

    Used as the state distribution before the first bar of a new sequence.
    If the chain has no unique stationary distribution (a block of states
    that is never left), the long-run distribution from a uniform start is
    returned.
    """
    A = np.asarray(transmat, dtype=float)
    K = len(A)
    M = np.vstack([A.T - np.eye(K), np.ones((1, K))])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, _, rank, _ = np.linalg.lstsq(M, b, rcond=None)
    if rank < K or not np.all(np.isfinite(pi)) or pi.min() < -1e-9:
        # (A + I) / 2 has the same stationary distributions and no cycles,
        # so its powers converge; squaring 64 times is the 2^64-th power.
        P = (A + np.eye(K)) / 2.0
        for _ in range(64):
            P = P @ P
            P /= P.sum(axis=1, keepdims=True)
        pi = P.mean(axis=0)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


@contextmanager
def _quiet_hmmlearn():
    """Mute hmmlearn's per-start convergence chatter; the best start is
    chosen by its final likelihood and its convergence is reported by us."""
    hmm_logger = logging.getLogger("hmmlearn")
    old_level = hmm_logger.level
    hmm_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        hmm_logger.setLevel(old_level)


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
        Added to the covariance diagonal when EM is *initialised* (hmmlearn
        does not apply it afterwards). The regulariser during EM is
        ``covar_prior``.
    tol : float
        EM convergence tolerance on the log-likelihood gain.
    random_state : int
        Base seed; restart i uses ``random_state + i``.
    sort_by : str
        What "state 0 = calmest" means.

        * ``"variance:0"`` (default) - spread of feature 0 in the state,
          measured around the training-window mean (state variance plus the
          squared distance of the state mean from the window mean). Put the
          return in column 0 (``prepare_hmm_features`` does); state 0 is
          then the state with the lowest return volatility. Measuring
          around the window mean matters for a state that holds a few
          outlier bars: its own variance is tiny, its returns are not.
        * ``"mean:j"`` - mean of feature j, e.g. a volatility feature.
        * ``"trace"`` - total variance of all standardised features (the rule
          used before version 0.2).
    covar_prior : float
        Ridge added to the diagonal of every state covariance in each M-step
        (standardised units), weighted like one observation. It keeps a state
        that explains only a few bars from collapsing to a singular
        covariance, where the likelihood is unbounded.

    Attributes
    ----------
    converged_, n_iter_ : EM convergence flag and iterations of the kept start.
    log_likelihood_, n_parameters_, aic_, bic_ : fit statistics on the training data.
    state_occupancy_ : expected number of training bars in each state.
    stress_states_ : boolean mask of the states whose ``sort_by`` score is
        above the occupancy-weighted average score of the training window.
    stationary_ : stationary distribution of the transition matrix.
    warnings_ : list of str, problems of the fit (EM not converged, a state
        that holds almost no bars). ``fit`` also logs them.

    Notes
    -----
    Inference uses the forward filter P(S_t | Y_1:t) only.
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
        sort_by: str = "variance:0",
        covar_prior: float = 1e-2,
    ):
        if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}, "
                f"got {covariance_type!r}"
            )
        if int(n_components) != n_components or n_components < 1:
            raise ValueError("n_components must be an integer >= 1")
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        if covar_prior < 0:
            raise ValueError("covar_prior must be >= 0")
        _parse_sort_by(sort_by)

        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_init = n_init
        self.scale = scale
        self.min_covar = min_covar
        self.tol = tol
        self.random_state = random_state
        self.sort_by = sort_by
        self.covar_prior = covar_prior

        self.model: Optional[GaussianHMM] = None
        self.order: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.converged_: Optional[bool] = None
        self.n_iter_: Optional[int] = None
        self.state_occupancy_: Optional[np.ndarray] = None
        self.stress_states_: Optional[np.ndarray] = None
        self.stationary_: Optional[np.ndarray] = None
        self.warnings_: list = []
        self._center: Optional[np.ndarray] = None

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
    def _new_hmm(self, seed: int, init_params: str = "stmc") -> GaussianHMM:
        d = self.n_features_
        # A scalar prior is added by hmmlearn to *every* element of a full
        # covariance (off-diagonals too), which does not bound the smallest
        # eigenvalue. A diagonal matrix does.
        prior = self.covar_prior * np.eye(d) if self.covariance_type == "full" else self.covar_prior
        return GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            covars_prior=prior,
            random_state=seed,
            init_params=init_params,
        )

    def _fit_single(self, Xs: np.ndarray, seed: int) -> GaussianHMM:
        hmm = self._new_hmm(seed)
        with _quiet_hmmlearn():
            hmm.fit(Xs)
        return hmm

    def _fit_warm(self, Xs: np.ndarray, other: "MarketRegimeHMM") -> GaussianHMM:
        """EM started from another model's parameters, re-expressed in this
        model's standardised units."""
        hmm = self._new_hmm(self.random_state, init_params="")
        K = self.n_components
        shift = self.mean_ if self.scale else 0.0
        div = self.scale_ if self.scale else np.ones(self.n_features_)
        means = (other.means_ - shift) / div
        covs = other.covars_ / np.outer(div, div)
        hmm.n_features = self.n_features_
        hmm.startprob_ = np.full(K, 1.0 / K)
        trans = other.model.transmat_ + 1e-6  # EM cannot revive an exact zero
        hmm.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        hmm.means_ = means
        if self.covariance_type == "diag":
            hmm.covars_ = np.array([np.diag(c) for c in covs])
        else:
            hmm.covars_ = np.array([(c + c.T) / 2.0 for c in covs])
        with _quiet_hmmlearn():
            hmm.fit(Xs)
        return hmm

    def fit(self, X: ArrayLike, warm_start: Optional["MarketRegimeHMM"] = None) -> "MarketRegimeHMM":
        """
        Calibrate on X (training data only).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        warm_start : MarketRegimeHMM, optional
            A fitted model with the same K, features and covariance type,
            e.g. the previous walk-forward window. Its parameters are used
            as one more EM start next to the ``n_init`` random ones; the
            start with the highest log-likelihood is kept as usual.
        """
        self._fit(X, warm_start=warm_start)
        for message in self.warnings_:
            logger.warning(message)
        return self

    def _fit(self, X: ArrayLike, warm_start: Optional["MarketRegimeHMM"] = None) -> "MarketRegimeHMM":
        """``fit`` without logging; problems are collected in ``warnings_``."""
        X = _as_2d(X)
        n_samples, n_features = X.shape
        # one mean + one covariance per state needs more than d + 1 bars each
        min_samples = self.n_components * (n_features + 2)
        if n_samples < min_samples:
            raise ValueError(
                f"Not enough samples for the requested number of states: "
                f"got {n_samples}, need at least {min_samples}."
            )

        self.n_features_ = n_features
        self.n_samples_ = n_samples
        stat, col = _parse_sort_by(self.sort_by)
        if col is not None and col >= n_features:
            raise ValueError(f"sort_by={self.sort_by!r} but X has {n_features} features.")
        self._center = X.mean(axis=0)
        if self.scale:
            self.mean_ = self._center
            std = X.std(axis=0)
            self.scale_ = np.where(std > 0, std, 1.0)
        Xs = self._transform(X)

        best, best_ll, errors = None, -np.inf, []

        def consider(make, label):
            nonlocal best, best_ll
            try:
                hmm = make()
                ll = float(hmm.score(Xs))
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append(f"{label}: {exc}")
                return
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = hmm, ll

        for i in range(self.n_init):
            seed = self.random_state + i
            consider(lambda: self._fit_single(Xs, seed), f"seed {seed}")
        if (
            warm_start is not None
            and warm_start.model is not None
            and warm_start.n_components == self.n_components
            and warm_start.n_features_ == n_features
            and warm_start.covariance_type == self.covariance_type
        ):
            consider(lambda: self._fit_warm(Xs, warm_start), "warm start")

        if best is None:
            raise RuntimeError(
                f"All {self.n_init} EM starts failed. Last errors: {errors[-3:]}"
            )
        if errors:
            logger.debug("%d EM starts failed: %s", len(errors), errors)

        self._sort_states(best)
        self.model = best
        self.log_likelihood_ = best_ll
        self.converged_ = bool(best.monitor_.converged)
        self.n_iter_ = int(best.monitor_.iter)
        self.stationary_ = stationary_distribution(best.transmat_)

        # diagnostics on the training window (smoothed posteriors are fine
        # here: they describe the fit, they are never used as a signal)
        with _quiet_hmmlearn():
            occupancy = best.predict_proba(Xs).sum(axis=0)
        self.state_occupancy_ = occupancy
        scores = self._state_scores(best)
        average = float(np.dot(occupancy / occupancy.sum(), scores))
        self.stress_states_ = scores > average + 1e-12 * max(1.0, abs(average))

        self.warnings_ = []
        if not self.converged_:
            self.warnings_.append(
                f"EM did not converge in {self.n_iter} iterations "
                f"(K={self.n_components}); raise n_iter or tol."
            )
        if occupancy.min() < n_features + 1:
            self.warnings_.append(
                f"A state holds only {occupancy.min():.1f} expected training bars "
                f"(K={self.n_components}): its covariance rests on the prior. "
                f"Consider fewer states (K=1 can be a candidate) or a longer window."
            )
        return self

    def _state_scores(self, hmm: GaussianHMM) -> np.ndarray:
        """Per-state volatility score in standardised units (see ``sort_by``)."""
        stat, col = _parse_sort_by(self.sort_by)
        # ``covars_`` always returns full (K, d, d) matrices in hmmlearn,
        # whatever the covariance type.
        if stat == "trace":
            return np.array([np.trace(c) for c in hmm.covars_])
        if stat == "variance":
            # around the training-window mean, which is 0 after standardising
            centre = 0.0 if self.scale else self._center[col]
            return np.array([c[col, col] for c in hmm.covars_]) + (hmm.means_[:, col] - centre) ** 2
        return np.asarray(hmm.means_[:, col], dtype=float)

    def _sort_states(self, hmm: GaussianHMM) -> None:
        """Reorder states by the ``sort_by`` score (state 0 = calmest)."""
        order = np.argsort(self._state_scores(hmm), kind="stable")
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
        initial: str = "stationary",
    ) -> np.ndarray:
        """
        Forward filter: P(S_t | Y_1:t), computed step by step without
        future information.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        prior_proba : array of shape (n_components,), optional
            Filtered state distribution at the bar *before* X[0]. If given,
            the first step is propagated through the transition matrix.
            This keeps the filter continuous across walk-forward refits.
        initial : {"stationary", "startprob"}
            State distribution for X[0] when ``prior_proba`` is not given.
            "stationary" (default) is the long-run distribution of the
            transition matrix, the right choice for any sequence. EM
            estimates ``startprob_`` from the single first training bar, so
            it puts all its mass on one state; "startprob" therefore fixes
            the state of X[0] whatever X[0] is, and is only meaningful when
            X is the training window itself.
        """
        self._check_fitted()
        X = _as_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        if initial not in ("stationary", "startprob"):
            raise ValueError("initial must be 'stationary' or 'startprob'")
        n_samples = len(X)
        K = self.n_components
        if n_samples == 0:
            return np.zeros((0, K))

        with np.errstate(divide="ignore"):
            log_trans = np.log(self.model.transmat_)
            if prior_proba is None:
                start = self.stationary_ if initial == "stationary" else self.model.startprob_
                log_prior = np.log(start)
            else:
                p = np.asarray(prior_proba, dtype=float)
                if (
                    p.shape != (K,)
                    or not np.all(np.isfinite(p))
                    or p.min() < 0
                    or not np.isclose(p.sum(), 1.0)
                ):
                    raise ValueError("prior_proba must be a length-K probability vector.")
                log_prior = np.logaddexp.reduce(np.log(p)[:, None] + log_trans, axis=0)

        with np.errstate(over="ignore", invalid="ignore"):
            frame_log_prob = self.model._compute_log_likelihood(self._transform(X))
        # A bar that no state can explain numerically (densities underflow to
        # zero in every state) carries no usable evidence: keep the predicted
        # distribution instead of turning the whole filter into NaN.
        unusable = ~np.isfinite(frame_log_prob).any(axis=1) | np.isnan(frame_log_prob).any(axis=1)
        if unusable.any():
            frame_log_prob = frame_log_prob.copy()
            frame_log_prob[unusable] = 0.0

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

    def stress_proba(self, filtered: np.ndarray) -> np.ndarray:
        """
        P(stress) per bar from filtered probabilities of this model: the
        summed probability of ``stress_states_``, the states that are more
        volatile than the training window's average. Unlike a state number
        this keeps its meaning when K changes between walk-forward refits.
        """
        self._check_fitted()
        P = np.asarray(filtered, dtype=float)
        return P[:, : self.n_components][:, self.stress_states_].sum(axis=1)

    def score(self, X: ArrayLike) -> float:
        """Log-likelihood of X in the *scaled* space (comparable across K)."""
        self._check_fitted()
        return float(self.model.score(self._transform(_as_2d(X))))

    @property
    def n_parameters_(self) -> int:
        """Free parameters: start probabilities, transitions, means, covariances."""
        self._check_fitted()
        K, d = self.n_components, self.n_features_
        cov = K * d * (d + 1) // 2 if self.covariance_type == "full" else K * d
        return (K - 1) + K * (K - 1) + K * d + cov

    @property
    def aic_(self) -> float:
        return -2.0 * self.log_likelihood_ + 2.0 * self.n_parameters_

    @property
    def bic_(self) -> float:
        return -2.0 * self.log_likelihood_ + self.n_parameters_ * np.log(self.n_samples_)

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
       no reset at the refit boundary.
    5. Filtered probabilities are produced for bars t ... t+step_size-1.

    Output rows before the first refit are NaN (no model exists yet).

    Parameters are those of ``MarketRegimeHMM`` plus:

    n_components : int or list of int
        K, or the candidates for the per-window AIC/BIC choice.
    window_size, step_size : int
        Training window length and bars between refits.
    expanding : bool
        Train on all bars since the start instead of the last ``window_size``.
    criterion : {"bic", "aic"}
    warm_start : bool
        Also start EM from the previous window's model when K is unchanged
        (one more start next to the ``n_init`` random ones). Uses past data
        only. Off by default.
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
        tol: float = 1e-4,
        sort_by: str = "variance:0",
        covar_prior: float = 1e-2,
        warm_start: bool = False,
    ):
        if criterion not in ("aic", "bic"):
            raise ValueError("criterion must be 'aic' or 'bic'")
        if step_size < 1 or window_size < 2:
            raise ValueError("window_size must be >= 2 and step_size >= 1")
        if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}, "
                f"got {covariance_type!r}"
            )
        _parse_sort_by(sort_by)

        self.candidates = (
            [int(n_components)] if np.isscalar(n_components) else [int(k) for k in n_components]
        )
        if not self.candidates or min(self.candidates) < 1:
            raise ValueError("n_components must be >= 1 (or a non-empty list of such values)")
        self.window_size = window_size
        self.step_size = step_size
        self.expanding = expanding
        self.covariance_type = covariance_type
        self.criterion = criterion
        self.warm_start = warm_start
        self.model_kwargs = dict(
            covariance_type=covariance_type,
            n_iter=n_iter,
            n_init=n_init,
            scale=scale,
            min_covar=min_covar,
            random_state=random_state,
            tol=tol,
            sort_by=sort_by,
            covar_prior=covar_prior,
        )
        self.max_k = max(self.candidates)
        self.history_: list = []

    def _fit_window(
        self, X_train: np.ndarray, previous: Optional[MarketRegimeHMM] = None
    ) -> MarketRegimeHMM:
        # local import avoids a circular import
        from hmm_regime.model_selection import select_model

        warm = previous if self.warm_start else None
        if len(self.candidates) == 1:
            return MarketRegimeHMM(
                n_components=self.candidates[0], **self.model_kwargs
            )._fit(X_train, warm_start=warm)
        # the selected model is returned as fitted: no second calibration
        return select_model(
            X_train, self.candidates, criterion=self.criterion,
            warm_start=warm, _quiet=True, **self.model_kwargs,
        )

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
        ``regime`` (argmax, state 0 = calmest), ``n_states`` (K used),
        ``refit`` (True on bars where a new model was calibrated) and
        ``prob_stress``: the summed probability of the states that are more
        volatile than their training window's average
        (``MarketRegimeHMM.stress_states_``). With K = 2 it equals
        ``prob_regime_1``; with K >= 3 it keeps one meaning across refits,
        which ``prob_regime_0`` does not.
        """
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number])
        X = _as_2d(X)
        n = len(X)
        if n <= self.window_size:
            raise ValueError(f"Need more than window_size={self.window_size} rows, got {n}.")

        probs = np.full((n, self.max_k), np.nan)
        stress = np.full(n, np.nan)
        n_states = np.full(n, np.nan)
        refit = np.zeros(n, dtype=bool)
        self.history_ = []
        model: Optional[MarketRegimeHMM] = None

        for t in range(self.window_size, n, self.step_size):
            start = 0 if self.expanding else t - self.window_size
            X_train = X[start:t]
            model = self._fit_window(X_train, previous=model)
            k = model.n_components

            # belief at t-1 from the new model, using data up to t-1 only
            belief = model.predict_filtered_proba(X_train)[-1]

            end = min(t + self.step_size, n)
            p = model.predict_filtered_proba(X[t:end], prior_proba=belief)

            probs[t:end, :k] = p
            if k < self.max_k:
                probs[t:end, k:] = 0.0
            stress[t:end] = model.stress_proba(p)
            n_states[t:end] = k
            refit[t] = True
            self.history_.append(
                {
                    "refit_pos": t,
                    "refit_time": index[t] if index is not None else t,
                    "train_start": start,
                    "n_states": k,
                    "log_likelihood": model.log_likelihood_,
                    "bic": model.bic_,
                    "aic": model.aic_,
                    "converged": model.converged_,
                    "min_state_obs": float(model.state_occupancy_.min()),
                    "stress_states": model.stress_states_.copy(),
                    "warnings": list(model.warnings_),
                    "model": model,
                }
            )

        flagged = sum(bool(h["warnings"]) for h in self.history_)
        if flagged:
            logger.warning(
                "%d of %d refits have a fit warning (EM not converged or a state "
                "with almost no bars); see the 'warnings' entries of history_.",
                flagged, len(self.history_),
            )

        cols = [f"prob_regime_{i}" for i in range(self.max_k)]
        out = pd.DataFrame(probs, columns=cols, index=index)
        regime = np.full(n, np.nan)
        valid = ~np.isnan(probs[:, 0])
        regime[valid] = np.nanargmax(probs[valid], axis=1)
        out["regime"] = pd.Series(regime, index=out.index).astype("Int64")
        out["n_states"] = n_states
        out["refit"] = refit
        out["prob_stress"] = np.clip(stress, 0.0, 1.0)
        return out
