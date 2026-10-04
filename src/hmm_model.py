import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class MarketRegimeHMM:
    """
    Gaussian HMM for market regime detection.
    
    States are reordered by variance so state 0 is always the calm/low-vol
    regime. Forward filtering is used instead of smoothed inference to 
    avoid lookahead bias during backtests.
    """

    def __init__(self, n_components=2, covariance_type="full", n_iter=100, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model = None
        self.order = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        hmm.fit(X)
        hmm.n_features = n_features

        # Sort regimes by total variance (state 0 = lowest vol)
        if self.covariance_type == "full":
            variances = np.array([np.trace(c) for c in hmm.covars_])
        elif self.covariance_type == "diag":
            variances = np.array([np.sum(c) for c in hmm.covars_])
        else:
            variances = hmm.covars_.flatten()

        order = np.argsort(variances)
        self.order = order

        # Align model parameters to the sorted order
        hmm.startprob_ = hmm.startprob_[order]
        hmm.transmat_ = hmm.transmat_[order, :][:, order]
        hmm.means_ = hmm.means_[order]
        hmm.covars_ = hmm.covars_[order]
        hmm.n_features = n_features

        self.model = hmm
        return self

    def predict_filtered_proba(self, X):
        """
        Forward filter: computes P(S_t | Y_1:t) online step-by-step
        without future information.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        log_trans = np.log(self.model.transmat_)
        log_start = np.log(self.model.startprob_)
        frame_log_prob = self.model._compute_log_likelihood(X)

        filtered = np.zeros((n_samples, self.n_components))

        # t = 0
        curr_log = log_start + frame_log_prob[0]
        curr_log -= np.logaddexp.reduce(curr_log)
        filtered[0] = np.exp(curr_log)

        # t > 0 forward recursion
        for t in range(1, n_samples):
            # prior = sum_i( alpha_t-1(i) * A_ij ) in log space
            prior = np.logaddexp.reduce(curr_log[:, None] + log_trans, axis=0)
            curr_log = prior + frame_log_prob[t]
            curr_log -= np.logaddexp.reduce(curr_log)
            filtered[t] = np.exp(curr_log)

        return filtered

    def predict_regimes(self, X):
        return np.argmax(self.predict_filtered_proba(X), axis=1)

    @property
    def transition_matrix_(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        cols = [f"regime_{i}" for i in range(self.n_components)]
        return pd.DataFrame(self.model.transmat_, index=cols, columns=cols)
