"""
Regression tests for: walk-forward estimation, input scaling, diag reordering,
multi-start EM, filter continuity across refits, and per-window K selection.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import MarketRegimeHMM, WalkForwardHMM
from src.model_selection import evaluate_regime_models, select_n_components


@pytest.fixture(scope="module")
def features():
    df = fetch_sample_data(n_bars=700, seed=42)
    feature_df, X = prepare_hmm_features(df, vol_window=21)
    return feature_df, X


# --- scaling: 3-state full covariance must not crash ---------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 7])
def test_three_state_full_covariance_fits(seed):
    _, X = prepare_hmm_features(fetch_sample_data(n_bars=600, seed=seed))
    model = MarketRegimeHMM(n_components=3, covariance_type="full", n_init=3,
                            random_state=seed).fit(X)
    assert np.isfinite(model.log_likelihood_)


def test_scaler_uses_training_data_only(features):
    _, X = features
    model = MarketRegimeHMM(n_components=2, n_init=2).fit(X[:300])
    np.testing.assert_allclose(model.mean_, X[:300].mean(axis=0))
    np.testing.assert_allclose(model.scale_, X[:300].std(axis=0))


def test_unscaled_params_reported_in_original_units(features):
    _, X = features
    model = MarketRegimeHMM(n_components=2, n_init=2).fit(X[:300])
    # vol feature (annualised) is ~0.1-0.5, return feature is ~0
    assert np.all(model.means_[:, 1] > 0.01)
    assert np.all(np.abs(model.means_[:, 0]) < 0.05)
    assert model.covars_.shape == (2, 2, 2)


# --- diag reordering ------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
def test_diag_covariance_reorders(features, k):
    _, X = features
    model = MarketRegimeHMM(n_components=k, covariance_type="diag", n_init=3).fit(X)
    traces = [np.trace(c) for c in model.model.covars_]
    assert all(np.diff(traces) >= 0), traces
    probs = model.predict_filtered_proba(X[:50])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_unsupported_covariance_type_rejected():
    with pytest.raises(ValueError):
        MarketRegimeHMM(covariance_type="tied")


# --- multi-start EM ---------------------------------------------------------
def test_multi_start_not_worse_than_single(features):
    _, X = features
    single = MarketRegimeHMM(n_components=3, n_init=1, random_state=0).fit(X)
    multi = MarketRegimeHMM(n_components=3, n_init=8, random_state=0).fit(X)
    assert multi.log_likelihood_ >= single.log_likelihood_ - 1e-8


# --- filter continuity ------------------------------------------------------
def test_prior_proba_continues_filter(features):
    """Filtering in two chunks with the carried belief equals one pass."""
    _, X = features
    model = MarketRegimeHMM(n_components=2, n_init=2).fit(X[:300])
    full = model.predict_filtered_proba(X[300:500])
    first = model.predict_filtered_proba(X[300:400])
    second = model.predict_filtered_proba(X[400:500], prior_proba=first[-1])
    np.testing.assert_allclose(np.vstack([first, second]), full, atol=1e-12)


def test_prior_proba_validation(features):
    _, X = features
    model = MarketRegimeHMM(n_components=2, n_init=1).fit(X[:300])
    with pytest.raises(ValueError):
        model.predict_filtered_proba(X[300:310], prior_proba=np.array([0.3, 0.3]))


# --- walk-forward -----------------------------------------------------------
def test_walk_forward_output_shape(features):
    feature_df, _ = features
    wf = WalkForwardHMM(n_components=2, window_size=252, step_size=63, n_init=2)
    out = wf.fit_predict_filtered(feature_df)

    assert out.index.equals(feature_df.index)
    assert out.iloc[:252]["prob_regime_0"].isna().all()
    oos = out.iloc[252:]
    assert oos["prob_regime_0"].notna().all()
    np.testing.assert_allclose(oos[["prob_regime_0", "prob_regime_1"]].sum(axis=1), 1.0)
    assert out["refit"].sum() == len(range(252, len(out), 63))


def test_walk_forward_has_no_lookahead(features):
    """Corrupting data after T must not change any output before T."""
    _, X = features
    T = 450
    kwargs = dict(n_components=[2, 3], window_size=200, step_size=50, n_init=2)
    base = WalkForwardHMM(**kwargs).fit_predict_filtered(X)

    Xc = X.copy()
    Xc[T:, 0] = -0.5
    Xc[T:, 1] = 10.0
    shocked = WalkForwardHMM(**kwargs).fit_predict_filtered(Xc)

    cols = ["prob_regime_0", "prob_regime_1", "prob_regime_2", "n_states"]
    pd.testing.assert_frame_equal(base[cols].iloc[:T], shocked[cols].iloc[:T],
                                  check_exact=False, atol=1e-10)


def test_walk_forward_differs_from_full_sample_fit(features):
    """Sanity check: the walk-forward result is not a full-sample fit in disguise."""
    _, X = features
    wf = WalkForwardHMM(n_components=2, window_size=252, step_size=63, n_init=2)
    out = wf.fit_predict_filtered(X)
    full = MarketRegimeHMM(n_components=2, n_init=2).fit(X).predict_filtered_proba(X)
    assert not np.allclose(out["prob_regime_1"].iloc[252:], full[252:, 1], atol=1e-6)


def test_no_jump_at_refit_boundary(features):
    """The first OOS bar after a refit uses the carried belief, not startprob_."""
    _, X = features
    wf = WalkForwardHMM(n_components=2, window_size=252, step_size=63, n_init=2)
    out = wf.fit_predict_filtered(X)
    h = wf.history_[1]
    t, model = h["refit_pos"], h["model"]
    belief = model.predict_filtered_proba(X[h["train_start"]:t])[-1]
    expected = model.predict_filtered_proba(X[t:t + 1], prior_proba=belief)[0]
    np.testing.assert_allclose(out.iloc[t][["prob_regime_0", "prob_regime_1"]].to_numpy(float),
                               expected, atol=1e-12)


# --- K selection ------------------------------------------------------------
def test_k_selected_per_window(features):
    _, X = features
    wf = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=126, n_init=2)
    wf.fit_predict_filtered(X)
    for h in wf.history_:
        train = X[h["train_start"]:h["refit_pos"]]
        assert h["n_states"] == select_n_components(train, [2, 3], n_init=2)


def test_evaluate_handles_failed_k():
    X = np.random.default_rng(0).normal(size=(5, 2))
    table = evaluate_regime_models(X, components_range=[2, 4], n_init=1)
    assert len(table) == 2
