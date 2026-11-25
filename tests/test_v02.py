"""
Regression tests for the version 0.2 audit fixes: first-bar distribution,
state ordering and the stress mapping, covariance ridge, single calibration
per candidate K, unrounded selection, input validation, the per-bar
volatility feature, synthetic data and the ``src`` compatibility layer.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import hmm_regime
from hmm_regime.data_loader import (
    compute_log_range_variance,
    compute_log_returns,
    compute_parkinson_volatility,
    fetch_sample_data,
    prepare_hmm_features,
    simulate_regime_ohlc,
)
from hmm_regime.hmm_model import MarketRegimeHMM, WalkForwardHMM, stationary_distribution
from hmm_regime.model_selection import (
    evaluate_regime_models,
    select_model,
    select_n_components,
)


@pytest.fixture(scope="module")
def features():
    df = fetch_sample_data(n_bars=700, seed=42)
    feature_df, X = prepare_hmm_features(df, vol_window=21)
    return feature_df, X


@pytest.fixture(scope="module")
def fitted(features):
    _, X = features
    return MarketRegimeHMM(n_components=2, n_init=3).fit(X[:300])


# --- stationary distribution and the first bar ---------------------------------
def test_stationary_distribution_known_chain():
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = stationary_distribution(A)
    np.testing.assert_allclose(pi, [5 / 7, 2 / 7], atol=1e-12)
    np.testing.assert_allclose(pi @ A, pi, atol=1e-12)


def test_stationary_distribution_without_unique_solution():
    """Two states that are never left: any mix is stationary; a valid one is returned."""
    pi = stationary_distribution(np.eye(2))
    np.testing.assert_allclose(pi, [0.5, 0.5])
    A = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]])
    pi = stationary_distribution(A)
    assert pi.min() >= 0 and pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ A, pi, atol=1e-9)


def test_em_startprob_is_one_state(fitted):
    """Why the fitted startprob_ must not start a new sequence: EM estimates it
    from the single first training bar."""
    assert fitted.model.startprob_.max() > 0.999


def test_first_bar_of_new_sequence_follows_the_data(features, fitted):
    """The first filtered bar must depend on the first observation."""
    _, X = features
    calm = fitted.means_[0][None, :]
    stressed = fitted.means_[1][None, :]

    p_calm = fitted.predict_filtered_proba(calm)[0]
    p_stressed = fitted.predict_filtered_proba(stressed)[0]
    assert p_calm[0] > 0.5 and p_stressed[1] > 0.5

    # the old behaviour: the state of bar 0 is fixed by startprob_, whatever bar 0 is
    forced = int(np.argmax(fitted.model.startprob_))
    for obs in (calm, stressed):
        assert fitted.predict_filtered_proba(obs, initial="startprob")[0, forced] > 0.999


def test_first_bar_prior_is_the_stationary_distribution(fitted):
    """With an uninformative observation the first bar is the stationary distribution."""
    A = fitted.model.transmat_
    np.testing.assert_allclose(fitted.stationary_ @ A, fitted.stationary_, atol=1e-10)
    # carrying the stationary distribution through one transition changes nothing
    x = fitted.means_[0][None, :]
    np.testing.assert_allclose(
        fitted.predict_filtered_proba(x),
        fitted.predict_filtered_proba(x, prior_proba=fitted.stationary_),
        atol=1e-10,
    )


def test_initial_argument_validated(features, fitted):
    _, X = features
    with pytest.raises(ValueError):
        fitted.predict_filtered_proba(X[:5], initial="uniform")


@pytest.mark.parametrize("bad", [[1.5, -0.5], [np.nan, 1.0], [0.5, 0.5, 0.0]])
def test_prior_proba_must_be_a_distribution(features, fitted, bad):
    _, X = features
    with pytest.raises(ValueError):
        fitted.predict_filtered_proba(X[:5], prior_proba=np.array(bad))


def test_empty_input_gives_empty_output(features, fitted):
    _, X = features
    assert fitted.predict_filtered_proba(X[:0]).shape == (0, 2)


def test_bar_no_state_can_explain_does_not_poison_the_filter(features, fitted):
    """A value so extreme that every state density underflows keeps the
    predicted distribution; the bars after it are still valid."""
    _, X = features
    Xb = X[300:320].copy()
    Xb[5] = [1e200, -1e200]
    p = fitted.predict_filtered_proba(Xb)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    clean = fitted.predict_filtered_proba(X[300:320])
    np.testing.assert_allclose(p[:5], clean[:5], atol=1e-12)


# --- state order and the stress mapping ----------------------------------------
def test_sort_by_orders_states(features):
    _, X = features
    by_var = MarketRegimeHMM(n_components=3, n_init=3, sort_by="variance:0").fit(X)
    spread = by_var.model.covars_[:, 0, 0] + by_var.model.means_[:, 0] ** 2
    assert np.all(np.diff(spread) >= 0)

    by_mean = MarketRegimeHMM(n_components=3, n_init=3, sort_by="mean:1").fit(X)
    assert np.all(np.diff(by_mean.means_[:, 1]) >= 0)

    by_trace = MarketRegimeHMM(n_components=3, n_init=3, sort_by="trace").fit(X)
    assert np.all(np.diff([np.trace(c) for c in by_trace.model.covars_]) >= 0)

    # the same fit, only relabelled
    assert by_var.log_likelihood_ == pytest.approx(by_mean.log_likelihood_)
    assert by_var.log_likelihood_ == pytest.approx(by_trace.log_likelihood_)


@pytest.mark.parametrize("bad", ["volatility", "variance", "variance:x", "mean:-1", "var:0"])
def test_invalid_sort_by_rejected(bad):
    with pytest.raises(ValueError):
        MarketRegimeHMM(sort_by=bad)
    with pytest.raises(ValueError):
        WalkForwardHMM(sort_by=bad)


def test_sort_by_feature_out_of_range(features):
    _, X = features
    with pytest.raises(ValueError):
        MarketRegimeHMM(sort_by="variance:5", n_init=1).fit(X)


def test_state_holding_a_few_outlier_bars_is_stress_not_calm():
    """Three near-identical extreme bars form a state with a tiny variance of
    its own. Sorting by that variance would call it the calm state and the
    other 300 bars 'stress'. Its spread around the window mean is large."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 300)
    x[[40, 150, 260]] = [8.0, 8.02, 7.98]
    model = MarketRegimeHMM(n_components=2, n_init=5).fit(x)

    rare = int(np.argmin(model.state_occupancy_))
    assert model.state_occupancy_[rare] == pytest.approx(3.0, abs=0.5)
    assert rare == 1
    assert model.stress_states_.tolist() == [False, True]
    assert model.model.covars_[1, 0, 0] < model.model.covars_[0, 0, 0]  # own variance is smaller
    stress = model.stress_proba(model.predict_filtered_proba(x))
    assert (stress > 0.5).sum() == 3


def test_stress_states_two_state_model_is_state_one(fitted):
    assert fitted.stress_states_.tolist() == [False, True]


def test_prob_stress_equals_state_one_for_two_states(features):
    feature_df, _ = features
    out = WalkForwardHMM(n_components=2, window_size=252, step_size=63,
                         n_init=2).fit_predict_filtered(feature_df)
    oos = out.dropna(subset=["prob_regime_0"])
    np.testing.assert_allclose(oos["prob_stress"], oos["prob_regime_1"], atol=1e-12)
    assert out["prob_stress"].iloc[:252].isna().all()


def test_prob_stress_sums_the_stress_states(features):
    _, X = features
    wf = WalkForwardHMM(n_components=3, window_size=252, step_size=126, n_init=2)
    out = wf.fit_predict_filtered(X)
    probs = out[["prob_regime_0", "prob_regime_1", "prob_regime_2"]].to_numpy()
    for h in wf.history_:
        mask = h["stress_states"]
        assert 1 <= mask.sum() <= 2          # never all states, never none
        assert not mask[0] and mask[-1]      # calmest is calm, most volatile is stress
        rows = slice(h["refit_pos"], h["refit_pos"] + 126)
        np.testing.assert_allclose(out["prob_stress"].to_numpy()[rows],
                                   probs[rows][:, mask].sum(axis=1), atol=1e-12)


def test_history_reports_fit_quality(features):
    _, X = features
    wf = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=126, n_init=2)
    wf.fit_predict_filtered(X)
    for h in wf.history_:
        assert {"converged", "min_state_obs", "bic", "aic", "stress_states", "warnings"} <= set(h)
        assert h["bic"] == pytest.approx(h["model"].bic_)
        assert h["min_state_obs"] == pytest.approx(h["model"].state_occupancy_.min())


# --- one regime: K = 1 ------------------------------------------------------------
def test_single_state_model():
    X = np.random.default_rng(1).normal(size=(200, 2))
    model = MarketRegimeHMM(n_components=1, n_init=1).fit(X)
    p = model.predict_filtered_proba(X[:10])
    np.testing.assert_allclose(p, 1.0)
    assert model.stress_states_.tolist() == [False]
    np.testing.assert_allclose(model.stress_proba(p), 0.0)


def test_walk_forward_can_choose_one_regime():
    """Data with one regime: with K = 1 among the candidates BIC picks it and
    nothing is called stress. With candidates [2, 3] a split is forced."""
    X = np.random.default_rng(2).normal(size=(500, 2))
    out = WalkForwardHMM(n_components=[1, 2, 3], window_size=250, step_size=125,
                         n_init=3).fit_predict_filtered(X)
    oos = out.dropna(subset=["prob_regime_0"])
    assert (oos["n_states"] == 1).all()
    assert (oos["prob_stress"] == 0).all()
    assert (oos["regime"] == 0).all()


# --- covariance ridge ---------------------------------------------------------------
@pytest.mark.parametrize("cov", ["full", "diag"])
def test_covariance_ridge_bounds_a_collapsing_state(cov):
    """Identical bars would give a singular covariance (unbounded likelihood)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    X[[20, 90, 160]] = [6.0, -6.0]
    model = MarketRegimeHMM(n_components=2, covariance_type=cov, n_init=4).fit(X)
    assert np.isfinite(model.log_likelihood_)
    rare = int(np.argmin(model.state_occupancy_))
    eig = np.linalg.eigvalsh(model.model.covars_[rare])
    n_k = model.state_occupancy_[rare]
    assert eig.min() >= 0.9 * model.covar_prior / (n_k + 1)
    # an off-diagonal prior would push the correlation of that state to +1
    c = model.model.covars_[rare]
    assert abs(c[0, 1]) / np.sqrt(c[0, 0] * c[1, 1]) < 0.5


def test_negative_covar_prior_rejected():
    with pytest.raises(ValueError):
        MarketRegimeHMM(covar_prior=-1.0)


def test_near_empty_state_is_reported(caplog):
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 300)
    x[150] = 9.0
    with caplog.at_level(logging.WARNING, logger="hmm_regime"):
        model = MarketRegimeHMM(n_components=2, n_init=5).fit(x)
    if model.state_occupancy_.min() < 2:
        assert model.warnings_ and "expected training bars" in caplog.text
    else:  # EM found another split: then nothing to report
        assert not model.warnings_


def test_too_few_samples_rejected():
    with pytest.raises(ValueError):
        MarketRegimeHMM(n_components=3, n_init=1).fit(np.random.default_rng(0).normal(size=(8, 2)))


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_n_components_rejected(bad):
    with pytest.raises(ValueError):
        MarketRegimeHMM(n_components=bad)


def test_walk_forward_rejects_bad_configuration():
    with pytest.raises(ValueError):
        WalkForwardHMM(n_components=[])
    with pytest.raises(ValueError):
        WalkForwardHMM(n_components=[0, 2])
    with pytest.raises(ValueError):
        WalkForwardHMM(covariance_type="tied")


# --- model selection ------------------------------------------------------------------
def test_fit_statistics(fitted):
    assert fitted.converged_ is True
    assert fitted.n_iter_ >= 1
    assert fitted.n_parameters_ == 13
    assert fitted.aic_ == pytest.approx(-2 * fitted.log_likelihood_ + 2 * 13)
    assert fitted.bic_ == pytest.approx(-2 * fitted.log_likelihood_ + 13 * np.log(300))
    assert fitted.state_occupancy_.sum() == pytest.approx(300)


def test_evaluate_reports_unrounded_scores(features):
    _, X = features
    table = evaluate_regime_models(X[:252], components_range=[2, 3], n_init=2)
    for _, row in table.iterrows():
        m = MarketRegimeHMM(n_components=int(row["States (K)"]), n_init=2).fit(X[:252])
        assert row["Log-Likelihood"] == pytest.approx(m.log_likelihood_, abs=1e-9)
        assert row["BIC"] == pytest.approx(m.bic_, abs=1e-9)
        assert row["Min state obs"] == pytest.approx(m.state_occupancy_.min())
    assert table["Converged"].all()
    assert table["BIC"].is_monotonic_increasing


def test_select_model_returns_the_fitted_winner(features):
    _, X = features
    model = select_model(X[:252], [2, 3], n_init=2)
    assert model.n_components == select_n_components(X[:252], [2, 3], n_init=2)
    direct = MarketRegimeHMM(n_components=model.n_components, n_init=2).fit(X[:252])
    assert model.log_likelihood_ == pytest.approx(direct.log_likelihood_)
    with pytest.raises(ValueError):
        select_model(X[:252], [2, 3], criterion="hqc")


def test_walk_forward_calibrates_each_candidate_once(features, monkeypatch):
    """Before 0.2 the selected K was calibrated a second time."""
    _, X = features
    calls = []
    original = MarketRegimeHMM._fit

    def counting(self, X, warm_start=None):
        calls.append(self.n_components)
        return original(self, X, warm_start=warm_start)

    monkeypatch.setattr(MarketRegimeHMM, "_fit", counting)
    wf = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=126, n_init=2)
    wf.fit_predict_filtered(X)
    assert len(calls) == 2 * len(wf.history_)


def test_walk_forward_passes_tol(features):
    _, X = features
    wf = WalkForwardHMM(n_components=2, window_size=252, step_size=200, n_init=1, tol=1e-2)
    wf.fit_predict_filtered(X)
    assert wf.history_[0]["model"].tol == 1e-2


# --- warm start -----------------------------------------------------------------------
@pytest.mark.parametrize("cov", ["full", "diag"])
def test_warm_start_is_one_more_start(features, cov):
    _, X = features
    kw = dict(n_components=2, covariance_type=cov, n_init=2)
    previous = MarketRegimeHMM(**kw).fit(X[:252])
    cold = MarketRegimeHMM(**kw).fit(X[21:273])
    warm = MarketRegimeHMM(**kw).fit(X[21:273], warm_start=previous)
    assert warm.log_likelihood_ >= cold.log_likelihood_ - 1e-8
    # a model with another K cannot seed the fit and is ignored
    other = MarketRegimeHMM(n_components=3, covariance_type=cov, n_init=2).fit(X[:252])
    same = MarketRegimeHMM(**kw).fit(X[21:273], warm_start=other)
    assert same.log_likelihood_ == pytest.approx(cold.log_likelihood_)


def test_warm_start_walk_forward_has_no_lookahead(features):
    _, X = features
    kwargs = dict(n_components=[2, 3], window_size=200, step_size=50, n_init=2, warm_start=True)
    base = WalkForwardHMM(**kwargs).fit_predict_filtered(X)
    Xc = X.copy()
    Xc[450:] = [-0.5, 10.0]
    shocked = WalkForwardHMM(**kwargs).fit_predict_filtered(Xc)
    cols = ["prob_regime_0", "prob_regime_1", "prob_regime_2", "n_states", "prob_stress"]
    pd.testing.assert_frame_equal(base[cols].iloc[:450], shocked[cols].iloc[:450],
                                  check_exact=False, atol=1e-10)


# --- features and input validation ------------------------------------------------------
def test_non_positive_prices_rejected():
    with pytest.raises(ValueError):
        compute_log_returns(pd.Series([100.0, 0.0, 101.0]))
    with pytest.raises(ValueError):
        compute_log_returns(pd.Series([100.0, -5.0, 101.0]))
    # a missing bar is not an error: its returns are NaN and get dropped later
    r = compute_log_returns(pd.Series([100.0, np.nan, 101.0]))
    assert r.isna().all()


def test_high_below_low_rejected():
    high, low = pd.Series([10.0, 9.0]), pd.Series([9.5, 9.5])
    with pytest.raises(ValueError):
        compute_parkinson_volatility(high, low, window=1)
    with pytest.raises(ValueError):
        compute_log_range_variance(high, low)


def test_unsorted_or_duplicated_index_rejected():
    df = fetch_sample_data(n_bars=100, seed=0)
    with pytest.raises(ValueError):
        prepare_hmm_features(df.iloc[::-1])
    with pytest.raises(ValueError):
        prepare_hmm_features(pd.concat([df.iloc[:50], df.iloc[49:]]))
    with pytest.raises(KeyError):
        prepare_hmm_features(df.rename(columns={"Close": "close"}))
    with pytest.raises(ValueError):
        prepare_hmm_features(df, vol_feature="garch")
    with pytest.raises(ValueError):
        prepare_hmm_features(df[["Close"]], vol_feature="bar")


def test_log_range_variance_formula_and_zero_range():
    high = pd.Series([101.0, 102.0, 100.0, 103.0])
    low = pd.Series([99.0, 100.0, 100.0, 100.0])   # bar 2 has no range
    v = compute_log_range_variance(high, low)
    expected0 = np.log(np.log(101.0 / 99.0) ** 2 / (4 * np.log(2)))
    assert v.iloc[0] == pytest.approx(expected0)
    assert v.iloc[2] == v.iloc[1]                  # takes the last earlier value
    assert np.isfinite(v).all()


def test_bar_feature_matches_per_bar_parkinson():
    """exp(log_range_var) is the one-bar Parkinson variance."""
    df = fetch_sample_data(n_bars=200, seed=5)
    feature_df, X = prepare_hmm_features(df, vol_feature="bar")
    assert list(feature_df.columns) == ["log_return", "log_range_var"]
    assert X.shape == (199, 2)
    one_bar = compute_parkinson_volatility(df["High"], df["Low"], window=1, annualize=False)
    np.testing.assert_allclose(np.exp(feature_df["log_range_var"]), one_bar.loc[feature_df.index] ** 2)


@pytest.mark.parametrize("vol_feature", ["rolling", "bar"])
def test_features_have_no_lookahead(vol_feature):
    df = fetch_sample_data(n_bars=300, seed=6)
    full, _ = prepare_hmm_features(df, vol_feature=vol_feature)
    for T in (60, 173, 299):
        part, _ = prepare_hmm_features(df.iloc[:T], vol_feature=vol_feature)
        pd.testing.assert_frame_equal(part, full.loc[part.index])


def test_periods_per_year_only_rescales_the_rolling_volatility():
    df = fetch_sample_data(n_bars=200, seed=7)
    a, _ = prepare_hmm_features(df)
    b, _ = prepare_hmm_features(df, periods_per_year=365)
    np.testing.assert_allclose(b["volatility"] / a["volatility"], np.sqrt(365 / 252))
    np.testing.assert_allclose(a["log_return"], b["log_return"])


def test_bar_feature_separates_the_regimes():
    """The per-bar feature carries the regime: its mean differs by about
    ln(sigma_1^2 / sigma_0^2) and its spread is about the same in both."""
    df, states = fetch_sample_data(n_bars=20000, seed=1, return_states=True)
    v = compute_log_range_variance(df["High"], df["Low"])
    gap = v[states == 1].mean() - v[states == 0].mean()
    assert gap == pytest.approx(np.log(0.025 ** 2 / 0.008 ** 2), rel=0.05)
    assert v[states == 1].std() == pytest.approx(v[states == 0].std(), rel=0.05)


# --- synthetic data ---------------------------------------------------------------------
def test_sample_data_leaves_the_global_random_state_alone():
    np.random.seed(123)
    expected = np.random.rand(3)
    np.random.seed(123)
    fetch_sample_data(n_bars=50, seed=9)
    np.testing.assert_array_equal(np.random.rand(3), expected)


def test_sample_data_values_unchanged_from_version_01():
    """Same stream as the old np.random.seed(seed) implementation."""
    np.random.seed(11)
    states = np.zeros(30, dtype=int)
    trans = np.array([[0.98, 0.02], [0.05, 0.95]])
    cur = 0
    for t in range(1, 30):
        cur = np.random.choice([0, 1], p=trans[cur])
        states[t] = cur
    mu, sig = np.array([0.0008, -0.0012]), np.array([0.008, 0.025])
    returns = np.array([np.random.normal(mu[s], sig[s]) for s in states])
    close = 100.0 * np.exp(np.cumsum(returns))
    df, st = fetch_sample_data(n_bars=30, seed=11, return_states=True)
    np.testing.assert_array_equal(st.to_numpy(), states)
    np.testing.assert_allclose(df["Close"].to_numpy(), close)


def test_simulated_bars_are_consistent():
    df, states = simulate_regime_ohlc(n_bars=2000, seed=4, return_states=True)
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-12).all()
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-12).all()
    np.testing.assert_allclose(df["Open"].to_numpy()[1:], df["Close"].to_numpy()[:-1])
    assert set(states.unique()) <= {0, 1}
    r = np.log(df["Close"] / df["Open"])
    assert r[states == 0].std() == pytest.approx(0.008, rel=0.1)
    assert r[states == 1].std() == pytest.approx(0.025, rel=0.1)
    again = simulate_regime_ohlc(n_bars=2000, seed=4)
    pd.testing.assert_frame_equal(df, again)


def test_simulated_fat_tails_keep_the_variance():
    df, states = simulate_regime_ohlc(n_bars=60000, seed=8, student_df=5, return_states=True)
    r = np.log(df["Close"] / df["Open"])[states == 0]
    assert r.std() == pytest.approx(0.008, rel=0.1)
    assert r.kurt() > 1.5   # excess kurtosis of a t(5) is 6; a Gaussian has 0


def test_simulator_validates_its_inputs():
    with pytest.raises(ValueError):
        simulate_regime_ohlc(transmat=[[0.9, 0.2], [0.1, 0.9]])
    with pytest.raises(ValueError):
        simulate_regime_ohlc(sigma=(0.01, 0.0))
    with pytest.raises(ValueError):
        simulate_regime_ohlc(student_df=2.0)


def test_three_state_simulation():
    A = [[0.97, 0.02, 0.01], [0.03, 0.95, 0.02], [0.02, 0.03, 0.95]]
    df, states = simulate_regime_ohlc(n_bars=3000, seed=2, transmat=A, mu=(0, 0, 0),
                                      sigma=(0.005, 0.012, 0.03), return_states=True)
    assert set(states.unique()) == {0, 1, 2}
    assert len(df) == 3000


# --- package and compatibility ------------------------------------------------------------
def test_package_exports():
    # Check the form, not the number: a hard-coded number broke CI on the 0.2.1 bump.
    # The release workflow checks that the tag matches __version__.
    from packaging.version import Version   # installed with pytest

    version = hmm_regime.__version__
    assert isinstance(version, str)
    assert str(Version(version)) == version, "not a normalised PEP 440 version"
    for name in hmm_regime.__all__:
        assert hasattr(hmm_regime, name), name


def test_src_imports_still_work():
    """Code written against version 0.1 imported from ``src``."""
    from src.data_loader import fetch_sample_data as f, prepare_hmm_features as p
    from src.hmm_model import MarketRegimeHMM as M, WalkForwardHMM as W, _as_2d
    from src.model_selection import count_free_parameters, evaluate_regime_models as e
    from src.whipsaw import add_filtered_regime as a, filter_regimes

    assert W is WalkForwardHMM and M is MarketRegimeHMM
    assert f is fetch_sample_data and p is prepare_hmm_features
    assert e is evaluate_regime_models and a is hmm_regime.add_filtered_regime
    assert count_free_parameters(2, 2) == 13
    assert _as_2d([1.0, 2.0]).shape == (2, 1)
    assert filter_regimes(np.array([[0.9, 0.1]])).tolist() == [0.0]


def test_version_01_call_signatures_still_work(features):
    """Positional arguments of the 0.1 constructors keep their meaning."""
    _, X = features
    model = MarketRegimeHMM(2, "diag", 50, 2, True, 1e-3, 1e-4, 7)
    assert (model.covariance_type, model.n_iter, model.n_init, model.random_state) == ("diag", 50, 2, 7)
    wf = WalkForwardHMM([2, 3], 252, 63, False, "full", "aic", 50, 2, True, 1e-3, 7)
    assert (wf.window_size, wf.step_size, wf.criterion) == (252, 63, "aic")
    assert wf.model_kwargs["random_state"] == 7
    out = wf.fit_predict_filtered(X)
    assert list(out.columns[:6]) == ["prob_regime_0", "prob_regime_1", "prob_regime_2",
                                     "regime", "n_states", "refit"]


# --- fetch_returns (yfinance replaced by a stub: no network in tests) -----------------------
def test_fetch_returns_with_stubbed_yfinance(monkeypatch):
    import sys
    import types

    from hmm_regime.data_loader import fetch_returns

    calls = []
    bars = fetch_sample_data(n_bars=120, seed=3)

    def download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        if symbol == "NOPE":
            return pd.DataFrame()
        raw = bars.copy()
        # newer yfinance versions return (field, ticker) columns
        raw.columns = pd.MultiIndex.from_product([raw.columns, [symbol]])
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))

    rolling = fetch_returns("xauusd", start="2020-01-01", vol_window=10)
    assert calls[-1][0] == "GC=F"                       # alias resolved, case-insensitive
    assert calls[-1][1]["auto_adjust"] is True
    expected, _ = prepare_hmm_features(bars, vol_window=10)
    pd.testing.assert_frame_equal(rolling, expected)

    per_bar = fetch_returns("SPY", vol_feature="bar")
    assert calls[-1][0] == "SPY"
    assert list(per_bar.columns) == ["log_return", "log_range_var"]

    with pytest.raises(ValueError):
        fetch_returns("NOPE")
