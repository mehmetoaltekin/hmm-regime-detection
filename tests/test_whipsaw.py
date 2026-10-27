"""
Tests for the whipsaw filter: causality, hysteresis, confirmation, minimum
dwell, K changes between refits, and the turnover / lag helpers.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import WalkForwardHMM
from src.whipsaw import (
    add_filtered_regime,
    calm_vs_stress,
    detection_lag,
    filter_regimes,
    switch_stats,
)


def two_state(p_stress):
    p = np.asarray(p_stress, dtype=float)
    return np.column_stack([1.0 - p, p])


def random_probs(n, k, seed, persistence=0.9):
    """Persistent random probabilities, so that runs and flips both occur."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(k)
    rows = []
    for _ in range(n):
        logits = persistence * logits + rng.normal(scale=1.0, size=k)
        e = np.exp(logits - logits.max())
        rows.append(e / e.sum())
    return np.array(rows)


def n_switches(labels):
    lab = labels[~np.isnan(labels)]
    return int((lab[1:] != lab[:-1]).sum())


# --- basic rules ------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
def test_no_filtering_equals_argmax(k):
    P = random_probs(500, k, seed=k)
    out = filter_regimes(P, threshold=1e-9, confirm_bars=1, min_dwell=0)
    np.testing.assert_array_equal(out, P.argmax(axis=1))


def test_single_bar_spike_is_removed():
    P = two_state([0.1] * 10 + [0.9] + [0.1] * 10)
    assert n_switches(P.argmax(axis=1).astype(float)) == 2
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    assert (out == 0).all()


def test_hysteresis_band_holds_regime():
    """Probabilities flipping around 0.5 but never reaching 0.7 cause no switch."""
    P = two_state([0.45, 0.65] * 50)
    assert n_switches(P.argmax(axis=1).astype(float)) == 99
    out = filter_regimes(P, threshold=0.7, confirm_bars=1)
    assert (out == 0).all()


def test_confirmation_delays_switch_by_confirm_bars_minus_one():
    P = two_state([0.1] * 5 + [0.9] * 10)
    out = filter_regimes(P, threshold=0.7, confirm_bars=3)
    assert np.flatnonzero(out == 1)[0] == 5 + 2


def test_confirmation_requires_consecutive_bars():
    P = two_state([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0, 1])


def test_min_dwell_spaces_switches():
    P = two_state([0.95, 0.95, 0.05, 0.05] * 25)
    out = filter_regimes(P, threshold=0.7, confirm_bars=1, min_dwell=5)
    switch_at = np.flatnonzero(out[1:] != out[:-1]) + 1
    assert len(switch_at) > 0
    assert np.all(np.diff(switch_at) >= 5)


def test_nan_rows_give_nan_and_start_at_argmax():
    P = two_state([np.nan] * 4 + [0.6, 0.6, 0.1])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    assert np.isnan(out[:4]).all()
    np.testing.assert_array_equal(out[4:], [1, 1, 1])


def test_regime_missing_after_k_shrinks_forces_switch():
    P = np.array([
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
        [0.55, 0.45, 0.00],  # refit with K=2: state 2 no longer exists
        [0.55, 0.45, 0.00],
    ])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2, n_states=[3, 3, 2, 2])
    np.testing.assert_array_equal(out, [2, 2, 0, 0])


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("params", [
    dict(threshold=0.6, confirm_bars=1, min_dwell=0),
    dict(threshold=0.7, confirm_bars=2, min_dwell=0),
    dict(threshold=0.4, confirm_bars=3, min_dwell=10),
])
def test_never_more_switches_than_raw_argmax(k, params):
    for seed in range(5):
        P = random_probs(400, k, seed=100 + seed)
        filtered = filter_regimes(P, **params)
        raw = P.argmax(axis=1).astype(float)
        assert n_switches(filtered) <= n_switches(raw)


@pytest.mark.parametrize("bad", [
    dict(threshold=0.0), dict(threshold=1.5), dict(confirm_bars=0), dict(min_dwell=-1),
])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        filter_regimes(two_state([0.1, 0.9]), **bad)


# --- causality ----------------------------------------------------------------
def test_filter_is_causal():
    """Labels up to T depend only on probabilities up to T."""
    P = random_probs(300, 3, seed=7)
    params = dict(threshold=0.6, confirm_bars=2, min_dwell=4)
    full = filter_regimes(P, **params)
    for T in (50, 151, 299):
        np.testing.assert_array_equal(filter_regimes(P[:T], **params), full[:T])
        Pc = P.copy()
        Pc[T:] = np.roll(Pc[T:], 1, axis=1)
        np.testing.assert_array_equal(filter_regimes(Pc, **params)[:T], full[:T])


@pytest.fixture(scope="module")
def walk_forward_features():
    df = fetch_sample_data(n_bars=700, seed=42)
    feature_df, X = prepare_hmm_features(df, vol_window=21)
    return feature_df, X


def test_walk_forward_plus_filter_has_no_lookahead(walk_forward_features):
    _, X = walk_forward_features
    T = 450
    kwargs = dict(n_components=[2, 3], window_size=200, step_size=50, n_init=2)
    base = add_filtered_regime(WalkForwardHMM(**kwargs).fit_predict_filtered(X))

    Xc = X.copy()
    Xc[T:, 0] = -0.5
    Xc[T:, 1] = 10.0
    shocked = add_filtered_regime(WalkForwardHMM(**kwargs).fit_predict_filtered(Xc))
    pd.testing.assert_series_equal(base["regime_filtered"].iloc[:T],
                                   shocked["regime_filtered"].iloc[:T])


# --- DataFrame wrapper ----------------------------------------------------------
def test_add_filtered_regime_on_walk_forward_output(walk_forward_features):
    feature_df, _ = walk_forward_features
    out = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=63,
                         n_init=2).fit_predict_filtered(feature_df)

    res = add_filtered_regime(out)
    assert res.index.equals(out.index)
    assert res["regime_filtered"].dtype == "Int64"
    assert res["regime_filtered"].isna().equals(out["regime"].isna())
    assert "regime_filtered" not in out  # input left untouched

    binary = add_filtered_regime(out, binary=True)
    stress = binary["stress_filtered"].dropna()
    assert set(stress.unique()) <= {0, 1}
    # unfiltered version of the same binary rule: argmax of [P(calm), P(not calm)]
    raw_stress = (out["prob_regime_0"].dropna() < 0.5).astype(float)
    assert n_switches(stress.to_numpy(float)) <= n_switches(raw_stress.to_numpy())


def test_calm_vs_stress_pairs_sum_to_one():
    P = random_probs(50, 3, seed=1)
    pair = calm_vs_stress(P)
    np.testing.assert_allclose(pair[:, 0], P[:, 0])
    np.testing.assert_allclose(pair.sum(axis=1), 1.0)


# --- turnover and lag -----------------------------------------------------------
def test_switch_stats_counts_runs():
    s = switch_stats([np.nan, 0, 0, 1, 1, 1, 0], periods_per_year=6)
    assert s["bars"] == 6
    assert s["switches"] == 2
    assert s["switches_per_year"] == pytest.approx(2.0)
    assert s["mean_run"] == pytest.approx(2.0)
    assert s["runs_under_5_bars"] == 3


def test_detection_lag_measures_bars_to_catch_up():
    true = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]
    lab = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    d = detection_lag(true, lab)
    assert d["changes"] == 4
    assert d["detected"] == 3        # lags 2, 1 and 0 (the last change back to 0)
    assert d["missed"] == 1          # the one-bar stress at index 10
    assert d["mean_lag"] == pytest.approx(1.0)


def test_sample_data_states_do_not_change_prices():
    df = fetch_sample_data(n_bars=300, seed=3)
    df2, states = fetch_sample_data(n_bars=300, seed=3, return_states=True)
    pd.testing.assert_frame_equal(df, df2)
    assert states.index.equals(df.index)
    assert set(states.unique()) <= {0, 1}
