"""
Tests for the whipsaw filter: causality, hysteresis, confirmation, minimum
dwell, NaN gaps, K changes between refits, and the turnover / lag helpers.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import WalkForwardHMM
from src.whipsaw import (
    add_filtered_regime,
    calm_vs_stress,
    change_lags,
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


def lookahead_cuts(fn, P):
    """Cut points T where fn(P[:T]) differs from fn(P)[:T], i.e. leaks."""
    full = fn(P)
    return [T for T in range(1, len(P) + 1)
            if not np.array_equal(fn(P[:T]), full[:T], equal_nan=True)]


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


def test_probability_equal_to_threshold_switches():
    out = filter_regimes(two_state([0.1, 0.7, 0.7]), threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 1])


def test_confirmation_delays_switch_by_confirm_bars_minus_one():
    P = two_state([0.1] * 5 + [0.9] * 10)
    out = filter_regimes(P, threshold=0.7, confirm_bars=3)
    assert np.flatnonzero(out == 1)[0] == 5 + 2


def test_confirmation_requires_consecutive_bars():
    P = two_state([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0, 1])


def test_confirmation_requires_the_same_new_state():
    """Two different states each passing once is not a confirmation."""
    P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 2])


def test_min_dwell_holds_exactly_min_dwell_bars():
    P = two_state([0.1] * 10 + [0.9] + [0.1] * 10)
    out = filter_regimes(P, threshold=0.7, confirm_bars=1, min_dwell=5)
    np.testing.assert_array_equal(np.flatnonzero(out == 1), [10, 11, 12, 13, 14])


def test_min_dwell_spaces_switches():
    P = two_state([0.95, 0.95, 0.05, 0.05] * 25)
    out = filter_regimes(P, threshold=0.7, confirm_bars=1, min_dwell=5)
    switch_at = np.flatnonzero(out[1:] != out[:-1]) + 1
    assert len(switch_at) > 0
    assert np.all(np.diff(switch_at) >= 5)


def test_first_bar_counts_for_min_dwell():
    out = filter_regimes(two_state([0.1] + [0.9] * 8), threshold=0.7,
                         confirm_bars=1, min_dwell=5)
    assert np.flatnonzero(out == 1)[0] == 5


# --- NaN rows -----------------------------------------------------------------
def test_leading_nan_rows_give_nan_and_start_at_argmax():
    P = two_state([np.nan] * 4 + [0.6, 0.6, 0.1])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    assert np.isnan(out[:4]).all()
    np.testing.assert_array_equal(out[4:], [1, 1, 1])


def test_nan_gap_keeps_held_regime():
    """A spike right after a gap still needs confirmation."""
    P = two_state([0.1] * 4 + [np.nan] + [0.9] + [0.1] * 3)
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 0, np.nan, 0, 0, 0, 0])


def test_nan_gap_breaks_confirmation():
    P = two_state([0.1, 0.1, 0.9, np.nan, 0.9, 0.1])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2)
    np.testing.assert_array_equal(out, [0, 0, 0, np.nan, 0, 0])


def test_nan_gap_counts_toward_min_dwell():
    P = two_state([0.9, np.nan, np.nan, 0.1])
    out = filter_regimes(P, threshold=0.7, confirm_bars=1, min_dwell=3)
    np.testing.assert_array_equal(out, [1, np.nan, np.nan, 0])


# --- K changes ----------------------------------------------------------------
def test_regime_missing_after_k_shrinks_forces_switch():
    P = np.array([
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
        [0.55, 0.45, 0.00],  # refit with K=2: state 2 no longer exists
        [0.55, 0.45, 0.00],
    ])
    out = filter_regimes(P, threshold=0.7, confirm_bars=2, n_states=[3, 3, 2, 2])
    np.testing.assert_array_equal(out, [2, 2, 0, 0])


def test_forced_switch_after_k_shrinks_restarts_min_dwell():
    P = np.array([[0.05, 0.05, 0.9]] * 6 + [[0.9, 0.1, 0.0]] + [[0.1, 0.9, 0.0]] * 4)
    out = filter_regimes(P, threshold=0.7, confirm_bars=1, min_dwell=3,
                         n_states=[3] * 6 + [2] * 5)
    np.testing.assert_array_equal(out, [2] * 6 + [0, 0, 0, 1, 1])


def test_wrapper_passes_n_states_for_k_shrink():
    out = pd.DataFrame({
        "prob_regime_0": [0.05, 0.05, 0.55, 0.55],
        "prob_regime_1": [0.05, 0.05, 0.45, 0.45],
        "prob_regime_2": [0.90, 0.90, 0.00, 0.00],
        "n_states": [3.0, 3.0, 2.0, 2.0],
    })
    res = add_filtered_regime(out, threshold=0.7, confirm_bars=2)
    assert res["regime_filtered"].tolist() == [2, 2, 0, 0]


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("params", [
    dict(threshold=0.6, confirm_bars=1, min_dwell=0),
    dict(threshold=0.7, confirm_bars=2, min_dwell=0),
    dict(threshold=0.4, confirm_bars=3, min_dwell=10),
])
def test_never_more_switches_than_raw_argmax(k, params):
    for seed in range(5):
        P = random_probs(400, k, seed=100 + seed)
        P[[50, 51, 200]] = np.nan
        filtered = filter_regimes(P, **params)
        raw = np.where(np.isnan(P[:, 0]), np.nan, np.nanargmax(np.nan_to_num(P, nan=-1), axis=1))
        assert n_switches(filtered) <= n_switches(raw)


@pytest.mark.parametrize("bad", [
    dict(threshold=0.0), dict(threshold=1.5), dict(confirm_bars=0), dict(min_dwell=-1),
    dict(confirm_bars=1.5), dict(min_dwell=2.5),
])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        filter_regimes(two_state([0.1, 0.9]), **bad)


# --- causality ----------------------------------------------------------------
CAUSAL_PARAMS = [
    dict(threshold=0.6, confirm_bars=1, min_dwell=0),
    dict(threshold=0.7, confirm_bars=2, min_dwell=0),
    dict(threshold=0.6, confirm_bars=2, min_dwell=4),
    dict(threshold=0.8, confirm_bars=3, min_dwell=7),
]


@pytest.fixture(scope="module")
def causal_probs():
    P = random_probs(150, 3, seed=7)
    P[:3] = np.nan     # before the first refit
    P[60:62] = np.nan  # a gap
    return P


@pytest.mark.parametrize("params", CAUSAL_PARAMS)
def test_filter_is_causal_at_every_cut(causal_probs, params):
    """Labels up to T depend only on probabilities up to T, for every T."""
    P = causal_probs
    ns = np.where(np.arange(len(P)) < 100, 3, 2)
    P2 = P.copy()
    P2[100:, 2] = 0.0
    P2[100:] /= P2[100:].sum(axis=1, keepdims=True)
    cases = {
        "K-state": lambda X: filter_regimes(X, **params),
        "binary": lambda X: filter_regimes(calm_vs_stress(X), **params),
        "K shrinks": lambda X: filter_regimes(X, n_states=ns[:len(X)], **params),
    }
    for name, fn in cases.items():
        X = P2 if name == "K shrinks" else P
        assert lookahead_cuts(fn, X) == [], name


def test_causality_check_catches_planted_leaks(causal_probs):
    """The check above must fail for filters that look one bar ahead."""
    P = causal_probs
    params = dict(threshold=0.6, confirm_bars=2, min_dwell=4)

    def shifted(X):  # label of the next bar
        f = filter_regimes(X, **params)
        return np.append(f[1:], f[-1])

    def backdated(X):  # each switch credited one bar early
        f = filter_regimes(X, **params)
        g = f.copy()
        g[np.flatnonzero(f[1:] != f[:-1])] = f[np.flatnonzero(f[1:] != f[:-1]) + 1]
        return g

    def centered(X):  # centred 3-bar majority of the raw argmax
        raw = pd.Series(np.where(np.isnan(X[:, 0]), np.nan,
                                 np.nan_to_num(X, nan=-1).argmax(axis=1)))
        return raw.rolling(3, center=True, min_periods=1).median().to_numpy()

    for fn in (shifted, backdated, centered):
        assert lookahead_cuts(fn, P), fn.__name__


@pytest.fixture(scope="module")
def walk_forward_features():
    df = fetch_sample_data(n_bars=700, seed=42)
    feature_df, X = prepare_hmm_features(df, vol_window=21)
    return feature_df, X


def test_walk_forward_plus_filter_is_causal(walk_forward_features):
    """Running on data up to T gives the same labels before T, both label types.
    Refits are at 200, 250, ...; the cuts fall inside refit windows."""
    _, X = walk_forward_features
    kwargs = dict(n_components=[2, 3], window_size=200, step_size=50, n_init=2)

    def labels(data):
        out = WalkForwardHMM(**kwargs).fit_predict_filtered(data)
        both = add_filtered_regime(add_filtered_regime(out), binary=True)
        return both[["regime_filtered", "stress_filtered"]]

    full = labels(X)
    assert full["stress_filtered"].dropna().nunique() == 2
    for T in (333, 471):
        pd.testing.assert_frame_equal(labels(X[:T]), full.iloc[:T])


# --- DataFrame wrapper ----------------------------------------------------------
@pytest.fixture(scope="module")
def wf_output(walk_forward_features):
    feature_df, _ = walk_forward_features
    return WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=63,
                          n_init=2).fit_predict_filtered(feature_df)


@pytest.mark.parametrize("binary", [False, True])
def test_add_filtered_regime_is_causal_at_every_cut(wf_output, binary):
    """The wrapper on rows up to T gives the same labels before T, for every T.
    Together with the walk-forward's own look-ahead test this covers the pipeline."""
    col = "stress_filtered" if binary else "regime_filtered"

    def labels(o):
        return add_filtered_regime(o, binary=binary)[col].to_numpy(float, na_value=np.nan)

    full = labels(wf_output)
    assert n_switches(full) > 0
    bad = [T for T in range(1, len(wf_output) + 1)
           if not np.array_equal(labels(wf_output.iloc[:T]), full[:T], equal_nan=True)]
    assert bad == []


def test_add_filtered_regime_on_walk_forward_output(wf_output):
    out = wf_output
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


def test_switch_stats_accepts_nullable_int():
    s = switch_stats(pd.Series([pd.NA, 0, 0, 1], dtype="Int64"))
    assert (s["bars"], s["switches"]) == (3, 1)


def test_detection_lag_times_tracked_changes():
    true = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]
    lab = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    d = detection_lag(true, lab)
    assert d["changes"] == 4
    assert d["detected"] == 2         # lags 2 and 1
    assert d["missed"] == 1           # the one-bar stress at index 10
    assert d["untracked"] == 1        # the change back at 11: the label never left 0
    assert d["mean_lag"] == pytest.approx(1.5)


def test_detection_lag_does_not_count_a_label_that_never_moved():
    d = detection_lag([0, 0, 0, 1, 1, 0, 0, 0], [0] * 8)
    assert (d["detected"], d["missed"], d["untracked"]) == (0, 1, 1)
    assert np.isnan(d["mean_lag"])


def test_detection_lag_false_alarm_is_untracked():
    d = detection_lag([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1])
    assert (d["detected"], d["untracked"]) == (0, 1)


def test_detection_lag_counts_nan_bars_and_censors_the_end():
    d = detection_lag([0, 0, 1, 1, 1, 1], [0, 0, np.nan, np.nan, np.nan, 1])
    assert d["mean_lag"] == 3
    d = detection_lag([0, 0, 1, 1], [0, 0, 0, 0])
    assert (d["missed"], d["censored"]) == (0, 1)


def test_detection_lag_by_direction():
    true = [0, 0, 1, 1, 1, 0, 0, 0]
    lab = [0, 0, 1, 1, 1, 1, 1, 0]
    assert detection_lag(true, lab, into=1)["mean_lag"] == 0
    assert detection_lag(true, lab, into=0)["mean_lag"] == 2


def test_detection_lag_into_counts_one_direction():
    true = [0, 0, 1, 1, 0, 0, 1, 1, 0]
    assert detection_lag(true, true, into=1)["changes"] == 2
    assert detection_lag(true, true, into=0)["changes"] == 2
    assert detection_lag(true, true)["changes"] == 4


def test_detection_lag_nan_label_before_change_is_untracked():
    d = detection_lag([0, 0, 1, 1], [0, np.nan, 1, 1])
    assert (d["untracked"], d["detected"]) == (1, 0)


def test_change_lags_one_row_per_change():
    true = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]
    lab = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    d = change_lags(true, lab)
    assert d["bar"].tolist() == [3, 7, 10, 11]
    assert d["status"].tolist() == ["detected", "detected", "missed", "untracked"]
    np.testing.assert_array_equal(d["lag"], [2, 1, np.nan, np.nan])
    assert change_lags([0, 0], [0, 0]).empty


def test_detection_lag_rejects_nan_true_states():
    with pytest.raises(ValueError):
        detection_lag([0, np.nan, 1], [0, 0, 1])


# --- synthetic data -------------------------------------------------------------
def test_sample_data_states_do_not_change_prices():
    df = fetch_sample_data(n_bars=300, seed=3)
    df2, states = fetch_sample_data(n_bars=300, seed=3, return_states=True)
    pd.testing.assert_frame_equal(df, df2)
    assert states.index.equals(df.index)
    assert set(states.unique()) <= {0, 1}


def test_sample_data_range_follows_regime():
    """Per-bar Parkinson volatility matches each regime's sigma (0.008 / 0.025)."""
    df, states = fetch_sample_data(n_bars=20000, seed=1, return_states=True)
    hl2 = np.log(df["High"] / df["Low"]) ** 2 / (4.0 * np.log(2.0))
    for state, sigma in ((0, 0.008), (1, 0.025)):
        assert np.sqrt(hl2[states == state].mean()) == pytest.approx(sigma, rel=0.05)
