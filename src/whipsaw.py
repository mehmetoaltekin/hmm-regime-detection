"""
Whipsaw filter for walk-forward regime output.

The raw regime (argmax of P(S_t | Y_1:t)) flips whenever the filtered
probabilities hover near a decision boundary. In live trading every flip is
a reallocation, so the jitter turns into turnover and costs. This module
turns the probabilities into a stickier regime label with three causal rules:

* ``threshold``    - switch only to a state whose probability is at least
                     ``threshold``. An ambiguous bar keeps the current regime
                     (hysteresis).
* ``confirm_bars`` - the same new state must pass the threshold on this many
                     consecutive bars before the switch happens.
* ``min_dwell``    - after a switch, hold the new regime for at least this
                     many bars.

The label at bar t uses probabilities up to bar t only, so it is known at the
close of bar t; act on it at the next bar's open. The filter trades fewer
switches for later detection: measure both with ``switch_stats`` and
``detection_lag``, and fix the parameters before any backtest (each setting
tried is one more trial).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.hmm_model import ArrayLike


def filter_regimes(
    probs: ArrayLike,
    threshold: float = 0.7,
    confirm_bars: int = 2,
    min_dwell: int = 0,
    n_states: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Causal regime label with hysteresis, confirmation and minimum dwell.

    Parameters
    ----------
    probs : array of shape (n_samples, K)
        Filtered probabilities, e.g. the ``prob_regime_*`` columns of
        ``WalkForwardHMM.fit_predict_filtered``. Rows containing NaN (bars
        before the first refit) give NaN and restart the filter.
    threshold : float in (0, 1]
        Minimum probability of the new state for a switch.
    confirm_bars : int >= 1
        Consecutive bars the same new state must pass ``threshold``.
        1 switches on the first bar that passes.
    min_dwell : int >= 0
        Minimum bars between two switches (0 or 1: no constraint).
    n_states : array of shape (n_samples,), optional
        K in use on each bar (the ``n_states`` column). If the held regime
        does not exist in the current model (K shrank at a refit), the filter
        moves to the argmax at once, ignoring the other rules.

    Returns
    -------
    np.ndarray of float, shape (n_samples,)
        Regime per bar (NaN where ``probs`` is NaN). The first valid bar
        starts at the argmax. Every switch goes to the bar's argmax, so the
        filtered label never switches more often than the raw argmax.
    """
    P = np.asarray(probs, dtype=float)
    if P.ndim != 2:
        raise ValueError("probs must be 2D (n_samples, K)")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if confirm_bars < 1 or min_dwell < 0:
        raise ValueError("confirm_bars must be >= 1 and min_dwell >= 0")
    n = len(P)
    ns = None
    if n_states is not None:
        ns = np.asarray(n_states, dtype=float)
        if ns.shape != (n,):
            raise ValueError("n_states must have one value per row of probs")

    out = np.full(n, np.nan)
    current = -1        # held regime, -1 = none yet
    since_switch = 0    # bars since the last switch
    pending, count = -1, 0  # candidate state and its consecutive passes

    for t in range(n):
        p = P[t]
        if np.isnan(p).any():
            current, pending, count = -1, -1, 0
            continue

        cand = int(np.argmax(p))
        if current < 0 or (ns is not None and current >= ns[t]):
            current, since_switch, pending, count = cand, 0, -1, 0
        else:
            since_switch += 1
            if cand != current and p[cand] >= threshold:
                count = count + 1 if cand == pending else 1
                pending = cand
            else:
                pending, count = -1, 0
            if pending >= 0 and count >= confirm_bars and since_switch >= min_dwell:
                current, since_switch, pending, count = pending, 0, -1, 0
        out[t] = current

    return out


def calm_vs_stress(probs: ArrayLike) -> np.ndarray:
    """
    Collapse K-state probabilities to [P(calm), P(not calm)].

    State 0 is the lowest-variance state in every window, so this pair keeps
    one meaning when the walk-forward switches K between refits, while
    labels such as "regime 2" do not.
    """
    P = np.asarray(probs, dtype=float)
    p0 = P[:, 0]
    return np.column_stack([p0, 1.0 - p0])


def add_filtered_regime(
    out: pd.DataFrame,
    binary: bool = False,
    threshold: float = 0.7,
    confirm_bars: int = 2,
    min_dwell: int = 0,
) -> pd.DataFrame:
    """
    Add a whipsaw-filtered label to ``WalkForwardHMM`` output.

    With ``binary=False`` the column ``regime_filtered`` uses the K-state
    labels of ``regime``. With ``binary=True`` the column ``stress_filtered``
    is 0 (calm, state 0) or 1 (any higher-volatility state); compare it with
    ``regime > 0``.
    """
    prob_cols = [c for c in out.columns if c.startswith("prob_regime_")]
    if not prob_cols:
        raise ValueError("out has no prob_regime_* columns")
    probs = out[prob_cols].to_numpy(float)
    kwargs = dict(threshold=threshold, confirm_bars=confirm_bars, min_dwell=min_dwell)

    res = out.copy()
    if binary:
        labels = filter_regimes(calm_vs_stress(probs), **kwargs)
        res["stress_filtered"] = pd.Series(labels, index=out.index).astype("Int64")
    else:
        n_states = out["n_states"].to_numpy(float) if "n_states" in out else None
        labels = filter_regimes(probs, n_states=n_states, **kwargs)
        res["regime_filtered"] = pd.Series(labels, index=out.index).astype("Int64")
    return res


def switch_stats(regimes: ArrayLike, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Turnover of a regime label: switches, switches per year and run lengths.

    ``periods_per_year`` must match the bar size (252 for daily bars).
    NaN bars are dropped.
    """
    r = pd.Series(np.asarray(regimes, dtype=float)).dropna().reset_index(drop=True)
    if r.empty:
        raise ValueError("no valid regime labels")
    switches = int((r != r.shift()).iloc[1:].sum())
    runs = r.groupby((r != r.shift()).cumsum()).size()
    return {
        "bars": len(r),
        "switches": switches,
        "switches_per_year": switches / len(r) * periods_per_year,
        "mean_run": float(runs.mean()),
        "runs_under_5_bars": int((runs < 5).sum()),
    }


def detection_lag(true_states: ArrayLike, labels: ArrayLike) -> Dict[str, float]:
    """
    Bars from each true regime change until ``labels`` first shows the new
    state (synthetic data only, where the true states are known).

    A change counts as missed if the true state changes again before the
    label catches up. Bars where ``labels`` is NaN are skipped.
    """
    s = np.asarray(true_states, dtype=float)
    lab = np.asarray(labels, dtype=float)
    if s.shape != lab.shape:
        raise ValueError("true_states and labels must have the same length")
    valid = ~np.isnan(lab)
    s, lab = s[valid], lab[valid]

    lags, missed = [], 0
    change_points = np.flatnonzero(s[1:] != s[:-1]) + 1
    for i, t in enumerate(change_points):
        end = change_points[i + 1] if i + 1 < len(change_points) else len(s)
        hit = np.flatnonzero(lab[t:end] == s[t])
        if hit.size:
            lags.append(int(hit[0]))
        else:
            missed += 1
    return {
        "changes": len(change_points),
        "detected": len(lags),
        "missed": missed,
        "mean_lag": float(np.mean(lags)) if lags else np.nan,
        "median_lag": float(np.median(lags)) if lags else np.nan,
    }
