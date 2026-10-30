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


def _as_float_1d(x: ArrayLike) -> np.ndarray:
    """1D float array; pandas <NA> (e.g. an Int64 column) becomes NaN."""
    return pd.Series(np.asarray(x, dtype=object)).to_numpy(dtype=float, na_value=np.nan)


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
        ``WalkForwardHMM.fit_predict_filtered``.
    threshold : float in (0, 1]
        Minimum probability of the new state for a switch (``>=``). Use a
        value above 0.5 for real hysteresis: at 0.5 or below a switch can
        happen on a tie, and at 1/K or below every change of the argmax
        passes.
    confirm_bars : int >= 1
        Consecutive bars the same new state must pass ``threshold``.
        1 switches on the first bar that passes.
    min_dwell : int >= 0
        Minimum bars between two switches (0 or 1: no constraint). The
        first valid bar counts as a switch, so the first real switch can
        come no earlier than ``min_dwell`` bars after it.
    n_states : array of shape (n_samples,), optional
        K in use on each bar (the ``n_states`` column). If the held regime
        does not exist in the current model (K shrank at a refit), the filter
        moves to the argmax at once, ignoring the other rules; this counts as
        a switch for ``min_dwell``. Pass it for
        walk-forward output: without it, a held state that disappeared keeps
        its label until another state passes the threshold.

    Returns
    -------
    np.ndarray of float, shape (n_samples,)
        Regime per bar, NaN where a row of ``probs`` contains NaN. The filter
        starts at the argmax of the first valid row. A NaN row later on keeps
        the held regime (it is no evidence of a change), breaks a running
        confirmation and counts toward ``min_dwell``. Every switch goes to
        the bar's argmax, so the filtered label never switches more often
        than the raw argmax of the same probabilities.

    Notes
    -----
    When K grows at a refit, the held label keeps its number although the
    variance-sorted states were re-estimated, and a running confirmation
    continues. ``add_filtered_regime(..., binary=True)`` avoids this.
    """
    P = np.asarray(probs, dtype=float)
    if P.ndim != 2:
        raise ValueError("probs must be 2D (n_samples, K)")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if int(confirm_bars) != confirm_bars or int(min_dwell) != min_dwell:
        raise ValueError("confirm_bars and min_dwell must be integers")
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
            pending, count = -1, 0
            if current >= 0:
                since_switch += 1
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
    is 0 (calm, state 0) or 1 (any higher-volatility state); its unfiltered
    counterpart is ``prob_regime_0 < 0.5``, not ``regime > 0`` (with K >= 3
    the argmax can be state 0 while P(calm) < 0.5).
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
    r = pd.Series(_as_float_1d(regimes)).dropna().reset_index(drop=True)
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


def change_lags(
    true_states: ArrayLike,
    labels: ArrayLike,
    into: Optional[int] = None,
) -> pd.DataFrame:
    """
    One row per true regime change: ``bar`` (position of the first bar of
    the new regime), ``state`` (the new state), ``status`` and ``lag``.

    Only changes the label was tracking are timed: on the bar before the
    change the label must show the old state. ``status`` is

    * ``detected``  - the label shows the new state before the true state
                      changes again; ``lag`` is the number of bars until then
                      (0 = on the bar of the change), NaN labels included.
    * ``missed``    - the true state changed again first.
    * ``censored``  - the data ended first.
    * ``untracked`` - the label was NaN on the bar before, or showed another
                      state (it had missed the previous regime, or moved early
                      on a false alarm); accuracy covers those bars.

    A one-bar flicker of the label counts as a detection, which favours an
    unfiltered label in comparisons. Synthetic data only, where the true
    states are known.

    Parameters
    ----------
    true_states, labels : arrays of the same length
    into : int, optional
        Keep only changes into this state, e.g. 1 for the onset of stress
        and 0 for the return to calm. Lags differ a lot by direction.
    """
    s = _as_float_1d(true_states)
    lab = _as_float_1d(labels)
    if s.shape != lab.shape:
        raise ValueError("true_states and labels must have the same length")
    if np.isnan(s).any():
        raise ValueError("true_states must not contain NaN")

    rows = []
    change_points = np.flatnonzero(s[1:] != s[:-1]) + 1
    for i, t in enumerate(change_points):
        if into is not None and s[t] != into:
            continue
        if not lab[t - 1] == s[t - 1]:
            rows.append((t, s[t], "untracked", np.nan))
            continue
        end = change_points[i + 1] if i + 1 < len(change_points) else len(s)
        hit = np.flatnonzero(lab[t:end] == s[t])
        if hit.size:
            rows.append((t, s[t], "detected", float(hit[0])))
        else:
            rows.append((t, s[t], "censored" if end == len(s) else "missed", np.nan))
    return pd.DataFrame(rows, columns=["bar", "state", "status", "lag"]).astype(
        {"bar": int, "state": float, "status": object, "lag": float}
    )


def detection_lag(
    true_states: ArrayLike,
    labels: ArrayLike,
    into: Optional[int] = None,
) -> Dict[str, float]:
    """
    Summary of ``change_lags``: counts by status and the mean / median lag
    of detected changes. The lags exclude misses and untracked changes, so
    read them with ``missed`` and ``tracked``; to compare two labels, pair
    their ``change_lags`` on ``bar`` instead.
    """
    d = change_lags(true_states, labels, into=into)
    n = d["status"].value_counts()
    lags = d.loc[d["status"] == "detected", "lag"]
    return {
        "changes": len(d),
        "tracked": len(d) - int(n.get("untracked", 0)),
        "detected": len(lags),
        "missed": int(n.get("missed", 0)),
        "censored": int(n.get("censored", 0)),
        "untracked": int(n.get("untracked", 0)),
        "mean_lag": float(lags.mean()) if len(lags) else np.nan,
        "median_lag": float(lags.median()) if len(lags) else np.nan,
    }
