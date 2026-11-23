"""
Data loading and stationary feature engineering module for financial time series.
Computes logarithmic returns, realized volatility, Parkinson volatility and the
per-bar log range variance.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

VOL_FEATURES = ("rolling", "bar")


def _check_positive(series: pd.Series, name: str) -> None:
    """Prices must be > 0 where present; NaN (a missing bar) is allowed."""
    values = np.asarray(series, dtype=float)
    bad = ~np.isnan(values) & ~(values > 0)
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum())} non-positive or infinite value(s); "
            f"log returns need strictly positive prices."
        )
    if np.isinf(values).any():
        raise ValueError(f"{name} contains inf.")


def _check_high_low(high: pd.Series, low: pd.Series) -> None:
    _check_positive(high, "high")
    _check_positive(low, "low")
    h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    bad = h < l  # False where either is NaN
    if bad.any():
        raise ValueError(f"high < low on {int(bad.sum())} bar(s); check the columns.")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series
        Asset close or adjusted close price series.

    Returns
    -------
    pd.Series
        Log return series.
    """
    _check_positive(prices, "prices")
    return np.log(prices / prices.shift(1))


def compute_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Compute backward-looking rolling realized volatility.

    Parameters
    ----------
    returns : pd.Series
        Log return series.
    window : int, default 21
        Rolling window size in trading periods (e.g., 21 days ~ 1 month).
    annualize : bool, default True
        If True, annualizes the standard deviation.
    periods_per_year : int, default 252
        Number of trading days per year.

    Returns
    -------
    pd.Series
        Rolling standard deviation of returns.
    """
    rolling_std = returns.rolling(window=window).std()
    if annualize:
        rolling_std *= np.sqrt(periods_per_year)
    return rolling_std


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Compute Parkinson volatility based on High-Low price range.
    More efficient estimator of volatility than close-to-close returns.

    Formula:
        sigma_P = sqrt( 1 / (4 * ln(2) * window) * sum( ln(H_t / L_t)^2 ) )

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    window : int, default 21
        Rolling window size.
    annualize : bool, default True
        If True, scales to annualized volatility.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    pd.Series
        Rolling Parkinson volatility series.
    """
    _check_high_low(high, low)
    factor = 1.0 / (4.0 * np.log(2.0))
    hl_ratio_sq = (np.log(high / low)) ** 2
    rolling_var = factor * hl_ratio_sq.rolling(window=window).mean()
    parkinson_vol = np.sqrt(rolling_var)

    if annualize:
        parkinson_vol *= np.sqrt(periods_per_year)
    return parkinson_vol


def compute_log_range_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Per-bar log Parkinson variance: ln( ln(H_t / L_t)^2 / (4 ln 2) ).

    One value per bar, no rolling window. The log of the range is close to
    Gaussian (Alizadeh, Brandt & Diebold, 2002), and a change of volatility
    shifts its mean while its spread stays about the same, which is what a
    Gaussian HMM assumes. A rolling volatility breaks that assumption: its
    overlapping windows make neighbouring bars almost identical, the model
    reads that as certainty, and the window still holds the old regime for
    ``window`` bars after a change.

    A bar with High == Low has no range (ln 0); it takes the last earlier
    value, which uses past data only.
    """
    _check_high_low(high, low)
    hl = np.log(high / low)
    var = hl ** 2 / (4.0 * np.log(2.0))
    with np.errstate(divide="ignore"):
        out = np.log(var.where(var > 0))
    return out.ffill()


def prepare_hmm_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    high_col: Optional[str] = "High",
    low_col: Optional[str] = "Low",
    vol_window: int = 21,
    vol_feature: str = "rolling",
    periods_per_year: int = 252,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Transform raw OHLC data into a clean, stationary 2D feature matrix for HMM.

    Features generated:
    1. Log Returns: Captures directional drift and regime shocks.
    2. Volatility: Captures conditional variance clusters. Either a rolling
       estimate or a per-bar one, see ``vol_feature``.

    Parameters
    ----------
    df : pd.DataFrame
        Market data frame containing price columns.
    price_col : str, default 'Close'
        Column name for closing prices.
    high_col : Optional[str], default 'High'
        Column name for high prices (used for Parkinson volatility if present).
    low_col : Optional[str], default 'Low'
        Column name for low prices.
    vol_window : int, default 21
        Lookback window for volatility estimation (``vol_feature="rolling"``).
    vol_feature : {"rolling", "bar"}, default "rolling"
        * "rolling" - annualised volatility over the last ``vol_window`` bars
          (Parkinson if High/Low are present, else close-to-close). Column
          ``volatility``.
        * "bar" - per-bar log range variance, ``compute_log_range_variance``.
          Needs High/Low. Column ``log_range_var``. On the synthetic data it
          detects regimes more accurately and returns to calm much sooner
          than a rolling window (see README); ``vol_window`` is not used.
    periods_per_year : int, default 252
        Annualisation factor of the rolling volatility: 252 for daily bars,
        365 for daily crypto, bars per year for intraday data. It only
        changes the reported level; the HMM standardises its inputs.

    Returns
    -------
    feature_df : pd.DataFrame
        DataFrame with aligned, non-NaN features and original index.
    X : np.ndarray of shape (n_samples, n_features)
        NumPy array ready for MarketRegimeHMM.fit().
    """
    if vol_feature not in VOL_FEATURES:
        raise ValueError(f"vol_feature must be one of {VOL_FEATURES}, got {vol_feature!r}")
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} not in the data frame")
    if not df.index.is_unique:
        raise ValueError("The index has duplicate entries; one row per bar is required.")
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        raise ValueError("The index is not sorted in time; sort it first (df.sort_index()).")

    has_range = (
        high_col is not None
        and low_col is not None
        and high_col in df.columns
        and low_col in df.columns
    )
    data = pd.DataFrame(index=df.index)
    data["log_return"] = compute_log_returns(df[price_col])

    if vol_feature == "bar":
        if not has_range:
            raise ValueError('vol_feature="bar" needs High and Low columns.')
        vol_col = "log_range_var"
        data[vol_col] = compute_log_range_variance(df[high_col], df[low_col])
    else:
        vol_col = "volatility"
        if has_range:
            data[vol_col] = compute_parkinson_volatility(
                df[high_col], df[low_col], window=vol_window,
                periods_per_year=periods_per_year,
            )
        else:
            data[vol_col] = compute_realized_volatility(
                data["log_return"], window=vol_window,
                periods_per_year=periods_per_year,
            )

    # Strictly drop initial rolling window NaNs
    feature_df = data.dropna()
    X = feature_df[["log_return", vol_col]].to_numpy()

    return feature_df, X


def fetch_sample_data(
    n_bars: int = 1000, seed: int = 42, return_states: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Generate synthetic regime-switching price data for offline testing and CI/CD pipelines.

    Parameters
    ----------
    n_bars : int, default 1000
        Number of synthetic daily bars to generate.
    seed : int, default 42
        Random seed for reproducibility.
    return_states : bool, default False
        If True, also return the true hidden states (0 = calm, 1 = stress)
        as a Series on the same index, e.g. to measure detection lag.

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV DataFrame, or ``(DataFrame, states)`` if
        ``return_states`` is True.
    """
    # A private generator with the same stream as np.random.seed(seed): the
    # data are unchanged, but the caller's global random state is left alone.
    rng = np.random.RandomState(seed)
    # Regime 0: Low vol bull (mu=0.0008, sigma=0.008)
    # Regime 1: High vol bear (mu=-0.0012, sigma=0.025)
    states = np.zeros(n_bars, dtype=int)
    trans_matrix = np.array([[0.98, 0.02], [0.05, 0.95]])

    current_state = 0
    for t in range(1, n_bars):
        current_state = rng.choice(
            [0, 1], p=trans_matrix[current_state]
        )
        states[t] = current_state

    params = {
        0: {"mu": 0.0008, "sigma": 0.008},
        1: {"mu": -0.0012, "sigma": 0.025},
    }

    returns = np.array([
        rng.normal(params[s]["mu"], params[s]["sigma"]) for s in states
    ])

    close = 100.0 * np.exp(np.cumsum(returns))
    # The bar range scales with the regime's sigma, so the Parkinson
    # volatility feature carries the regime. With ln(H/L) = a*sigma*(|z1|+|z2|),
    # E[ln(H/L)^2] = 4 ln2 sigma^2 when a^2 (2 + 4/pi) = 4 ln2.
    sigma = np.array([params[s]["sigma"] for s in states])
    a = np.sqrt(4.0 * np.log(2.0) / (2.0 + 4.0 / np.pi))
    high = close * np.exp(a * sigma * np.abs(rng.normal(0, 1.0, n_bars)))
    low = close * np.exp(-a * sigma * np.abs(rng.normal(0, 1.0, n_bars)))
    open_price = (high + low) / 2.0

    dates = pd.date_range(end="2026-01-01", periods=n_bars, freq="B")
    df = pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close},
        index=dates,
    )
    if return_states:
        return df, pd.Series(states, index=dates, name="state")
    return df


def simulate_regime_ohlc(
    n_bars: int = 1000,
    seed: int = 42,
    transmat: Sequence[Sequence[float]] = ((0.98, 0.02), (0.05, 0.95)),
    mu: Sequence[float] = (0.0008, -0.0012),
    sigma: Sequence[float] = (0.008, 0.025),
    student_df: Optional[float] = None,
    intrabar_steps: int = 48,
    return_states: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Synthetic regime-switching OHLC bars built from an intrabar price path.

    Unlike ``fetch_sample_data`` the four prices of a bar are consistent:
    Open is the previous Close, the path moves in ``intrabar_steps`` steps,
    High / Low are its extremes and Close its last value. A "fill at the
    next open" is therefore a real next-bar price here.

    Parameters
    ----------
    n_bars, seed : int
    transmat : K x K row-stochastic matrix of the hidden Markov chain.
    mu, sigma : per-state mean and standard deviation of the bar's log return.
    student_df : float > 2, optional
        If given, bar returns are Student-t with this many degrees of
        freedom (fat tails) and the same per-state standard deviation:
        each bar draws its own volatility multiplier. Use it to see how a
        Gaussian HMM copes with returns it does not assume.
    intrabar_steps : int
        Steps of the intrabar path. With few steps the path misses part of
        the true range, so Parkinson volatility reads low.
    return_states : bool
        Also return the true hidden states (0 .. K-1) on the same index.
    """
    A = np.asarray(transmat, dtype=float)
    mu_ = np.asarray(mu, dtype=float)
    sigma_ = np.asarray(sigma, dtype=float)
    K = len(A)
    if A.shape != (K, K) or not np.allclose(A.sum(axis=1), 1.0) or A.min() < 0:
        raise ValueError("transmat must be a square row-stochastic matrix")
    if mu_.shape != (K,) or sigma_.shape != (K,) or sigma_.min() <= 0:
        raise ValueError("mu and sigma need one value per state, sigma > 0")
    if student_df is not None and student_df <= 2:
        raise ValueError("student_df must be > 2 (finite variance)")
    if intrabar_steps < 1 or n_bars < 1:
        raise ValueError("n_bars and intrabar_steps must be >= 1")

    rng = np.random.default_rng(seed)
    states = np.zeros(n_bars, dtype=int)
    u = rng.random(n_bars)
    cum = np.cumsum(A, axis=1)
    for t in range(1, n_bars):
        states[t] = min(int(np.searchsorted(cum[states[t - 1]], u[t])), K - 1)

    bar_sigma = sigma_[states]
    if student_df is not None:
        # Student-t as a normal scale mixture, rescaled to unit variance
        w = (student_df - 2.0) / rng.chisquare(student_df, n_bars)
        bar_sigma = bar_sigma * np.sqrt(w)

    m = intrabar_steps
    steps = rng.normal(size=(n_bars, m)) * (bar_sigma / np.sqrt(m))[:, None]
    steps += (mu_[states] / m)[:, None]
    path = np.cumsum(steps, axis=1)                 # log price relative to the open
    bar_return = path[:, -1]
    log_open = np.log(100.0) + np.concatenate([[0.0], np.cumsum(bar_return)[:-1]])

    dates = pd.date_range(end="2026-01-01", periods=n_bars, freq="B")
    df = pd.DataFrame(
        {
            "Open": np.exp(log_open),
            "High": np.exp(log_open + np.maximum(path.max(axis=1), 0.0)),
            "Low": np.exp(log_open + np.minimum(path.min(axis=1), 0.0)),
            "Close": np.exp(log_open + bar_return),
        },
        index=dates,
    )
    if return_states:
        return df, pd.Series(states, index=dates, name="state")
    return df


# Common FX / metal symbols mapped to Yahoo Finance tickers.
_YAHOO_ALIASES: Dict[str, str] = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDTRY": "TRY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}


def fetch_returns(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    vol_window: int = 21,
    vol_feature: str = "rolling",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Download OHLC data from Yahoo Finance and return stationary HMM features.

    Requires the optional ``yfinance`` package (``pip install yfinance``).
    Common symbols such as "XAUUSD" are mapped to their Yahoo tickers.

    Parameters
    ----------
    ticker : str
        Symbol, e.g. "SPY", "XAUUSD", "BTCUSD".
    start, end : str, optional
        Date bounds, e.g. "2015-01-01".
    interval : str, default "1d"
        Bar size passed to yfinance.
    vol_window : int, default 21
        Volatility lookback.
    vol_feature, periods_per_year
        Passed to ``prepare_hmm_features``.

    Returns
    -------
    pd.DataFrame
        Columns ``log_return`` and ``volatility`` (or ``log_range_var``),
        NaNs dropped, ready for ``WalkForwardHMM.fit_predict_filtered``.

    Notes
    -----
    During market hours Yahoo's last row is the bar still in progress: its
    Close, High and Low change until the session ends. For a live signal
    drop that row, or the label of "today" is not final.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fetch_returns needs yfinance: pip install yfinance"
        ) from exc

    symbol = _YAHOO_ALIASES.get(ticker.upper(), ticker)
    raw = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r} ({symbol}).")

    # Newer yfinance versions return a (field, ticker) MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    feature_df, _ = prepare_hmm_features(
        raw, vol_window=vol_window, vol_feature=vol_feature,
        periods_per_year=periods_per_year,
    )
    return feature_df
