"""
Data loading and stationary feature engineering module for financial time series.
Computes logarithmic returns, realized volatility, and Parkinson volatility.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


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
    factor = 1.0 / (4.0 * np.log(2.0))
    hl_ratio_sq = (np.log(high / low)) ** 2
    rolling_var = factor * hl_ratio_sq.rolling(window=window).mean()
    parkinson_vol = np.sqrt(rolling_var)

    if annualize:
        parkinson_vol *= np.sqrt(periods_per_year)
    return parkinson_vol


def prepare_hmm_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    high_col: Optional[str] = "High",
    low_col: Optional[str] = "Low",
    vol_window: int = 21,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Transform raw OHLC data into a clean, stationary 2D feature matrix for HMM.

    Features generated:
    1. Log Returns: Captures directional drift and regime shocks.
    2. Rolling Volatility: Captures conditional variance clusters.

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
        Lookback window for volatility estimation.

    Returns
    -------
    feature_df : pd.DataFrame
        DataFrame with aligned, non-NaN features and original index.
    X : np.ndarray of shape (n_samples, n_features)
        NumPy array ready for MarketRegimeHMM.fit().
    """
    data = pd.DataFrame(index=df.index)
    data["log_return"] = compute_log_returns(df[price_col])

    if (
        high_col is not None
        and low_col is not None
        and high_col in df.columns
        and low_col in df.columns
    ):
        data["volatility"] = compute_parkinson_volatility(
            df[high_col], df[low_col], window=vol_window
        )
    else:
        data["volatility"] = compute_realized_volatility(
            data["log_return"], window=vol_window
        )

    # Strictly drop initial rolling window NaNs
    feature_df = data.dropna()
    X = feature_df[["log_return", "volatility"]].to_numpy()

    return feature_df, X


def fetch_sample_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic regime-switching price data for offline testing and CI/CD pipelines.

    Parameters
    ----------
    n_bars : int, default 1000
        Number of synthetic daily bars to generate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV DataFrame.
    """
    np.random.seed(seed)
    # Regime 0: Low vol bull (mu=0.0008, sigma=0.008)
    # Regime 1: High vol bear (mu=-0.0012, sigma=0.025)
    states = np.zeros(n_bars, dtype=int)
    trans_matrix = np.array([[0.98, 0.02], [0.05, 0.95]])

    current_state = 0
    for t in range(1, n_bars):
        current_state = np.random.choice(
            [0, 1], p=trans_matrix[current_state]
        )
        states[t] = current_state

    params = {
        0: {"mu": 0.0008, "sigma": 0.008},
        1: {"mu": -0.0012, "sigma": 0.025},
    }

    returns = np.array([
        np.random.normal(params[s]["mu"], params[s]["sigma"]) for s in states
    ])

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_price = (high + low) / 2.0

    dates = pd.date_range(end="2026-01-01", periods=n_bars, freq="B")
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close},
        index=dates,
    )


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

    Returns
    -------
    pd.DataFrame
        Columns ``log_return`` and ``volatility``, NaNs dropped, ready for
        ``WalkForwardHMM.fit_predict_filtered``.
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

    feature_df, _ = prepare_hmm_features(raw, vol_window=vol_window)
    return feature_df
