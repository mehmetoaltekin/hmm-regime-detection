"""
Market Regime Detection Demo
Demonstrates sequential out-of-sample forward filtering and variance-based state sorting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import MarketRegimeHMM


def run_regime_pipeline():
    print("=" * 60)
    print("1. Generating Synthetic Market Data & Stationary Features...")
    print("=" * 60)
    raw_df = fetch_sample_data(n_bars=800, seed=42)
    feature_df, X = prepare_hmm_features(raw_df, vol_window=21)
    print(f"Feature matrix shape: {X.shape}")

    print("\n" + "=" * 60)
    print("2. Fitting Gaussian HMM with Variance-Based State Alignment...")
    print("=" * 60)
    # State 0 is strictly Low Volatility, State 1 is High Volatility
    model = MarketRegimeHMM(n_components=2, covariance_type="full", random_state=42)
    model.fit(X)

    print("\nRegime Transition Matrix:")
    print(model.transition_matrix_)

    print("\n" + "=" * 60)
    print("3. Computing Forward-Filtered Probabilities (Zero Look-Ahead)...")
    print("=" * 60)
    filtered_probs = model.predict_filtered_proba(X)
    regimes = model.predict_regimes(X)

    feature_df["Regime"] = regimes
    feature_df["Prob_HighVol"] = filtered_probs[:, 1]
    feature_df["Close"] = raw_df.loc[feature_df.index, "Close"]

    print("\nLatest 5 Period Detections:")
    print(feature_df[["Close", "log_return", "volatility", "Regime", "Prob_HighVol"]].tail())

    # 4. Visualization
    print("\nGenerating regime visualizer plot...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    dates = feature_df.index
    close_prices = feature_df["Close"]

    ax1.plot(dates, close_prices, color="#1f2937", lw=1.5, label="Asset Close Price")

    high_vol_mask = feature_df["Regime"] == 1
    ax1.fill_between(
        dates, close_prices.min(), close_prices.max(), where=high_vol_mask,
        color="#ef4444", alpha=0.25, label="Regime 1: High Volatility (Shock)"
    )
    ax1.fill_between(
        dates, close_prices.min(), close_prices.max(), where=~high_vol_mask,
        color="#10b981", alpha=0.10, label="Regime 0: Low Volatility (Calm)"
    )

    ax1.set_title("Market Regime Detection (Out-of-Sample Forward Filtered)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    ax2.plot(dates, feature_df["Prob_HighVol"], color="#dc2626", lw=1.2, label="P(S_t = High Vol | Y_{1:t})")
    ax2.axhline(0.5, color="#6b7280", linestyle="--", lw=1, label="Decision Threshold (0.5)")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("regime_detection_plot.png", dpi=300)
    print("Plot saved successfully to 'regime_detection_plot.png'.")


if __name__ == "__main__":
    run_regime_pipeline()
