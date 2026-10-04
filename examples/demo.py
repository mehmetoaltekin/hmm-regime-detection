import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import MarketRegimeHMM


def main():
    # pull data & prep features
    df_raw = fetch_sample_data(n_bars=800, seed=42)
    df, X = prepare_hmm_features(df_raw, vol_window=21)

    # 2-state HMM: 0 = low vol, 1 = high vol
    hmm = MarketRegimeHMM(n_components=2, covariance_type="full", random_state=42)
    hmm.fit(X)

    print("Transition Matrix:")
    print(np.round(hmm.transition_matrix_, 4))

    # forward-filtered probs (no lookahead)
    probs = hmm.predict_filtered_proba(X)
    df["regime"] = hmm.predict_regimes(X)
    df["prob_high_vol"] = probs[:, 1]
    df["close"] = df_raw.loc[df.index, "Close"]

    print("\nRecent rows:")
    print(df[["close", "log_return", "volatility", "regime", "prob_high_vol"]].tail())

    # quick inspection plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    ax1.plot(df.index, df["close"], color="black", lw=1, label="Close")

    # shade regimes
    y0, y1 = df["close"].min() * 0.98, df["close"].max() * 1.02
    ax1.set_ylim(y0, y1)

    is_shock = df["regime"] == 1
    ax1.fill_between(df.index, y0, y1, where=is_shock, color="red", alpha=0.15, label="High Vol")
    ax1.fill_between(df.index, y0, y1, where=~is_shock, color="green", alpha=0.08, label="Low Vol")

    ax1.set_title("Market Regime Detection (Forward Filtered)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # prob plot
    ax2.plot(df.index, df["prob_high_vol"], color="firebrick", lw=1)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("P(High Vol)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regime_detection_plot.png", dpi=150)
    plt.close()
    print("Done. Saved to regime_detection_plot.png")


if __name__ == "__main__":
    main()
