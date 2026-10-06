"""
Walk-forward regime detection demo.

Every number shown is out-of-sample: the scaler, HMM parameters and the
number of states are estimated only on data before each bar.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import WalkForwardHMM
from src.model_selection import evaluate_regime_models

WINDOW = 252
STEP = 21


def main():
    df_raw = fetch_sample_data(n_bars=1000, seed=42)
    feature_df, X = prepare_hmm_features(df_raw, vol_window=21)

    # K comparison on the FIRST training window only (illustration;
    # the walk-forward below re-selects K inside every window).
    print("AIC/BIC on the first training window:")
    print(evaluate_regime_models(X[:WINDOW], components_range=[2, 3, 4]), "\n")

    wf = WalkForwardHMM(
        n_components=[2, 3],
        window_size=WINDOW,
        step_size=STEP,
        covariance_type="full",
        n_init=5,
    )
    out = wf.fit_predict_filtered(feature_df)
    out["close"] = df_raw.loc[out.index, "Close"]

    oos = out.dropna(subset=["prob_regime_0"]).copy()
    oos["prob_stress"] = 1.0 - oos["prob_regime_0"]

    print(f"Refits: {len(wf.history_)}  |  K used: "
          f"{oos['n_states'].value_counts().to_dict()}")
    print("Latest transition matrix:")
    print(np.round(wf.history_[-1]["model"].transition_matrix_, 4), "\n")
    print(oos[["close", "regime", "n_states", "prob_stress"]].tail())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    ax1.plot(oos.index, oos["close"], color="black", lw=1, label="Close")
    y0, y1 = oos["close"].min() * 0.98, oos["close"].max() * 1.02
    ax1.set_ylim(y0, y1)
    stress = (oos["regime"] > 0).to_numpy()
    ax1.fill_between(oos.index, y0, y1, where=stress, color="red", alpha=0.15, label="Higher vol")
    ax1.fill_between(oos.index, y0, y1, where=~stress, color="green", alpha=0.08, label="Calm (state 0)")
    for h in wf.history_:
        ax1.axvline(h["refit_time"], color="gray", lw=0.3, alpha=0.4)
    ax1.set_title("Walk-Forward Regime Detection (out-of-sample, forward filtered)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(oos.index, oos["prob_stress"], color="firebrick", lw=1)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("P(not calm)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regime_detection_plot.png", dpi=150)
    plt.close()
    print("\nSaved regime_detection_plot.png")


if __name__ == "__main__":
    main()
