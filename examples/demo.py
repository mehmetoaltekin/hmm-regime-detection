"""
Walk-forward regime detection demo.

Every number shown is out-of-sample: the scaler, HMM parameters and the
number of states are estimated only on data before each bar. The whipsaw
filter is then applied causally on top of the filtered probabilities.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import fetch_sample_data, prepare_hmm_features
from src.hmm_model import WalkForwardHMM
from src.model_selection import evaluate_regime_models
from src.whipsaw import add_filtered_regime, detection_lag, switch_stats

WINDOW = 252
STEP = 21
VOL_WINDOWS = (21, 5)  # slow and fast volatility feature
PLOT_VOL_WINDOW = 5    # the fast feature flips more, so the filter matters there
THRESHOLD = 0.7        # whipsaw filter: a switch needs P >= 0.7 ...
CONFIRM_BARS = 2       # ... on 2 bars in a row


def run_walk_forward(df_raw, vol_window):
    feature_df, _ = prepare_hmm_features(df_raw, vol_window=vol_window)
    wf = WalkForwardHMM(
        n_components=[2, 3],
        window_size=WINDOW,
        step_size=STEP,
        covariance_type="full",
        n_init=5,
    )
    out = wf.fit_predict_filtered(feature_df)
    out = add_filtered_regime(out, binary=True, threshold=THRESHOLD, confirm_bars=CONFIRM_BARS)
    out["close"] = df_raw.loc[out.index, "Close"]
    oos = out.dropna(subset=["prob_regime_0"]).copy()
    oos["prob_stress"] = 1.0 - oos["prob_regime_0"]
    return wf, oos


def label_stats(lab, truth):
    s = switch_stats(lab)
    d = detection_lag(truth, lab)
    return {
        "switches": s["switches"],
        "per_year": s["switches_per_year"],
        "mean_run": s["mean_run"],
        "runs_<5": s["runs_under_5_bars"],
        "accuracy": float((lab == truth).mean()),
        "mean_lag": d["mean_lag"],
        "missed": d["missed"],
    }


def main():
    df_raw, true_state = fetch_sample_data(n_bars=1000, seed=42, return_states=True)
    _, X = prepare_hmm_features(df_raw, vol_window=21)

    # K comparison on the FIRST training window only (illustration;
    # the walk-forward below re-selects K inside every window).
    print("AIC/BIC on the first training window:")
    print(evaluate_regime_models(X[:WINDOW], components_range=[2, 3, 4]), "\n")

    runs = {vw: run_walk_forward(df_raw, vw) for vw in VOL_WINDOWS}
    common = runs[VOL_WINDOWS[0]][1].index
    for _, oos in runs.values():
        common = common.intersection(oos.index)

    wf, oos = runs[PLOT_VOL_WINDOW]
    print(f"Volatility window {PLOT_VOL_WINDOW}: refits {len(wf.history_)}  |  K used: "
          f"{oos['n_states'].value_counts().to_dict()}")
    print("Latest transition matrix:")
    print(np.round(wf.history_[-1]["model"].transition_matrix_, 4), "\n")
    print(oos[["close", "regime", "n_states", "prob_stress", "stress_filtered"]].tail(), "\n")

    # Whipsaw filter: fewer switches, paid for with detection lag.
    # Synthetic data only: here the true states are known.
    truth = true_state.loc[common].astype(float)
    rows = {"true state": label_stats(truth, truth)}
    for vw, (_, o) in runs.items():
        o = o.loc[common]
        rows[f"vol {vw:2d}, raw"] = label_stats((o["prob_stress"] > 0.5).astype(float), truth)
        rows[f"vol {vw:2d}, filtered"] = label_stats(o["stress_filtered"].astype(float), truth)
    print(f"Calm vs not calm, {len(common)} out-of-sample bars "
          f"(filter: P >= {THRESHOLD} on {CONFIRM_BARS} bars in a row):")
    print(pd.DataFrame(rows).T.round(2).to_string())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    ax1.plot(oos.index, oos["close"], color="black", lw=1, label="Close")
    y0, y1 = oos["close"].min() * 0.98, oos["close"].max() * 1.02
    ax1.set_ylim(y0, y1)
    stress = (oos["stress_filtered"] == 1).to_numpy()
    ax1.fill_between(oos.index, y0, y1, where=stress, color="red", alpha=0.15,
                     label="Not calm (filtered)")
    ax1.fill_between(oos.index, y0, y1, where=~stress, color="green", alpha=0.08,
                     label="Calm (filtered)")
    for h in wf.history_:
        ax1.axvline(h["refit_time"], color="gray", lw=0.3, alpha=0.4)
    ax1.set_title(f"Walk-Forward Regime Detection (out-of-sample, forward filtered, "
                  f"whipsaw filtered, volatility window {PLOT_VOL_WINDOW})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(oos.index, oos["prob_stress"], color="firebrick", lw=1, label="P(not calm)")
    ax2.step(oos.index, oos["stress_filtered"].astype(float), where="post",
             color="black", lw=0.8, alpha=0.7, label="Filtered label")
    for level in (THRESHOLD, 1.0 - THRESHOLD):
        ax2.axhline(level, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("P(not calm)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regime_detection_plot.png", dpi=150)
    plt.close()
    print("\nSaved regime_detection_plot.png")


if __name__ == "__main__":
    main()
