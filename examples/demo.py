"""
Walk-forward regime detection demo.

Every number shown is out-of-sample: the scaler, HMM parameters and the
number of states are estimated only on data before each bar. The whipsaw
filter is then applied causally on top of the filtered probabilities.

Three volatility features are compared on the same synthetic bars: a slow
and a fast rolling volatility, and the per-bar log range variance.

    PYTHONPATH=. python examples/demo.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

from hmm_regime.data_loader import fetch_sample_data, prepare_hmm_features
from hmm_regime.hmm_model import WalkForwardHMM
from hmm_regime.model_selection import evaluate_regime_models
from hmm_regime.whipsaw import add_filtered_regime, detection_lag, switch_stats

WINDOW = 252
STEP = 21
FEATURES = {                       # label -> arguments of prepare_hmm_features
    "vol 21": dict(vol_window=21),     # slow rolling volatility
    "vol  5": dict(vol_window=5),      # fast rolling volatility
    "bar": dict(vol_feature="bar"),    # per-bar log range variance, no window
}
PLOT_FEATURE = "bar"
THRESHOLD = 0.7        # whipsaw filter: a switch needs P >= 0.7 ...
CONFIRM_BARS = 2       # ... on 2 bars in a row


def run_walk_forward(df_raw, feature):
    """Walk-forward on one feature set (a key of FEATURES)."""
    feature_df, _ = prepare_hmm_features(df_raw, **FEATURES[feature])
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
    return wf, oos


def oracle_stress_proba(df_raw):
    """
    P(stress | data up to t) from a forward filter that *knows* the true
    transition matrix, means, volatilities and range distribution of
    ``fetch_sample_data``. No estimated model can be expected to label these
    bars more accurately from the same returns and ranges, so its accuracy
    is the ceiling the walk-forward results should be read against.
    """
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([0.0008, -0.0012])
    sigma = np.array([0.008, 0.025])
    a = np.sqrt(4.0 * np.log(2.0) / (2.0 + 4.0 / np.pi))   # range scale, see fetch_sample_data
    r = np.log(df_raw["Close"] / df_raw["Close"].shift(1)).to_numpy()
    hl = np.log(df_raw["High"] / df_raw["Low"]).to_numpy()

    belief = np.array([5.0 / 7.0, 2.0 / 7.0])              # stationary distribution of A
    out = np.empty(len(df_raw))
    for t in range(len(df_raw)):
        # ln(H/L) = a * sigma * (|z1| + |z2|); the sum s of two half-normals
        # has density (2 / sqrt(pi)) * exp(-s^2 / 4) * erf(s / 2)
        s = hl[t] / (a * sigma)
        like = np.exp(-s ** 2 / 4.0) * erf(s / 2.0) / sigma
        if t > 0:                                           # bar 0 has no return
            like = like * np.exp(-0.5 * ((r[t] - mu) / sigma) ** 2) / sigma
        belief = (belief @ A if t > 0 else belief) * like
        belief = belief / belief.sum()
        out[t] = belief[1]
    return pd.Series(out, index=df_raw.index, name="oracle_prob_stress")


def common_index(runs):
    """Bars that are out-of-sample for every feature set."""
    common = None
    for oos in runs:
        common = oos.index if common is None else common.intersection(oos.index)
    return common


def label_stats(lab, truth):
    """Turnover, accuracy and detection lag by direction. "timed" counts the
    changes the label was tracking; lags are over the detected ones."""
    s = switch_stats(lab)
    onset = detection_lag(truth, lab, into=1)
    back = detection_lag(truth, lab, into=0)
    return {
        "switches": s["switches"],
        "runs_<5": s["runs_under_5_bars"],
        "accuracy": float((lab == truth).mean()),
        "timed_stress": onset["tracked"],
        "missed_stress": onset["missed"],
        "lag_to_stress": onset["mean_lag"],
        "timed_calm": back["tracked"],
        "missed_calm": back["missed"],
        "lag_to_calm": back["mean_lag"],
    }


def main():
    df_raw, true_state = fetch_sample_data(n_bars=1000, seed=42, return_states=True)

    # K comparison on the FIRST training window only (illustration;
    # the walk-forward below re-selects K inside every window).
    for name in ("vol 21", "bar"):
        _, X = prepare_hmm_features(df_raw, **FEATURES[name])
        print(f"AIC/BIC on the first training window, feature '{name}' (the data have 2 regimes):")
        table = evaluate_regime_models(X[:WINDOW], components_range=[1, 2, 3, 4])
        print(table.round(2).to_string(index=False), "\n")

    runs = {name: run_walk_forward(df_raw, name) for name in FEATURES}
    common = common_index([oos for _, oos in runs.values()])

    wf, oos = runs[PLOT_FEATURE]
    print(f"Feature '{PLOT_FEATURE}': refits {len(wf.history_)}  |  K used: "
          f"{oos['n_states'].value_counts().to_dict()}")
    print("Latest transition matrix:")
    print(np.round(wf.history_[-1]["model"].transition_matrix_, 4), "\n")
    print(oos[["close", "regime", "n_states", "prob_stress", "stress_filtered"]].tail(), "\n")

    # Whipsaw filter: fewer switches, paid for with detection lag.
    # Synthetic data only: here the true states are known.
    truth = true_state.loc[common].astype(float)
    rows = {"true state": label_stats(truth, truth)}
    oracle = (oracle_stress_proba(df_raw).loc[common] > 0.5).astype(float)
    rows["true parameters"] = label_stats(oracle, truth)
    for name, (_, o) in runs.items():
        o = o.loc[common]
        rows[f"{name}, raw"] = label_stats((o["prob_stress"] > 0.5).astype(float), truth)
        rows[f"{name}, filtered"] = label_stats(o["stress_filtered"].astype(float), truth)
    print(f"Calm vs stress, {len(common)} out-of-sample bars, "
          f"{switch_stats(truth)['switches']} true changes "
          f"(filter: P >= {THRESHOLD} on {CONFIRM_BARS} bars in a row; lags in bars):")
    print(pd.DataFrame(rows).T.round(2).to_string())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    ax1.plot(oos.index, oos["close"], color="black", lw=1, label="Close")
    y0, y1 = oos["close"].min() * 0.98, oos["close"].max() * 1.02
    ax1.set_ylim(y0, y1)
    stress = (oos["stress_filtered"] == 1).to_numpy()
    ax1.fill_between(oos.index, y0, y1, where=stress, color="red", alpha=0.15,
                     label="Stress (filtered)")
    ax1.fill_between(oos.index, y0, y1, where=~stress, color="green", alpha=0.08,
                     label="Calm (filtered)")
    for h in wf.history_:
        ax1.axvline(h["refit_time"], color="gray", lw=0.3, alpha=0.4)
    ax1.set_title(f"Walk-Forward Regime Detection (out-of-sample, forward filtered, "
                  f"whipsaw filtered, feature '{PLOT_FEATURE}')")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(oos.index, oos["prob_stress"], color="firebrick", lw=1, label="P(stress)")
    ax2.step(oos.index, oos["stress_filtered"].astype(float), where="post",
             color="black", lw=0.8, alpha=0.7, label="Filtered label")
    true_oos = true_state.loc[oos.index].astype(float)
    ax2.fill_between(oos.index, 0, 1, where=(true_oos == 1).to_numpy(), step="post",
                     color="gray", alpha=0.2, label="True stress")
    for level in (THRESHOLD, 1.0 - THRESHOLD):
        ax2.axhline(level, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("P(stress)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regime_detection_plot.png", dpi=150)
    plt.close()
    print("\nSaved regime_detection_plot.png")


if __name__ == "__main__":
    main()
