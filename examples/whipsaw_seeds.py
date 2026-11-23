"""
Feature sets and the whipsaw filter over several synthetic seeds.

One seed is one draw; this runs the demo's comparison (raw vs filtered label;
21-bar, 5-bar and per-bar volatility features) on a range of seeds and prints
the mean and range over seeds. Timed and missed changes are summed over all
seeds. The lag cost of the filter is also measured on matched changes, the
ones both the raw and the filtered label detect, because the two labels time
different changes. The row "true parameters" is a forward filter that knows
the data generating process: the ceiling for any estimated model.

    PYTHONPATH=. python examples/whipsaw_seeds.py            # seeds 0-9
    PYTHONPATH=. python examples/whipsaw_seeds.py 10 30      # seeds 10-29
    PYTHONPATH=. python examples/whipsaw_seeds.py 0 10 4     # Student-t(4) returns

With a third argument the bars come from ``simulate_regime_ohlc`` with
fat-tailed Student-t returns of that many degrees of freedom (0: Gaussian
returns from the same simulator).
"""

import sys

import pandas as pd

from examples.demo import (
    CONFIRM_BARS, FEATURES, THRESHOLD, common_index, label_stats, oracle_stress_proba,
    run_walk_forward,
)
from hmm_regime.data_loader import fetch_sample_data, simulate_regime_ohlc
from hmm_regime.whipsaw import change_lags

DIRECTIONS = {"to stress": 1, "to calm": 0}


def make_data(seed, student_df):
    if student_df is None:
        return fetch_sample_data(n_bars=1000, seed=seed, return_states=True)
    return simulate_regime_ohlc(n_bars=1000, seed=seed, return_states=True,
                                student_df=student_df or None)


def main(first=0, stop=10, student_df=None):
    seeds = range(first, stop)
    rows, paired = [], []
    for seed in seeds:
        df_raw, true_state = make_data(seed, student_df)
        runs = {name: run_walk_forward(df_raw, name)[1] for name in FEATURES}
        common = common_index(runs.values())
        truth = true_state.loc[common].astype(float)
        rows.append({"seed": seed, "label": "true state", "k3": 0.0, **label_stats(truth, truth)})
        if student_df is None:
            # the ceiling: a forward filter that knows the true parameters
            oracle = (oracle_stress_proba(df_raw).loc[common] > 0.5).astype(float)
            rows.append({"seed": seed, "label": "true parameters", "k3": 0.0,
                         **label_stats(oracle, truth)})
        for name, oos in runs.items():
            o = oos.loc[common]
            k3 = float((o["n_states"] == 3).mean())
            raw = (o["prob_stress"] > 0.5).astype(float)
            filtered = o["stress_filtered"].astype(float)
            rows.append({"seed": seed, "label": f"{name}, raw", "k3": k3, **label_stats(raw, truth)})
            rows.append({"seed": seed, "label": f"{name}, filtered", "k3": k3,
                         **label_stats(filtered, truth)})
            for direction, state in DIRECTIONS.items():
                a = change_lags(truth, raw, into=state)
                b = change_lags(truth, filtered, into=state)
                m = a.merge(b, on="bar", suffixes=("_raw", "_filt"))
                m = m[(m["status_raw"] == "detected") & (m["status_filt"] == "detected")]
                d = m["lag_filt"] - m["lag_raw"]
                paired.append({"seed": seed, "feature": name, "direction": direction,
                               "matched": len(d), "mean": d.mean(), "min": d.min(), "max": d.max()})

    res = pd.DataFrame(rows)
    order = list(dict.fromkeys(res["label"]))
    g = res.groupby("label", sort=False)

    def mean_range(col, digits):
        return g[col].agg(lambda x: f"{x.mean():.{digits}f} ({x.min():.{digits}f}-{x.max():.{digits}f})")

    summary = pd.DataFrame({
        "K=3 share": g["k3"].mean().round(2),
        "switches": mean_range("switches", 1),
        "accuracy": mean_range("accuracy", 2),
        "timed_stress": g["timed_stress"].sum(),
        "missed_stress": g["missed_stress"].sum(),
        "lag_to_stress": mean_range("lag_to_stress", 1),
        "timed_calm": g["timed_calm"].sum(),
        "missed_calm": g["missed_calm"].sum(),
        "lag_to_calm": mean_range("lag_to_calm", 1),
    }).loc[order]
    data = "fetch_sample_data" if student_df is None else (
        f"simulate_regime_ohlc, Student-t({student_df:g}) returns" if student_df
        else "simulate_regime_ohlc, Gaussian returns")
    print(f"Seeds {seeds.start}-{seeds.stop - 1} ({data}), filter P >= {THRESHOLD} on "
          f"{CONFIRM_BARS} bars. Mean (min-max) over seeds; timed and missed changes summed "
          f"over seeds; lags in bars over detected changes.")
    print(summary.to_string())

    base = res[res["label"].str.contains("raw")].reset_index(drop=True)
    filt = res[res["label"].str.contains("filtered")].reset_index(drop=True)
    print("\nFiltered minus raw, per seed: mean (min to max) over seeds")
    for name in FEATURES:
        m = base["label"].str.startswith(name)
        cols = ["switches", "accuracy"]
        d = filt.loc[m, cols].reset_index(drop=True) - base.loc[m, cols].reset_index(drop=True)
        print(f"  {name}: " + ", ".join(
            f"{c} {d[c].mean():+.2f} ({d[c].min():+.2f} to {d[c].max():+.2f})" for c in cols))

    pr = pd.DataFrame(paired)
    print("\nLag cost on matched changes (detected by both labels), filtered minus raw, in bars:")
    for (name, direction), x in pr.groupby(["feature", "direction"], sort=False):
        print(f"  {name} {direction:9}: {x['matched'].sum():3d} changes, per-seed mean "
              f"{x['mean'].min():+.2f} to {x['mean'].max():+.2f}, "
              f"single change {x['min'].min():+.0f} to {x['max'].max():+.0f}")


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    main(
        first=int(args[0]) if len(args) > 0 else 0,
        stop=int(args[1]) if len(args) > 1 else 10,
        student_df=args[2] if len(args) > 2 else None,
    )
