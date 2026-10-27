"""
Whipsaw filter over several synthetic seeds.

One seed is one draw; this runs the demo's comparison (raw vs filtered label,
21- and 5-bar volatility windows) on seeds 0-9 and prints the mean and range
over seeds. Timed and missed changes are summed over all seeds. The lag cost
of the filter is also measured on matched changes, the ones both the raw and
the filtered label detect, because the two labels time different changes.

    PYTHONPATH=. python examples/whipsaw_seeds.py
"""

import pandas as pd

from examples.demo import (
    CONFIRM_BARS, THRESHOLD, VOL_WINDOWS, label_stats, run_walk_forward,
)
from src.data_loader import fetch_sample_data
from src.whipsaw import change_lags

SEEDS = range(10)
DIRECTIONS = {"to stress": 1, "to calm": 0}


def main():
    rows, paired = [], []
    for seed in SEEDS:
        df_raw, true_state = fetch_sample_data(n_bars=1000, seed=seed, return_states=True)
        runs = {vw: run_walk_forward(df_raw, vw)[1] for vw in VOL_WINDOWS}
        common = runs[VOL_WINDOWS[0]].index
        for oos in runs.values():
            common = common.intersection(oos.index)
        truth = true_state.loc[common].astype(float)
        rows.append({"seed": seed, "label": "true state", **label_stats(truth, truth)})
        for vw, oos in runs.items():
            o = oos.loc[common]
            raw = (o["prob_stress"] > 0.5).astype(float)
            filtered = o["stress_filtered"].astype(float)
            rows.append({"seed": seed, "label": f"vol {vw:2d}, raw", **label_stats(raw, truth)})
            rows.append({"seed": seed, "label": f"vol {vw:2d}, filtered", **label_stats(filtered, truth)})
            for name, state in DIRECTIONS.items():
                a = change_lags(truth, raw, into=state)
                b = change_lags(truth, filtered, into=state)
                m = a.merge(b, on="bar", suffixes=("_raw", "_filt"))
                m = m[(m["status_raw"] == "detected") & (m["status_filt"] == "detected")]
                d = m["lag_filt"] - m["lag_raw"]
                paired.append({"seed": seed, "vol": vw, "direction": name, "matched": len(d),
                               "mean": d.mean(), "min": d.min(), "max": d.max()})

    res = pd.DataFrame(rows)
    order = list(dict.fromkeys(res["label"]))
    g = res.groupby("label", sort=False)

    def mean_range(col, digits):
        return g[col].agg(lambda x: f"{x.mean():.{digits}f} ({x.min():.{digits}f}-{x.max():.{digits}f})")

    summary = pd.DataFrame({
        "switches": mean_range("switches", 1),
        "accuracy": mean_range("accuracy", 2),
        "timed_stress": g["timed_stress"].sum(),
        "missed_stress": g["missed_stress"].sum(),
        "lag_to_stress": mean_range("lag_to_stress", 1),
        "timed_calm": g["timed_calm"].sum(),
        "missed_calm": g["missed_calm"].sum(),
        "lag_to_calm": mean_range("lag_to_calm", 1),
    }).loc[order]
    print(f"Seeds {SEEDS.start}-{SEEDS.stop - 1}, filter P >= {THRESHOLD} on {CONFIRM_BARS} bars. "
          f"Mean (min-max) over seeds; timed and missed changes summed over seeds; "
          f"lags in bars over detected changes.")
    print(summary.to_string())

    base = res[res["label"].str.contains("raw")].reset_index(drop=True)
    filt = res[res["label"].str.contains("filtered")].reset_index(drop=True)
    print("\nFiltered minus raw, per seed: mean (min to max) over seeds")
    for vw in VOL_WINDOWS:
        m = base["label"].str.startswith(f"vol {vw:2d}")
        cols = ["switches", "accuracy"]
        d = filt.loc[m, cols].reset_index(drop=True) - base.loc[m, cols].reset_index(drop=True)
        print(f"  vol {vw:2d}: " + ", ".join(
            f"{c} {d[c].mean():+.2f} ({d[c].min():+.2f} to {d[c].max():+.2f})" for c in cols))

    pr = pd.DataFrame(paired)
    print("\nLag cost on matched changes (detected by both labels), filtered minus raw, in bars:")
    for (vw, name), x in pr.groupby(["vol", "direction"], sort=False):
        print(f"  vol {vw:2d} {name:9}: {x['matched'].sum():3d} changes, per-seed mean "
              f"{x['mean'].min():+.2f} to {x['mean'].max():+.2f}, "
              f"single change {x['min'].min():+.0f} to {x['max'].max():+.0f}")


if __name__ == "__main__":
    main()
