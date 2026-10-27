"""
Whipsaw filter over several synthetic seeds.

One seed is one draw; this runs the demo's comparison (raw vs filtered label,
21- and 5-bar volatility windows) on seeds 0-9 and prints the mean and range
over seeds. Missed changes are summed over all seeds.

    PYTHONPATH=. python examples/whipsaw_seeds.py
"""

import pandas as pd

from examples.demo import (
    CONFIRM_BARS, THRESHOLD, VOL_WINDOWS, label_stats, run_walk_forward,
)
from src.data_loader import fetch_sample_data

SEEDS = range(10)


def main():
    rows = []
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

    res = pd.DataFrame(rows)
    order = list(dict.fromkeys(res["label"]))
    g = res.groupby("label", sort=False)

    def mean_range(col, digits):
        return g[col].agg(lambda x: f"{x.mean():.{digits}f} ({x.min():.{digits}f}-{x.max():.{digits}f})")

    summary = pd.DataFrame({
        "switches": mean_range("switches", 1),
        "accuracy": mean_range("accuracy", 2),
        "lag_to_stress": mean_range("lag_to_stress", 1),
        "missed_stress": g["missed_stress"].sum(),
        "lag_to_calm": mean_range("lag_to_calm", 1),
        "missed_calm": g["missed_calm"].sum(),
    }).loc[order]
    print(f"Seeds {SEEDS.start}-{SEEDS.stop - 1}, filter P >= {THRESHOLD} on {CONFIRM_BARS} bars; "
          f"mean (min-max) over seeds, misses summed; lags in bars, misses excluded.")
    print(summary.to_string())

    diff = res[res["label"].str.contains("filtered")].reset_index(drop=True)
    base = res[res["label"].str.contains("raw")].reset_index(drop=True)
    print("\nFiltered minus raw, per seed (mean, min, max):")
    for vw in VOL_WINDOWS:
        m = diff["label"].str.startswith(f"vol {vw:2d}")
        d = diff.loc[m, ["switches", "accuracy", "lag_to_stress", "lag_to_calm"]].reset_index(drop=True) \
            - base.loc[m, ["switches", "accuracy", "lag_to_stress", "lag_to_calm"]].reset_index(drop=True)
        print(f"vol {vw:2d}: " + ", ".join(
            f"{c} {d[c].mean():+.2f} ({d[c].min():+.2f} to {d[c].max():+.2f})" for c in d))


if __name__ == "__main__":
    main()
