# STATUS

Last updated: 2026-10-28 (branch `feature/whipsaw-filter`).

## Where things stand
- Library only: walk-forward Gaussian HMM with a causal forward filter, variance-sorted states, K by AIC/BIC per window. No strategy and no real-data results in this repo.
- New: whipsaw filter `src/whipsaw.py` (hysteresis threshold, confirmation bars, minimum dwell; causal), `switch_stats`, `detection_lag` (tracked changes only, misses and direction reported). 73 tests pass; the whipsaw tests catch ten planted bugs (one-bar look-ahead on both label paths, off-by-one dwell, strict threshold, NaN-gap handling, lag-metric artefacts, the old regime-blind synthetic data).
- Fixed: the synthetic High/Low range ignored the regime, so the Parkinson volatility feature carried no regime information. The range now scales with the regime's sigma.
- Results on synthetic seeds 0–9 (`examples/whipsaw_seeds.py`, table in `README.md`): with a 5-bar volatility feature the filter removes about a quarter of the switches; it detects stress onset about one bar later and costs about 0.02 accuracy.
- Reviewed by an independent audit before the push; its findings were fixed in the second round of commits.

## Next step
- Use the filter in a strategy project only after that strategy's raw edge has passed; fix `threshold`, `confirm_bars` and `min_dwell` in its plan before the backtest and count them as trials.

## Known limitations
- Synthetic `Open = (High + Low) / 2` sits on the same bar's close, so a "fill at next open" backtest on `fetch_sample_data` would fill near the next close. Don't backtest on the synthetic data without fixing this.
- `requirements.txt` allows pandas >= 2.0; tested on pandas 2.3.3 and 3.0.5 only.
- With `binary=False`, a regime label keeps its number when K grows at a refit although the states were re-estimated; `binary=True` avoids this.
