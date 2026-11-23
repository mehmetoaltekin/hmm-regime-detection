# STATUS

Last updated: 2026-11-23. Version 0.2.0 (audit release) is merged into `main` (PR #3); CI passes on Python 3.10, 3.11, 3.12 and 3.13, on the PR and on `main` after the merge.

## Where things stand
- Library only: walk-forward Gaussian HMM with a causal forward filter, volatility-sorted states, K by AIC/BIC per window, whipsaw filter. No strategy and no real-data results in this repo.
- 0.2.0 is the result of a code audit: findings and measurements in `AUDIT.md`, the list of changes in `CHANGELOG.md`. The code is now an installable package, `hmm_regime`; `from src...` imports still work from a checkout.
- Main corrections: the first bar of a new sequence no longer starts from EM's one-state `startprob_`; state 0 is the lowest return volatility, not the lowest trace of all standardised features; the binary calm / stress label uses `prob_stress`, which keeps its meaning when K = 3; the covariance regulariser is a ridge on the diagonal; the selected K is calibrated once, not twice.
- New: per-bar volatility feature (`vol_feature="bar"`), K = 1 as a candidate, fit diagnostics in `history_`, `simulate_regime_ohlc` (consistent OHLC, optional fat tails).
- 141 tests pass: locally on Python 3.13 (pandas 3.0.5, numpy 2.5.3) and Python 3.12 (pandas 3.1.1, numpy 2.5.5); in CI on Python 3.10 (pandas 2.3.3, numpy 2.2.6) and 3.11–3.13 (pandas 3.1.1); hmmlearn 0.3.3 everywhere. CI installs the package (`pip install -e ".[dev]"`), imports it outside the checkout and runs the demo.
- `pyproject.toml` gives the license as an SPDX string (`license = "MIT"`, needs setuptools >= 77); the older table form was deprecated and setuptools warned about it on every build.
- Results on synthetic data (`examples/whipsaw_seeds.py`, tables in `README.md`): accuracy of the unfiltered calm / stress label is 0.77 (21-bar volatility), 0.94 (5-bar) and 0.96 (per-bar) on seeds 0–9, against 0.98 for a filter that knows the true parameters; 0.76 / 0.93 / 0.96 on seeds 10–29, which were not looked at while the changes were designed. Version 0.1 gave 0.72 and 0.86–0.87 for the two rolling features on the same seeds.

## Next step
- Decide whether `prepare_hmm_features` should default to `vol_feature="bar"`. It is opt-in for now, because it changes the output of every existing caller and its advantage is measured on synthetic data only.
- Use the filter in a strategy project only after that strategy's raw edge has passed; fix `threshold`, `confirm_bars` and `min_dwell` in its plan before the backtest and count them as trials.

## Known limitations
- No real-data evaluation: there is no true state to score against. The synthetic results say how well the method recovers regimes that exist, not whether markets have such regimes.
- Fat tails: with Student-t(4) returns the per-bar feature's unfiltered label flips 116 times per seed against 18 true changes; it needs the whipsaw filter there. Emissions are Gaussian; a Student-t HMM is not implemented.
- Regimes are relative to each training window (see README, Limitations).
- `fetch_sample_data`: `Open = (High + Low) / 2` sits on the same bar's close, so a "fill at next open" backtest on it would fill near the next close. `simulate_regime_ohlc` does not have this problem.
- `requirements.txt` and `pyproject.toml` allow pandas >= 2.0; 0.2.0 is tested on pandas 2.3.3, 3.0.5 and 3.1.1 only.
- With `binary=False`, a regime label keeps its number when K grows at a refit although the states were re-estimated; `binary=True` avoids this.
- `warm_start=True` is implemented and tested for causality, but showed no accuracy gain on the synthetic seeds; it is off by default.
