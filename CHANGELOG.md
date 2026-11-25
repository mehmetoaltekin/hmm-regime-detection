# Changelog

## 0.2.1

- Published on PyPI as `hmm-regime` (`pip install hmm-regime`); the import name stays `hmm_regime`. Before, the distribution was named `hmm-regime-detection` and could only be installed from GitHub.
- `.github/workflows/publish.yml`: pushing a tag `vX.Y.Z` that matches `hmm_regime.__version__` builds the package, tests the built wheel and uploads it to PyPI (trusted publishing, no API token in the repository). Run by hand, it uploads to TestPyPI instead.

## 0.2.0 — audit release

Findings, evidence and measurements behind these changes: [`AUDIT.md`](AUDIT.md).

### Changes that alter numbers

Existing code keeps running, but these change output. Each is a correction; the old behaviour stays available where a switch is named.

- **First bar of a new sequence.** `predict_filtered_proba(X)` without `prior_proba` now starts from the stationary distribution of the transition matrix. Before, it started from the fitted `startprob_`, which EM estimates from one bar and which therefore puts all mass on one state: the first bar's state was fixed whatever the data said. Old behaviour: `initial="startprob"`. The walk-forward was only marginally affected (it carries the belief through 252 training bars).
- **State order.** State 0 is the state with the lowest return volatility (`sort_by="variance:0"`: spread of feature 0 around the training-window mean). Before, states were sorted by the trace of the covariance of all standardised features, which mixes the return with whatever else is in the feature matrix. Old behaviour: `sort_by="trace"`.
- **Binary label.** `add_filtered_regime(binary=True)` now reads the new `prob_stress` column (states more volatile than their training window's average). Before, it was "everything except state 0", which calls a second calm state "stress" whenever K = 3. Identical for K = 2. Old behaviour: `stress_from="state0"`.
- **Covariance regulariser.** A ridge on the covariance *diagonal* (`covar_prior`, default 0.01). Before, hmmlearn's scalar prior was added to every element of a full covariance, off-diagonals included, which does not bound the smallest eigenvalue. `min_covar` was documented as a floor but hmmlearn only uses it to initialise EM; the docstring now says so.
- **K selection** compares unrounded AIC/BIC (they were rounded to two decimals first). `evaluate_regime_models` returns unrounded values.

### Added

- `hmm_regime` package, installable with `pip install .` (`pyproject.toml`). `from src... import ...` still works from a checkout.
- `prepare_hmm_features(vol_feature="bar")`: per-bar log range variance instead of a rolling volatility (`compute_log_range_variance`). The default stays `"rolling"`.
- `WalkForwardHMM` output column `prob_stress`; `MarketRegimeHMM.stress_states_`, `stress_proba()`.
- K = 1 as a candidate (`n_components=[1, 2, 3]`): a window with one regime is allowed to say so.
- Fit diagnostics: `converged_`, `n_iter_`, `state_occupancy_`, `aic_`, `bic_`, `n_parameters_`, `warnings_`, `stationary_`; the walk-forward `history_` carries them, and one summary warning is logged per run.
- `select_model()` returns the fitted winner; `WalkForwardHMM(warm_start=True)`; `tol`, `sort_by`, `covar_prior` on `WalkForwardHMM`.
- `simulate_regime_ohlc()`: synthetic bars from an intrabar path (Open = previous Close, consistent High/Low), any number of states, optional Student-t returns.
- `prepare_hmm_features(periods_per_year=...)`; `fetch_returns(vol_feature=..., periods_per_year=...)`.

### Fixed

- The selected K was calibrated twice per window (once to score it, once to use it). Now once.
- `fetch_sample_data` reseeded NumPy's global random generator. It uses a private generator with the same stream; the data are unchanged.
- Non-positive prices, High < Low, an unsorted or duplicated index and a missing price column raise a clear error instead of producing NaN or inf further down.
- A bar that no state can explain numerically (all densities underflow) no longer turns every later probability into NaN.
- `prior_proba` with negative or non-finite entries is rejected; too few samples for K states is rejected with the required number.
- `WalkForwardHMM` could not pass `tol` to the model.
- README: imports, the BibTeX entry, a Turkish fragment in the references.

### Tests

79 → 141, all passing. The unmodified 0.1 test files, importing from `src`, give 78 passed and 1 failed against 0.2: the failing test compares the binary label with `prob_regime_0 < 0.5`, the definition that changed. In this repository that test and two that checked the state order by trace were updated to the new definitions.
