# Audit of version 0.1 (commit `2a83816`) and what 0.2.0 does about it

Date: 2026-11-15. Scope: every file in the repository. Environment: Python 3.13, numpy 2.5.3, pandas 3.0.5, hmmlearn 0.3.3.

## Summary

Version 0.1 had no crash and no look-ahead: its 79 tests pass and the causality tests are sound. The problems were in what the numbers mean.

On the synthetic data, where the true regime is known, the unfiltered calm / stress label of 0.1 was right on 72% of the out-of-sample bars with the default 21-bar volatility feature and on 86–87% with a 5-bar one. A forward filter that knows the true parameters is right on 97–98%. Most of that gap came from two things: the rolling volatility feature, and reading "not state 0" as stress when BIC had picked three states for data with two regimes.

| Accuracy, mean over seeds | 0.1, vol 21 | 0.1, vol 5 | 0.2, vol 21 | 0.2, vol 5 | 0.2, per-bar feature | True parameters |
|---|---|---|---|---|---|---|
| Seeds 0–9 | 0.72 | 0.86 | 0.77 | 0.94 | 0.96 | 0.98 |
| Seeds 10–29 | 0.72 | 0.87 | 0.76 | 0.93 | 0.96 | 0.97 |

Seeds 0–9 were used while the changes were designed; seeds 10–29 were run once, afterwards. `fetch_sample_data`, 1000 bars, window 252, step 21, K chosen from [2, 3] by BIC, `n_init=5`, 728 out-of-sample bars per seed. The 0.2 columns are `examples/whipsaw_seeds.py` and `examples/whipsaw_seeds.py 10 30`; the 0.1 columns are the same walk-forward run on the old commit.

**What these numbers do not show.** They are synthetic. On real prices there is no true state, so accuracy cannot be measured, and nothing here says that a regime label is worth trading. The per-bar feature's advantage also depends on the return distribution: with Student-t(4) returns its unfiltered label is no better than the 5-bar window's (finding 2).

## Findings

Severity: **high** = changes the conclusions drawn from the output; **medium** = wrong in specific situations; **low** = small numerical or usability defect.

### 1. Binary label reads a second calm state as stress — high

`add_filtered_regime(binary=True)` and the README's unfiltered counterpart defined stress as "not state 0". When the walk-forward picks K = 3 for data with two regimes, the calm regime is usually split into two states, and the calmer half alone is "calm".

Evidence (0.1, seeds 0–9): K = 3 in 98% of the windows with the 21-bar feature and 64% with the 5-bar one. Forcing K = 2 raised accuracy from 0.72 to 0.77 and from 0.86 to 0.94.

Fix: `prob_stress`, the summed probability of the states that are more volatile than their training window's occupancy-weighted average (`MarketRegimeHMM.stress_states_`). With K = 2 it is exactly `prob_regime_1`. With K from [2, 3] it reaches the accuracy of fixed K = 2 (0.77 and 0.94), so the choice of K no longer changes what "stress" means. `binary=True` uses it; `stress_from="state0"` gives the old rule.

### 2. The rolling volatility feature violates the model's assumption and adds its window to the lag — high

A Gaussian HMM assumes each bar is an independent draw given the regime. A 21-bar rolling volatility shares 20 of 21 bars with its neighbour. Three consequences, all measured:

- BIC over-selects K (finding 1's 98%), because extra states that slice volatility levels fit the smooth series very well.
- The return to calm is detected about 15–19 bars late with a 21-bar window and 3–4 bars late with a 5-bar one: the window still holds the old regime.
- Accuracy: 0.72–0.77 (21-bar), 0.86–0.94 (5-bar).

Fix: `prepare_hmm_features(vol_feature="bar")`, the log of each bar's own Parkinson variance. One reading per bar, close to Gaussian in logs (Alizadeh, Brandt & Diebold, 2002), smoothing left to the transition matrix. Accuracy 0.96, return to calm about 1 bar late, K = 3 picked in 7% of the windows.

Limits of this fix, also measured:

- With bars built from an intrabar path and Gaussian returns (`simulate_regime_ohlc`): 0.98 against 0.93 (5-bar) and 0.78 (21-bar).
- With Student-t(4) returns: 0.89 against 0.88 (5-bar), and the unfiltered label switches 116 times per seed against 18 true changes, because one extreme bar looks like stress. With the whipsaw filter it is 0.92 against 0.86, at 30 switches. The per-bar feature should be used with the filter.
- Its raw label also switches more on Gaussian data (33 per seed against 20 true changes; the true-parameter filter itself switches 29 times).

The default stays `"rolling"`: changing it would change the output of every existing caller on the strength of synthetic evidence. That decision is open (see STATUS).

### 3. The first bar of a new sequence was fixed in advance — medium

`predict_filtered_proba(X)` without `prior_proba` started from the fitted `startprob_`. EM estimates it from the single first training bar, so it was one-hot in all 426 calibrations checked (exactly zero for the other states in over 99%). The first bar of any new sequence was therefore assigned to the state the training window happened to start in, whatever the bar looked like, and the following bars inherited that until the data overruled it.

Affected: standalone use of `MarketRegimeHMM` on new data (as in `tests/test_lookahead.py`). `WalkForwardHMM` was barely affected: it carries the belief through the whole training window first.

Fix: start from the stationary distribution of the transition matrix. `initial="startprob"` restores the old behaviour. Test: `test_first_bar_of_new_sequence_follows_the_data`.

### 4. "State 0 = calmest" was decided by the trace of all standardised features — medium

States were sorted by the trace of the state covariance in standardised units, i.e. return variance plus the within-state variance of the volatility feature. That is not the volatility of the state. Evidence (0.1): the trace order disagreed with the order by return variance in 2–7% of the calibrations and with the order by mean volatility in 25–47%.

Fix: `sort_by="variance:0"`, the spread of the return around the training-window mean. Measured around the window mean rather than the state's own mean, because a state that holds a few near-identical outlier bars has a tiny variance of its own and would otherwise be labelled the calm state, with the remaining 99% of the bars as stress (`test_state_holding_a_few_outlier_bars_is_stress_not_calm`). On seeds 0–5 with the per-bar feature: accuracy 0.961 with the new order, 0.951 with `"trace"`, 0.939 with `"mean:1"`.

### 5. `min_covar` was not a floor, and the actual regulariser did not bound the covariance — medium

The docstring called `min_covar` a "floor added to covariance diagonals". hmmlearn uses it only to initialise EM. What regularises the M-step is `covars_prior`, a scalar that hmmlearn adds to every element of a full covariance, off-diagonals included: a rank-one term that leaves the smallest eigenvalue free to go to zero, where the likelihood is unbounded and the multi-start search, which keeps the highest likelihood, would select exactly that solution.

In practice it was rare on the test data (a state with fewer than 10 expected bars in 1.4% of the 5-bar calibrations, smallest eigenvalue 0.002), so this is a latent defect more than an observed one.

Fix: `covar_prior` as a diagonal matrix, same strength (0.01). Docstring corrected. Test: `test_covariance_ridge_bounds_a_collapsing_state`.

### 6. The selected K was calibrated twice — low (speed)

`select_n_components` fitted every candidate, returned only K, and `_fit_window` then fitted that K again from scratch. With candidates [2, 3] that is three multi-start fits per window instead of two. Fix: `select_model` returns the fitted winner. Test: `test_walk_forward_calibrates_each_candidate_once`.

### 7. K was selected on rounded scores — low

`evaluate_regime_models` rounded AIC/BIC to two decimals and `select_n_components` picked the minimum of the rounded column. Fix: unrounded values throughout; ties go to the smaller K.

### 8. `fetch_sample_data` reseeded NumPy's global generator — low

`np.random.seed(seed)` inside a library function changes the random stream of the calling program. Fix: a private `RandomState(seed)`, which produces the same stream, so the synthetic data and every number derived from them are unchanged (`test_sample_data_values_unchanged_from_version_01`).

### 9. Inputs were not validated — low

A zero or negative price produced `-inf`/NaN returns that surfaced later as "Input contains NaN or inf"; High < Low, an unsorted index or a duplicated timestamp produced wrong features silently. All four now raise where they occur. A bar whose density underflows in every state turned all later probabilities into NaN; it now keeps the predicted distribution. `prior_proba` with negative entries was accepted if it summed to one.

### 10. No fit diagnostics — low

EM convergence was never checked and nothing reported how many bars the rarest state held. Both are now recorded per refit (`history_`), with one summary warning per walk-forward run. In the seed 0–9 runs every calibration converged.

### 11. Packaging and documentation — low

The code was importable only as `src` with `PYTHONPATH=.`, a name that collides with any other project laid out the same way. Now `hmm_regime`, with `pyproject.toml`; `src/` remains as a thin compatibility layer. README: the BibTeX entry was not valid BibTeX, a reference contained a Turkish fragment, and two sections listed the same limitations.

## Checked and found correct

- **Look-ahead.** Scaler, K, EM parameters and the warm-start belief use bars before the refit only; the forward filter is causal; the whipsaw filter is causal at every cut point. The existing tests establish this and still pass. The new `prob_stress` column and `warm_start` are covered by the same corruption test.
- **Forward filter recursion**, including the carried belief across refits (`test_prior_proba_continues_filter`).
- **Free-parameter count** for AIC/BIC.
- **Parkinson estimator** and the calibration of the synthetic High/Low range.
- **Whipsaw filter logic** (hysteresis, confirmation, dwell, NaN rows, K shrinking) and the lag statistics. No change needed beyond the definition of stress.

## Tried and not adopted

- **Warm-starting EM from the previous window** (`warm_start=True`): accuracy 0.962 against 0.961 on seeds 0–5. No measurable gain; kept as an option, off by default.
- **Rejecting EM solutions with a near-empty state.** It would also reject a correct solution that isolates a short, real stress episode, and replace it with a worse local optimum. Reported as a warning instead.
- **Diagonal covariance as the default**: 0.966 against 0.961 on six seeds, within noise.

## Not done

- No evaluation on real data.
- Student-t emissions, which is the direct answer to finding 2's fat-tail result.
- The oracle benchmark exists only for `fetch_sample_data`; `simulate_regime_ohlc` has no closed-form range density.
- CI was edited but has not run on GitHub.
