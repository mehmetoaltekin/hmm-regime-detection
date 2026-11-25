# hmm-regime-detection
Quantitative market regime identification via Gaussian HMMs: walk-forward parameter estimation, state sorting, and information criteria (AIC/BIC) selection.

## Disclaimer
This repository is strictly for academic and quantitative research purposes. 
Nothing contained herein constitutes financial, investment, or trading advice. 
The authors accept no responsibility for any financial losses incurred through 
the use of these models or strategies.

## References

* **Alizadeh, S., Brandt, M. W., & Diebold, F. X. (2002).** *Range-Based Estimation of Stochastic Volatility Models.* The Journal of Finance, 57(3), 1047–1091.
* **Hamilton, J. D. (1989).** *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica, 57(2), 357–384.
* **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* John Wiley & Sons. (Chapter: Structural Breaks).
* **Parkinson, M. (1980).** *The Extreme Value Method for Estimating the Variance of the Rate of Return.* The Journal of Business, 53(1), 61–65.
* **Rabiner, L. R. (1989).** *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* Proceedings of the IEEE, 77(2), 257–286.

# Market Regime Detection via Walk-Forward Gaussian HMM
[![CI Status](https://github.com/mehmetoaltekin/hmm-regime-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/mehmetoaltekin/hmm-regime-detection/actions)

An econometric and quantitative research framework for out-of-sample market regime detection: rolling re-calibration, causal forward filtering, variance-sorted state labels and per-window model selection.

## 1. Theoretical Framework

Financial return series rarely follow a single stationary distribution. In practice, markets alternate between distinct market regimes shifting rapidly between tranquil periods and sudden crisis states. To capture these structural shifts and the resulting time varying volatility, we model returns using a Hidden Markov Model (HMM).
Let $Y_t$ denote the observed return at time $t$. We assume the data generating process is driven by an unobserved discrete Markov chain $S_t \in \{1, \dots, K\}$, with state transition probabilities given by:

$$P(S_t = j \mid S_{t-1} = i) = A_{ij}$$ 

where $A$ is the $K \times K$ transition probability matrix. Conditional on being in regime $S_t = k$, observations are drawn from a Gaussian distribution: 

$$Y_t \mid (S_t = k) \sim \mathcal{N}(\mu_k, \sigma_k^2)$$ 


# Methodological & Implementation Details 

Applying regime switching models to sequential financial data introduces three specific estimation challenges that require structural safeguards:

Preventing Look Ahead Bias: Full sample inference techniques (such as the Viterbi algorithm or two sided smoothing) calculate smoothed probabilities $P(S_t \mid Y_{1:T})$. Because this conditions state assignment at time $t$ on future data ($t+1 \dots T$), it creates severe look ahead bias in backtests.

To maintain strict out of sample validity, we use only the forward algorithm to compute real time filtered probabilities:

$$P(S_t \mid Y_{1:t})$$ 

Resolving Label Switching: Because Expectation Maximization (EM / Baum-Welch) is an unsupervised procedure, state labels are arbitrary and can switch unpredictably between rolling estimation windows.
To ensure economic consistency over time, we sort and re index the regimes by their estimated return variance ($\sigma_k^2$) at each calibration step, pinning State 0 to the lowest volatility regime.

Model Selection ($K$): The number of hidden states is selected by balancing fit against parameter bloat using the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC):

$$\text{BIC} = -2\ln(\hat{L}) + p \ln(T)$$ 

where $\hat{L}$ is the maximized likelihood, $p$ is the number of free parameters, and $T$ is the number of observations.


## 2. Installation & Quickstart

To use the library in your own project:

```bash
pip install hmm-regime                # core: numpy, pandas, hmmlearn
pip install "hmm-regime[data]"        # also yfinance, for fetch_returns()
```

The package is imported as `hmm_regime`. The latest unreleased code: `pip install git+https://github.com/mehmetoaltekin/hmm-regime-detection.git`.

To run the examples and tests:

```bash
git clone https://github.com/mehmetoaltekin/hmm-regime-detection.git
cd hmm-regime-detection
pip install -e ".[dev]"
PYTHONPATH=. python examples/demo.py            # offline demo on synthetic data, with the whipsaw filter
PYTHONPATH=. python examples/whipsaw_seeds.py   # feature sets and whipsaw filter over seeds 0-9 (several minutes)
python -m pytest tests -v                       # look-ahead & regression tests
```

### Minimal usage

```python
from hmm_regime import WalkForwardHMM, add_filtered_regime, fetch_returns

# 1. Stationary features: log return + per-bar log range variance
features = fetch_returns("XAUUSD", start="2015-01-01", end="2024-01-01", vol_feature="bar")

# 2. Walk-forward HMM: refit every 21 bars on the last 252 bars.
#    Passing a list lets BIC choose K inside each training window.
model = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=21)

# 3. Out-of-sample filtered probabilities P(S_t | Y_1:t)
out = model.fit_predict_filtered(features)
out[["prob_regime_0", "prob_regime_1", "regime", "n_states", "prob_stress"]].tail()

# 4. A calm / stress label that does not flip on every noisy bar
out = add_filtered_regime(out, binary=True)        # adds stress_filtered: 0 calm, 1 stress
```

Offline alternative: `fetch_sample_data()` + `prepare_hmm_features()` (see `examples/demo.py`).

Code written against version 0.1 (`from src.hmm_model import WalkForwardHMM`) still runs from a checkout; [`CHANGELOG.md`](https://github.com/mehmetoaltekin/hmm-regime-detection/blob/main/CHANGELOG.md) lists what changed in the numbers and how to get the old behaviour, and [`AUDIT.md`](https://github.com/mehmetoaltekin/hmm-regime-detection/blob/main/AUDIT.md) the findings behind it.

### How the walk-forward avoids look-ahead

At every refit time *t* the following uses only bars `< t`:

| Step | What is estimated | Data used |
|---|---|---|
| Feature scaling | mean / std of each feature | training window |
| Model selection | K by AIC/BIC (if a list is given) | training window |
| EM calibration | start, transition, means, covariances (best of `n_init` restarts) | training window |
| Filter warm-start | state belief at *t-1* | training window, forward filter |
| Output | P(S_t \| Y_1:t) for bars *t … t+step-1* | causal forward filter |

`tests/test_fixes.py::test_walk_forward_has_no_lookahead` corrupts all data after a cutoff and asserts every earlier output is unchanged.

### Implementation notes

* **Scaling.** Raw returns (~0.01) and annualised volatility (~0.2) differ by an order of magnitude, which makes full covariance matrices near-singular. Features are standardised with training-window statistics; `MarketRegimeHMM.means_` / `.covars_` report parameters back in original units.
* **Multi-start EM.** `n_init` (default 10) independent EM runs; the highest log-likelihood is kept.
* **Continuous filtering.** `predict_filtered_proba(X, prior_proba=...)` carries the previous belief through the transition matrix instead of restarting from `startprob_`, so there is no artificial jump at each refit.
* **First bar.** Without a `prior_proba` the filter starts from the stationary distribution of the transition matrix. EM estimates `startprob_` from the single first training bar, so it puts all its mass on one state and would fix the state of the first bar whatever the data say.
* **State order.** State 0 is the state with the lowest return volatility: the spread of feature 0 (the return) around the training-window mean (`sort_by="variance:0"`). Alternatives: `"mean:j"` (e.g. the mean of a volatility feature) and `"trace"` (total variance of all standardised features, the rule before 0.2).
* **`prob_stress`.** The summed probability of the states that are more volatile than their training window's average. With K = 2 it is `prob_regime_1`. With K = 3 it does not call a second calm state "stress", which `1 - prob_regime_0` does; use it instead of a state number whenever K can change between refits.
* **One regime.** Add 1 to the candidates (`n_components=[1, 2, 3]`) to let a window without a second regime say so; `prob_stress` is then 0. With `[2, 3]` a split is forced on every window.
* **Covariance regulariser.** `covar_prior` (default 0.01, standardised units) is a ridge on the covariance diagonal in every M-step. It keeps a state that explains a handful of bars from collapsing to a singular covariance. `min_covar` only initialises EM (hmmlearn does not apply it afterwards).
* **Fit quality.** Every refit records `converged`, `min_state_obs` (expected training bars in the rarest state), `bic`, `aic` and `warnings` in `history_`; one summary line is logged if any refit has a warning.
* **Covariance types.** `"full"` and `"diag"` are supported. `"tied"` has no per-state variance to sort by; `"spherical"` is stored inconsistently by hmmlearn 0.3.x.
* **Choosing K.** Rolling volatility is highly autocorrelated, so information criteria favour more states that slice volatility levels: on synthetic data with two regimes BIC picks K = 3 in 98% of the windows with a 21-bar volatility and in 64% with a 5-bar one, against 7% with the per-bar feature (`fetch_sample_data`, seeds 0–9). The per-bar feature is not immune: on bars built from an intrabar path (`simulate_regime_ohlc`) its distribution is not exactly Gaussian and BIC picks K = 3 almost always. `prob_stress` is what keeps the calm / stress reading stable in those windows. Compare K candidates on economic interpretability, not BIC alone.

### Choosing the volatility feature

`prepare_hmm_features` builds two features: the log return and a volatility measure.

| `vol_feature` | Second feature | Notes |
|---|---|---|
| `"rolling"` (default) | annualised volatility over the last `vol_window` bars (Parkinson with High/Low, else close-to-close) | the 0.1 behaviour |
| `"bar"` | log of the bar's own Parkinson variance, `ln(ln(H/L)² / (4 ln 2))` | needs High/Low; no window |

A Gaussian HMM assumes that, given the regime, each bar is a fresh draw. A rolling volatility is not: neighbouring windows share all bars but one, so the model sees the same information many times and treats it as certainty, and for `vol_window` bars after a change the window still contains the old regime. The per-bar log range is one noisy reading per bar, close to Gaussian (Alizadeh, Brandt & Diebold, 2002), and leaves the smoothing over time to the HMM, which is what the transition matrix is for.

Accuracy of the unfiltered calm / stress label (`prob_stress > 0.5`) against the true state, mean over seeds (min–max), 728 out-of-sample bars per seed. "True parameters" is a forward filter that knows the data generating process (`oracle_stress_proba` in `examples/demo.py`): the ceiling for any estimated model.

| Data | vol 21 | vol 5 | bar | True parameters |
|---|---|---|---|---|
| `fetch_sample_data`, seeds 0–9 | 0.77 (0.70–0.89) | 0.94 (0.92–0.95) | 0.96 (0.94–0.98) | 0.98 (0.96–0.99) |
| `fetch_sample_data`, seeds 10–29 | 0.76 (0.70–0.84) | 0.93 (0.90–0.95) | 0.96 (0.88–0.98) | 0.97 (0.96–0.99) |
| `simulate_regime_ohlc`, Gaussian returns, seeds 0–9 | 0.78 (0.68–0.83) | 0.93 (0.84–0.96) | 0.98 (0.97–0.99) | |
| `simulate_regime_ohlc`, Student-t(4) returns, seeds 0–9 | 0.75 (0.65–0.81) | 0.88 (0.80–0.93) | 0.89 (0.83–0.93) | |
| the same, whipsaw-filtered label | 0.74 (0.63–0.80) | 0.86 (0.79–0.91) | 0.92 (0.86–0.96) | |

Reading:
* With Gaussian returns the per-bar feature is the most accurate and close to the ceiling, and it returns to calm about 1 bar after the true change, against 3–4 bars (5-bar window) and 15–18 bars (21-bar window).
* With fat-tailed returns one extreme bar looks like stress to a per-bar feature: its unfiltered label is no more accurate than the 5-bar window's and flips 116 times per seed against 18 true changes. With the whipsaw filter it drops to 30 switches and is the most accurate of the three. Use `"bar"` together with the filter.
* All of this is synthetic data, where the true state is known. On real data there is no true state to score against; the argument for `"bar"` there is the one above (independent readings, no window memory), not these numbers.

Reproduce: `examples/whipsaw_seeds.py` (seeds 0–9), `... 10 30`, `... 0 10 0` (Gaussian path simulator), `... 0 10 4` (Student-t(4)).

### Whipsaw filter

The raw label (argmax of the filtered probabilities) flips whenever the probabilities hover near a boundary, and every flip is a trade. `hmm_regime/whipsaw.py` turns the probabilities into a stickier label with three causal rules:

| Rule | Parameter | Effect |
|---|---|---|
| Hysteresis | `threshold` (default 0.7) | switch only to a state with P ≥ threshold; an ambiguous bar keeps the current regime |
| Confirmation | `confirm_bars` (default 2) | the same new state must pass the threshold on this many consecutive bars |
| Minimum dwell | `min_dwell` (default 0) | hold a regime at least this many bars before the next switch |

```python
from hmm_regime import add_filtered_regime, switch_stats

out = model.fit_predict_filtered(features)
out = add_filtered_regime(out, binary=True)      # adds stress_filtered: 0 calm, 1 stress
switch_stats(out["stress_filtered"])             # switches, switches per year, run lengths
```

* **Causal.** The label at bar *t* uses probabilities up to *t* only: it is known at the close of *t*; act on it at the next bar's open. `tests/test_whipsaw.py` checks every cut point, for the filter and for the DataFrame wrapper on walk-forward output (both label types), and shows that the check catches planted one-bar leaks.
* **K changes.** `binary=True` filters `prob_stress` (the states more volatile than their training window's average), which keeps one meaning when the walk-forward switches K between refits; its unfiltered counterpart is `prob_stress > 0.5`. `stress_from="state0"` gives the rule before 0.2, calm = state 0 and stress = everything else (unfiltered counterpart `prob_regime_0 < 0.5`). With `binary=False` the label follows the K-state `regime`: if K shrinks and the held state disappears, the filter moves to the argmax at once (pass `n_states` when calling `filter_regimes` directly); if K grows, the held label keeps its number although the states were re-estimated.
* **Edge cases.** The first valid bar starts at the argmax and counts as a switch for `min_dwell`. A NaN row after that keeps the held regime, breaks a running confirmation and counts toward `min_dwell`. Use `threshold` > 0.5: at 0.5 or below a switch can happen on a tie.
* Every switch goes to that bar's argmax, so the filtered label never switches more often than the raw argmax of the same probabilities.
* **Cost: later detection.** `change_lags` lists every true change with its status and lag: *detected*, *missed* (the true state changed again first), *censored* (the data ended first) or *untracked* (on the bar before, the label was not on the old state, e.g. after missing the previous regime). `detection_lag` summarises it and can split by direction (`into=1`: onset of stress, `into=0`: return to calm). Two labels time different changes, so compare their lags on matched changes. `examples/whipsaw_seeds.py` (several minutes) runs the demo's comparison on synthetic seeds 0–9 with the filter at 0.7 × 2 bars. Switches, accuracy and lags: mean (min–max) over seeds; timed and missed changes: summed over seeds; lags in bars over detected changes:

  | Feature | Label | Switches | Accuracy | To stress: timed / missed | Lag to stress | To calm: timed / missed | Lag to calm |
  |---|---|---|---|---|---|---|---|
  | | true state | 19.6 (12–28) | 1.00 | 98 / 0 | 0 | 98 / 0 | 0 |
  | | true parameters | 29.1 (24–38) | 0.98 (0.96–0.99) | 97 / 2 | 0.4 (0.1–0.6) | 94 / 0 | 0.9 (0.3–1.3) |
  | vol 21 | raw | 17.9 (8–22) | 0.77 (0.70–0.89) | 71 / 5 | 2.5 (0.5–5.3) | 91 / 24 | 15.3 (10.7–19.8) |
  | vol 21 | filtered | 16.3 (8–22) | 0.75 (0.68–0.87) | 71 / 7 | 3.7 (1.5–6.3) | 89 / 23 | 16.6 (11.7–20.8) |
  | vol 5 | raw | 21.8 (18–30) | 0.94 (0.92–0.95) | 94 / 3 | 0.8 (0.2–1.4) | 94 / 4 | 3.3 (2.9–4.1) |
  | vol 5 | filtered | 19.8 (15–26) | 0.91 (0.88–0.93) | 91 / 7 | 1.9 (1.2–2.4) | 90 / 6 | 4.5 (3.5–5.2) |
  | bar | raw | 32.8 (22–55) | 0.96 (0.94–0.98) | 97 / 4 | 0.8 (0.3–1.4) | 91 / 0 | 1.0 (0.4–2.0) |
  | bar | filtered | 19.4 (12–25) | 0.94 (0.91–0.95) | 95 / 10 | 2.0 (1.4–2.6) | 86 / 2 | 2.4 (1.9–4.2) |

  Reading:
  * The filter matters most for the per-bar feature, whose raw label follows every noisy bar: it removes 13.4 switches per seed (−2 to −30) and brings the count to that of the true state. The rolling features are already smooth, so there is little to remove (−1.6 and −2.0 per seed).
  * It costs about 0.02–0.03 accuracy for every feature.
  * On every change both labels detect, the filtered label is at least one bar later, because confirmation needs two bars: per seed +1.0 to +2.0 bars to stress and +1.0 to +2.4 bars to calm (up to +8 for a single change).
  * It misses more short regimes: stress onsets missed over the ten seeds go from 5 to 7 (vol 21), 3 to 7 (vol 5) and 4 to 10 (bar).
  * The 21-bar feature returns to calm about 15 bars late and never shows 24 of the calm spells it was tracking. After such a miss the label is already in stress, so 27 of the 98 stress onsets are never timed.
  * Even a filter that knows the true parameters switches 29 times against 20 true changes: part of the flipping is the noise in the data, not estimation error.

  `examples/demo.py` shows one seed (42) and plots it.

  Synthetic regimes are clean draws; real data (fat tails, intraday noise) flips more. With Student-t(4) returns the per-bar raw label switches 116 times per seed and the filter is no longer a cost: it removes 86 of them and raises accuracy from 0.89 to 0.92 (`examples/whipsaw_seeds.py 0 10 4`). Judge the filter by the costs it saves against the moves it misses, in the strategy itself, and fix its parameters before the backtest: each setting tried is one more trial.

## 3. Limitations

While standard pitfalls like look ahead bias and arbitrary state switching are controlled for, several structural constraints remain inherent to the framework:

Detection Lag: Because forward filtering relies strictly on past and contemporaneous information ($P(S_t \mid Y_{1:t})$), regime changes require confirming bars to register. Consequently, the model naturally lags behind sudden liquidity cascades, flash crashes, or sharp exogenous shocks: the initial move is absorbed before the regime shift is registered. A rolling volatility feature adds its own window to that lag.

Gaussian Tail Underestimation: Asset returns systematically exhibit skewness and heavy tails (leptokurtosis). While Gaussian emissions ensure numerical stability and fast convergence during calibration, they structurally underestimate the frequency and impact of extreme tail events compared to fat-tailed alternatives like Student's $t$-distributions. In practice a single extreme bar is read as a change of regime; the per-bar feature is the most exposed to this (see the Student-t rows above).

Sensitivity to Initialization (Local Optima): Expectation-Maximization (Baum-Welch) is a hill climbing algorithm inherently sensitive to starting parameter vectors. The implementation mitigates this with multi-start EM (`n_init` restarts, best likelihood kept), but a global maximum is still not guaranteed.

Relative Regimes: "Calm" and "stress" are defined inside each training window. A window that contains only one regime is still split in two unless K = 1 is among the candidates, and a level of volatility that counts as stress in a quiet year can count as calm in a turbulent one.

Whipsaw & Turnover Drag: In noisy or transitional market phases, probabilities can hover near decision boundaries, causing the model to flip rapidly between regimes. In live execution, this regime jitter leads to excessive turnover, accumulating bid-ask spreads, exchange fees and execution slippage. The whipsaw filter (`hmm_regime/whipsaw.py`, see above) reduces the flips at the price of a later detection.

No Real-Data Results: Everything measured in this repository is on synthetic data, where the true state is known. It is a library, not a strategy.

## Citation

If you use this architecture or reference this implementation in your research, please cite the repository:

```bibtex
@software{altekin_hmm_regime_detection,
  author    = {Altekin, Mehmet {\"O}mer},
  title     = {hmm-regime: walk-forward Gaussian HMM market regime detection with causal forward filtering},
  year      = {2026},
  version   = {0.2.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24524692},
  url       = {https://doi.org/10.5281/zenodo.24524692}
}
```
