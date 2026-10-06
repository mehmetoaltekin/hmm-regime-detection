# hmm-regime-detection
Quantitative market regime identification via Gaussian HMMs: walk-forward parameter estimation, state sorting, and information criteria (AIC/BIC) selection.

## Disclaimer
This repository is strictly for academic and quantitative research purposes. 
Nothing contained herein constitutes financial, investment, or trading advice. 
The authors accept no responsibility for any financial losses incurred through 
the use of these models or strategies.

## References

* **Hamilton, J. D. (1989).** *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica, 57(2), 357–384.
* **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* John Wiley & Sons. (Bölüm: Structural Breaks & Regime Shifts).
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
To ensure economic consistency over time, we sort and re index the regimes by their estimated variance ($\sigma_k^2$) at each calibration step, pinning State 0 to the lowest volatility regime.

Model Selection ($K$): The number of hidden states is selected by balancing fit against parameter bloat using the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC):

$$\text{BIC} = -2\ln(\hat{L}) + p \ln(T)$$ 

where $\hat{L}$ is the maximized likelihood, $p$ is the number of free parameters, and $T$ is the number of observations.


## 2. Installation & Quickstart

```bash
git clone https://github.com/mehmetoaltekin/hmm-regime-detection.git
cd hmm-regime-detection
pip install -r requirements.txt
PYTHONPATH=. python examples/demo.py      # offline demo on synthetic data
PYTHONPATH=. python -m pytest tests -v    # look-ahead & regression tests
```

### Minimal usage

```python
from src.data_loader import fetch_returns          # needs: pip install yfinance
from src.hmm_model import WalkForwardHMM

# 1. Stationary features (log returns + Parkinson volatility)
features = fetch_returns("XAUUSD", start="2015-01-01", end="2024-01-01")

# 2. Walk-forward HMM: refit every 21 bars on the last 252 bars.
#    Passing a list lets BIC choose K inside each training window.
model = WalkForwardHMM(n_components=[2, 3], window_size=252, step_size=21)

# 3. Out-of-sample filtered probabilities P(S_t | Y_1:t)
out = model.fit_predict_filtered(features)
out[["prob_regime_0", "prob_regime_1", "regime", "n_states"]].tail()
```

Offline alternative: `fetch_sample_data()` + `prepare_hmm_features()` (see `examples/demo.py`).

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
* **Covariance types.** `"full"` and `"diag"` are supported. `"tied"` has no per-state variance to sort by; `"spherical"` is stored inconsistently by hmmlearn 0.3.x.
* **Choosing K.** Rolling volatility is highly autocorrelated, so information criteria tend to favour more states that slice volatility levels. Compare K candidates on economic interpretability, not BIC alone.

## 3. Limitations & Edge Cases

Inherent Detection Lag: Because filtered probabilities update strictly sequentially without looking ahead, the model inherently lags behind sudden market dislocations. Rapid, exogenous shocks are only recognized after a few confirming bars, meaning the initial move is absorbed before the regime shift is fully registered.

Fat Tail Underestimation: Assuming Gaussian emissions keeps numerical optimization stable and fast, but real returns exhibit heavy tails (leptokurtosis). As a result, the model structurally underestimates the probability of extreme tail events compared to heavy tailed alternatives like Student's $t$-distributions.

Whipsaw & Turnover Drag: In noisy or transitional market phases, probabilities can hover near decision boundaries, causing the model to flip rapidly between regimes. In live execution, this regime jitter leads to excessive turnover, accumulating execution slippage and transaction costs.

## 4. Methodological Limitations

While standard pitfalls like look ahead bias and arbitrary state switching are controlled for, several structural constraints remain inherent to the framework:

Detection Lag: Because forward filtering relies strictly on past and contemporaneous information ($P(S_t \mid Y_{1:t})$), regime changes require several confirming bars to register.

Consequently, the model naturally lags behind sudden liquidity cascades, flash crashes, or sharp exogenous shocks.

Gaussian Tail Underestimation: Asset returns systematically exhibit skewness and heavy tails (leptokurtosis). While Gaussian emissions ensure numerical stability and fast convergence during calibration, they structurally underestimate the frequency and impact of extreme tail events compared to fat-tailed alternatives like Student's $t$-distributions.

Sensitivity to Initialization (Local Optima): Expectation-Maximization (Baum-Welch) is a hill climbing algorithm inherently sensitive to starting parameter vectors. The implementation mitigates this with multi-start EM (`n_init` restarts, best likelihood kept), but a global maximum is still not guaranteed.

Turnover & Execution Friction: In choppy, sideways markets, filtered probabilities often hover near decision thresholds. This regime jitter creates whipsaws triggering rapid portfolio reallocations that can quickly erode alpha through bid-ask spreads, exchange fees, and execution slippage.

## Citation

If you use this architecture or reference this implementation in your research, please cite the repository:

@software hmm_regime_detection,

author = Mehmet Altekin / mehmetoaltekin,

title = Quantitative market regime identification via Gaussian HMMs: walk-forward parameter estimation, state sorting, and information criteria (AIC/BIC) selection.,

year = 2026,

publisher = GitHub,

howpublished = https://github.com/mehmetoaltekin/hmm-regime-detection
