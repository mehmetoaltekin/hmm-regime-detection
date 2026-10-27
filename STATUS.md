# STATUS

Last updated: 2026-10-27 (branch `feature/whipsaw-filter`, not pushed).

## Where things stand
- Library only: walk-forward Gaussian HMM with a causal forward filter, variance-sorted states, K by AIC/BIC per window. No strategy, no trading rules, no real-data results in this repo.
- New: whipsaw filter `src/whipsaw.py` (hysteresis threshold, confirmation bars, minimum dwell; causal), turnover and lag helpers, `tests/test_whipsaw.py` (26 tests). All 52 tests pass (JupyterLab Desktop Python 3.12, hmmlearn 0.3.3).
- Fixed: the synthetic data's High/Low range ignored the regime, so the Parkinson volatility feature carried no regime information. The range now scales with the regime's sigma (checked: 0.0080 / 0.0251 per bar against 0.008 / 0.025).
- Demo result (synthetic, seed 42): the filter cuts switches only where the raw label flips (volatility window 5: 31 → 29) and adds about one bar of lag; the 21-bar volatility window alone adds about 7 bars of lag over the 5-bar one. Numbers in `README.md`, from `examples/demo.py`.

## Next step
- The intended use is VIOP futures with SMC (smart money concepts) entries. Following `~/.claude/quant-lessons.md`, the SMC rule needs its own raw-edge test on VIOP data (`/edge-check`) before this regime filter is attached to it. The earlier ICT/SMC test on US stocks (`~/Documents/ict_sweep`, Oct 2026) found no edge in 6 of 6 pre-registered tries.
- Then, only if the SMC edge passes: run the HMM on the same VIOP bars, fix the filter parameters in the plan, and judge the filter by costs saved against moves missed.

## Open questions
- VIOP data: which contract (e.g. BIST30 index futures), bar size, source, years? A continuous series needs roll handling (bi-monthly expiries) so the roll gap is not a fake return, and session gaps (evening session) need a rule.
- Binary label: calm (state 0) vs the rest is used now. "Highest-variance state vs the rest" scored better at window 21 with K = 2–3 on synthetic data, worse at window 5; no clear winner, so it was not added.
