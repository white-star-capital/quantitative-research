# results/

## The committed artifacts are stale and must be regenerated

`results.csv` and the three PNGs in this directory were produced before two
reporting bugs were fixed. **Do not cite these numbers.**

1. **`oos_sharpe = -inf` (rows W1s0, W2s0, W2s1, W2s2) is not performance.**
   It is the worst-fitness ranking sentinel that `evaluate()` returns so NSGA-II
   can order unusable individuals. It was recorded verbatim as an OOS Sharpe.
   In every one of those rows the OOS trade filter tripped — the 50-trade
   threshold written for a ~12-month training window was applied unchanged to a
   3-month OOS window — so those runs were never measured at all. Any median or
   IQR taken over that column is also meaningless.

2. **The `dsr` column is identically zero by construction.** `E[SR_max]` was
   computed without its `sigma_SR` scale factor, putting the significance
   hurdle at roughly 24 annualized. No strategy clears that, so every row reads
   between `1e-170` and `1e-286` regardless of what was evolved. The values
   carry no information about the strategies.

3. **The `dsr` column was sized to the wrong trial population.** N was the 9
   reported winners rather than the thousands of individuals the GP actually
   evaluated, which left the multiple-testing correction nearly inert — the
   hurdle multiplier is 1.52 at N=9 against 3.24 at N=3600. On signal-free data
   the winners-only convention certifies noise above DSR 0.95 while the
   evaluation-based one rejects the same runs below 0.05.

4. **No null control was run.** DSR cannot detect bias shared by every trial, so
   nothing in that run rules out a lookahead, a reused training window or a
   survivor-biased universe.

Regenerate with `make start` (or `python scripts/run.py`). The new schema adds
`oos_status`, `oos_n_trades`, `oos_min_trades`, `dsr_bests_only`,
`dsr_n_trials`, `dsr_trial_source`, `dsr_n_evaluations`, `dsr_trial_sr_std` and
`n_evaluations`, and writes `null_control.txt` alongside. `NaN` in a metric
column means "not measured", never "bad result".

Read three numbers together, not one: the OOS Sharpe, the conservative `dsr`,
and the null control p-value.

## Note on the IS/OOS gap

Separately from the bugs above, the stale run shows IS Sharpe 3.1–4.1 against
OOS Sharpe 0.7–1.3 where OOS was measurable at all. That gap is a
generalization result in its own right and is not addressed by fixing the
reporting path — see the DSR and null-control sections of the top-level README.
