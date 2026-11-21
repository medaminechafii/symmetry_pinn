# Week 6, stage 1: hybrid PINN with learned g and learned nu

Burgers with an unknown source: u_t + u u_x = nu u_xx + g(u), nu = 0.1, T = 2, periodic.
True g in {0, -u^3}. Cases called `g0` and `cubic`.

Status: stage 1 only (no symmetry loss, mu = 0). Numbers below are copied from the
analysis output pasted in the working session; the code that produced them is
`experiments/analyse_learned_g_all.py` (plus the ν-g block appended to it) on the ubuntu machine.

---

## 1. What was run

- Loss: data MSE + lambda * mean(PDE residual^2), lambda = 1, 2000 collocation points,
  Adam 4000 steps + L-BFGS. Same as the oracle runs, except that g and nu are no longer given.
- Residual: `u_t + u u_x - nu_hat u_xx - g_hat(u)`. nu_hat = exp(rho), rho learned, start 0.05.
  **The true g is never used in training.** It is used only afterwards, to score.
- Two g arms:
  - `poly`: g_hat(u) = sum_{k=1..5} c_k u^k, c starts at 0, so g_hat(0) = 0 by construction.
  - `MLP`: 1 -> 32 -> 32 -> 1 tanh, with g_hat(u) = net(u) - net(0).
- Each experiment k (0, 1, 2) trains its own network and its own g_hat (no shared g yet).
- Grid: case in {g0, cubic}, N in {50, 200, 1000}, sigma in {1%, 3%}, 3 experiments, 3 seeds
  = 108 models per arm. Per cell: 9 runs. The 9 runs are not independent samples: each experiment's
  data are fixed, and the seeds only change initialisation and collocation points.
- Assumption to remember: g(0) = 0 is built in (all our cases satisfy it). A constant source
  would be removed by it.

## 2. Metrics

- `err_g`: RMS(g_hat - g_true) over the u values of the true dense solution, divided by RMS(u^3).
  For cubic this is a relative error of g. For g0 the true g is 0, so it is an absolute error
  measured against the size of the cubic term. **The two cases are not comparable to each other.**
- `nu_err`: |nu_hat - 0.1| / 0.1.
- `D`: PDE residual of the network's jets plugged into the TRUE equation (true nu, true g), on the dense grid.
  Reference values: oracle floor about 0.004; data-only surrogate at 1% noise about 0.06 / 0.19 / 0.5
  for N = 1000 / 200 / 50.

Reading D correctly: the network's jets are trained to satisfy the learned equation, so
true residual ≈ (g_hat - g) + (nu_hat - nu) u_xx. So **D contains the g error and the nu error;
it is not a pure jet-quality score.** (My first description of D, "only tells you how good the jets
are", was too simple; corrected during the session.)

## 3. Results (medians over the 9 runs of each cell)

| case | N | sigma | arm | err_g | nu_err | D |
|---|---|---|---|---|---|---|
| cubic | 1000 | 1 | MLP / poly | 0.052 / 0.051 | 0.0023 / 0.0027 | 0.0117 / 0.0109 |
| cubic | 1000 | 3 | MLP / poly | 0.083 / 0.062 | 0.0025 / 0.0021 | 0.0203 / 0.0153 |
| cubic | 200 | 1 | MLP / poly | 0.144 / 0.121 | 0.0081 / 0.0082 | 0.0260 / 0.0225 |
| cubic | 200 | 3 | MLP / poly | 0.246 / 0.217 | 0.0208 / 0.0164 | 0.0459 / 0.0421 |
| cubic | 50 | 1 | MLP / poly | 0.218 / 0.135 | 0.0104 / 0.0036 | 0.0419 / 0.0266 |
| cubic | 50 | 3 | MLP / poly | 0.351 / 0.300 | 0.0226 / 0.0199 | 0.0635 / 0.0598 |
| g0 | 1000 | 1 | MLP / poly | 0.0071 / 0.0228 | 0.0021 / 0.0022 | 0.0047 / 0.0074 |
| g0 | 1000 | 3 | MLP / poly | 0.0184 / 0.0316 | 0.0034 / 0.0037 | 0.0065 / 0.0105 |
| g0 | 200 | 1 | MLP / poly | 0.040 / 0.058 | 0.0064 / 0.0062 | 0.0116 / 0.0144 |
| g0 | 200 | 3 | MLP / poly | 0.145 / 0.165 | 0.0144 / 0.0103 | 0.0392 / 0.0462 |
| g0 | 50 | 1 | MLP / poly | 0.062 / 0.106 | 0.0084 / 0.0053 | 0.0176 / 0.0279 |
| g0 | 50 | 3 | MLP / poly | 0.234 / 0.266 | 0.0375 / 0.0440 | 0.0496 / 0.0683 |

Ratio of D to the oracle floor (0.004; floor taken from earlier oracle runs, not re-queried per cell),
poly / MLP: g0 from 1.9x / 1.2x (N = 1000, 1%) up to 17x / 12x (N = 50, 3%);
cubic from 2.7x / 2.9x up to 15x / 16x.

### What the table says

1. **Learned g and nu work well in the easy cell and degrade smoothly with N and noise.**
   Cubic at N = 1000, 1%: g error about 5%, nu error about 0.25%, D about 3x the oracle floor.
   nu is not worse than when only nu was learned with g known (about 0.5%).
2. **MLP better on g0 (6/6 cells by median), poly better or equal on cubic (6/6).** At cubic N = 1000, 1% it is a tie.
   The ranges overlap in almost every cell, so this is a tendency, not a clean separation.
   "The restricted form helps" is not a full explanation: the polynomial class also contains the
   truth for g0 (all coefficients 0), yet it loses there. Possible reason (untested): collinear coefficients
   drift when the data constrain them weakly.
3. **D is well below data-only but not at the oracle.** At 1% noise the learned-g D is 5-12x below
   data-only at N = 1000 and roughly midway (log scale) between oracle and data-only at N = 50.
   Data-only baselines at 3% noise are not available here.
4. **Hard cells have a weak spot.** g0 at N = 50, 3%: learned g has about 25% of RMS(u^3) although the
   true g is 0. In that cell a weak source can be confused with no source.
5. An outlier: MLP, cubic, N = 50, 1% has a run with err_g 0.82, nu_err 0.10, D 0.17 (a failed run;
   not inspected yet).

### The coefficients are not reliable on their own

Cubic runs: c_3 between -0.97 and -1.22 (8 of 9 below -1), non-cubic norm 0.2-0.7, yet err_g only 2-8%.
Reason: u, u^3, u^5 are strongly collinear over the range of u the solution visits, so large coefficients cancel.
**Report the function error, not c_3.** The learned g is accurate only inside the visited u range.
A made-up example of the cancellation, at |u| = 0.5: -1.2 u^3 + 0.8 u^5 = -0.125, same as -u^3, but
at |u| = 1 it gives -0.4 instead of -1. (Earlier single Mac run: c = [2.9e-4, 2.9e-3, -0.94, 9.0e-3, -0.081].)

Flaw in my own metric: for g0 the homogeneity number R used a norm over 60,416 points (scales with sqrt(N));
the g0 values (about 1.2) are therefore not comparable with the cubic ones (relative form, 0.03-0.15).
Divide by sqrt(N) if R is used again.

## 4. Is the network compensating for a wrong g with a wrong nu?

Test: cosine between (g_hat - g) and (nu_hat - nu) u_xx on the grid (-1 = errors cancel, 0 = unrelated).
Also `combined` = ||dg + dnu|| / sqrt(||dg||^2 + ||dnu||^2) (1 = independent, <1 = cancelling).

- cos is negative in most cells (medians -0.1 to -0.6; about 0 only in the easy cubic cell). Both arms agree.
- `combined` is 0.9-1.0 in most cells. Strongest: g0 MLP N = 50, 3%: 0.68 (about half the squared error cancels),
  g0 MLP N = 200, 1%: 0.82.
- Cubic: the nu error is only 8-24% of the size of the g error, so it cannot compensate for much.
- g0: size ratio 0.15-0.96, so more room, and the coupling is strongest there.
- Individual runs vary a lot (cos from -0.66 to +0.49 within one cell).

Conclusion: no large compromise (D already excludes it), but a moderate nu-g coupling in the harder settings.
A low D does not certify g_hat there. Caveats: the cosine is a correlation on the grid, not proof of causation;
it uses the true u_xx; no shuffled-pairing control was run.

## 5. Reasoning and prediction record

Prediction written before the run (in words, from the surrogate results): u is fit to ~0.5% by the data term;
u_x, u_t, u_xx had ~1%, 4%, 8% error in the data-only surrogate; therefore g would be off by ~10-15%, nu by ~5%,
and D would look like the data-only value.
Underlying picture: "fit the data first, then read g off the jets".

Outcome at the predicted setting (N = 1000, 1%): g error 2-5% (cubic) and well below the prediction, nu error 0.2-0.3%, D 0.005-0.012 vs
data-only 0.06. **All three were off in the optimistic direction.** The prediction roughly describes the harder cells
(cubic N = 200, 1%: g error 12-14%; N <= 200, 3%: nu error 1.5-4%).

Why the picture was too pessimistic (hypothesis, not proven): training is not staged. The PDE loss is active from
step 0 and, with only 5 coefficients plus nu to learn, constrains the jets strongly, so errors in the jets that are not
a function of u average out in the regression for g, and the PDE loss improves the jets in return. Only the part
of the jet error correlated with u (and u_xx through the front shape) leaks into g.
Not checked: which loss dominates the gradient early (`grad_his` stores both norms).

Note from the session: the one Mac run with c_3 = -0.94 had been seen before the prediction was written.

## 6. Open items

- Inspect the failed MLP run (cubic, N = 50, 1%, err_g 0.82).
- Paired comparison MLP vs poly on the same (experiment, seed); control for the nu-g cosine.
- Exact oracle D per cell from `metrics.jsonl` instead of the 0.004 floor.
- Condition number of [u_xx, u, ..., u^5] on the true jets (identifiability of nu vs g).
- Shared g across experiments (currently one g per experiment).
- Stage 0: floor of the substituted discovery on true and oracle jets (not measured).
- Symmetry loss: skeleton with mu = 0 that reproduces stage 1 exactly; true generators first;
  test on the hard cells (N = 50, 200 at 3%); budget-matched baseline needed to attribute any gain to L_sym.
- `dense_cubic.npz` md5 differs between the Mac and the ideapad (Mac: 702a012bd90964ad38c406977e37972f);
  cause unknown, ideapad hash not yet obtained.

## 7. Reproduce

Training (ubuntu, all arms; skips models that already exist):

```
bash run_lg_all.sh        # xargs -P 6, arms {MLP, poly}, N {1000, 200, 50}, sigma {1, 3}, seeds 0-2, --learn-nu 0.05
```

Check: `grep -c '"g_model": "MLP"' results/oracle/metrics.jsonl` and the same for `poly` (expect 108 each).
Analysis: `python -m experiments.analyse_learned_g_all` (writes `results/learned_g_per_run.csv`).
Models: `models/oracle_{name}_N{N}_s{sig}_lam1_k{k}_nu0.05_{poly|MLP}_seed{seed}[_g].pt`.
