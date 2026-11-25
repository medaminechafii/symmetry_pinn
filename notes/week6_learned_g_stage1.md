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
- Stage 0: floor of the substituted discovery on true and oracle jets: done, see section 8.
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

---

## 8. Symmetry detection on the learned equations (Stage 0 / 0b)

Full explanation with worked examples: `notes/symmetry_detection_problem.pdf`. Short version below.
Detector: 30 basis fields (xi, tau, phi in {1, x, t, u, x^2, t^2, u^2, xt, xu, tu}), Lie criterion at 3000 points,
columns normalised, SVD. gap_k = log10(sigma_(k+1) / sigma_(k)), sigma_(k) = k-th smallest singular value.
True counts: cubic 3 (d/dx, d/dt, scaling), g0 5 (+ Galilean, projective).

### 8.1 Stage 0 (true nu and g)
- Substituting u_t, u_xt with the equation gives gap about 14 for truth, oracle and data-only jets, and the swap test
  changes nothing. Reason: with the true g the symmetry columns are identically 0 as formulas, so jets cannot matter.
  The floor of substituted discovery is set by the learned g_hat and nu_hat, not by the jets.
  (My earlier claim that u_xx/u_xxx errors would limit it was wrong.)
- Raw columns (u_t, u_xt from the network): oracle 0.92 (count 2), data-only 0.07-0.37 (cubic N = 1000, 1%).
- Jet errors, cubic N = 1000, 1%: oracle u 0.17%, u_x 0.50%, u_xx 1.44%, u_xxx 3.57%; data-only 0.48 / 1.19 / 3.37 / 8.73%.

### 8.2 Learned equations (stage 1 runs, 108 poly + 108 MLP)
- Largest-jump count: 2 in 216 / 216 runs (only the trivial d/dx, d/dt). This rule is flawed: the jump from the
  trivial values (1e-17) to the next value always wins.
- gap at the true k, medians per cell: cubic poly 0.27-0.55, cubic MLP 0.22-0.33, g0 poly 0.11-0.54, g0 MLP 0.20-1.18.
  No trend with N or noise. Exact equation: about 14.
- Calibration on exact g = -u^3 + eps u (data-free): eps 0.003 -> gap 2.0; 0.01 -> 1.5; 0.03 -> 1.0; 0.1 -> 0.6; 0.3 -> 0.4.
  RMS(u) = 0.197, RMS(u^3) = 0.0395, so eps = 0.003 is a breaking of 1.5% of the cubic term. The detector is very sensitive.

### 8.3 Looking for the cause
- Scaling breaking B = u g' - 3 g (0 for g = c u^3; = -2 eps u for the eps equations).
  eps_eff = RMS(B) / (2 RMS(u)): learned cubic runs 0.02-0.05 (poly), 0.03-0.09 (MLP). This predicts gaps about 0.8-1.2;
  the actual gaps are 0.6-0.8 lower in every cell and both arms. **Not explained.**
- Coefficient attribution (poly, cubic): c3 only -> gap about 14; c3 + c5 alone -> same gap as the full equation;
  removing c5 -> 0.6-1.6 (c4, c1 still break it). Median |c5| 0.30-0.55 against |c3| about 1.1 (signs not checked).
- MLP shows the same low gaps -> not a polynomial artefact.
- Derivative check (cubic, units of RMS(u^3)), N = 1000, 1%: value error 0.054 (poly) / 0.068 (MLP),
  derivative error about 2x larger, B = 0.257 / 0.359, i.e. 4-5x the value error. In harder cells B is about 2x the value error.
- Power-law test (best c u^p, p = 1..5): p = 3 in 106 / 108 cubic runs; misfit 4-29%, tracks the function error.
  Calibration: eps 0.003 -> 1%, 0.01 -> 3.5%, 0.03 -> 11%, 0.1 -> p = 5 with 35%. No control on learned near-symmetric runs yet.

### 8.4 Deflated count (drop values below 1e-9 x largest, then largest jump among the next 8 if >= J_min)
J_min was fixed before looking: 0.5 (accepts about eps 0.1, i.e. 50% of the cubic term) and 1.0 (about eps 0.03).

| case, arm | correct @0.5 | correct @1.0 |
|---|---|---|
| cubic poly (true 3) | 11 / 54 | 1 / 54 |
| cubic MLP | 2 / 54 (plus one 4 and one 5) | 0 / 54 |
| g0 poly (true 5) | 5 / 54 | 1 / 54 |
| g0 MLP | 30 / 54 (18 / 18 at N = 1000) | 14 / 54 |

Cubic is not rescued; g0 with the MLP at N = 1000 is (all 5 symmetries in 18 / 18 runs at the loose tolerance).

### 8.5 Substituted vs raw on the learned runs (same network jets, same g_hat, g_hat', nu_hat)
`python -m discovery.raw_vs_sub_learned --arm poly|MLP`. raw = substituted part - 3 x (network residual w.r.t. its own learned equation).

| case, arm, cell | residual | sub gap | sub correct | raw gap | raw correct |
|---|---|---|---|---|---|
| cubic poly 1000/1% | 0.0040 | 0.38 | 2 / 9 | 0.30 | 2 / 9 |
| cubic poly 50/1% | 0.0053 | 0.51 | 5 / 9 | 0.38 | 2 / 9 |
| cubic MLP 200/3% | 0.0065 | 0.35 | 0 / 9 | 0.17 | 0 / 9 |
| g0 poly 1000/1% | 0.0044 | 0.55 | 4 / 9 | 0.56 | 3 / 9 |
| g0 MLP 1000/1% | 0.0046 | 1.18 | 9 / 9 | 0.89 | 8 / 9 |
| g0 MLP 1000/3% | 0.0047 | 1.14 | 9 / 9 | 0.85 | 9 / 9 |
| g0 MLP 200/1% | 0.0044 | 0.78 | 5 / 9 | 0.79 | 6 / 9 |

(Selected cells; counts with the deflated rule, J_min = 0.5. Totals: cubic poly 13 / 54 sub vs 5 / 54 raw; g0 MLP 30 / 54 both.)
- The networks satisfy their own learned equation to 0.4-0.6% (about 1% at N = 50, 3%).
- Raw is never better and usually only slightly worse (0-0.13 decades); the largest loss is MLP g0 at N = 1000 (about 0.3),
  where g_hat is nearly symmetric and the residual starts to matter.
- Conclusion: the bottleneck is the symmetry breaking of g_hat, not the time derivatives. Substitution helps modestly and is kept.
- Here the substituted columns are evaluated at the network's u, so the sub counts differ slightly from 8.4 (13 vs 11 for cubic poly).

### 8.6 Comparison with LieNLSD (Hu, Li, Lin, ICML 2025; code github.com/hulx2002/LieNLSD)
From a reading of 7 repo files (not all verified): same 10-monomial library and SVD; the number of generators is a fixed
absolute threshold on sqrt(singular values of sum C^T C), hand-set per task (Burgers 0.2, heat 0.1, KdV 1, ...);
no column normalisation; criterion from the network Jacobian and finite differences (not substituted); noise-free data;
ADMM sparsification of the basis afterwards. Our deflated J_min rule is not their method.
Normalisation does not change exact null spaces; on approximate ones skipping it gave about 0.3-0.4 decades more separation in examples.

### 8.7 Open
- Why the actual gap is 0.6-0.8 below the eps_eff prediction.
- False-positive control: stage 1 on near-symmetric data (eps = 0.03, 0.1), then the power-law test and the deflated count.
- g0: no reference for "g_hat consistent with zero".
- Scripts: discovery/stage0_floor.py, eps_calibration.py, stage0b_learned.py, stage0b_general.py, eps_eff_check.py,
  attribution_check.py, derivative_powerlaw_check.py, deflated_count_check.py, raw_vs_sub_learned.py.

## 9. Symmetry loss with the true generators, and the weight-decay baseline (2026-11-24 / 25)
Setup for both: poly arm, N = 200, sigma = 3%, k = 0-2, seeds 0-2, nu learned from 0.05, lambda = 1. Same cells as stage 1.
- Symmetry loss: L = L_data + lambda L_PDE + mu L_sym, L_sym = mean of S^2 over the generators of the case
  (S = pr V(Delta) - lambda Delta), network jets detached, so only g_hat feels it. `--sym-mu`, script run_sym_pilot.sh.
- Weight decay: L = L_data + lambda L_PDE + wd * sum c_k^2. `--g-wd`, script run_wd_pilot.sh (54 models).
  Note: wd = 1e-4 is stored as 0.0001 in metrics.jsonl, and the logs are named ..._wd1e-4_....
- Read with `python -m experiments.sym_coeffs [--wd 1e-6 1e-5 1e-4]` and `python -m experiments.analyse_learned_g_all --mu m | --wd w`.

### 9.1 Results (medians over 9 runs)
| case | stage 1 | mu 0.01 / 0.1 / 1 | wd 1e-6 / 1e-5 / 1e-4 |
|---|---|---|---|
| cubic err_g | 0.217 | 0.181 / 0.167 / 0.156 | 0.201 / 0.429 / 0.626 |
| g0 err_g | 0.165 | 0.020 / 0.0028 / 0.0003 | 0.164 / 0.114 / 0.060 |
| g0 nu error | 1.0% | ... / ... / 0.4% | 1.2% / 1.3% / 1.3% |
| cubic nu error | | | 1.7% / 0.8% / 3.0% |
| data/noise | 0.77-0.95 | unchanged | up to 1.03 (k0), 1.21 (k1) at 1e-4 |

### 9.2 What the coefficients show (experiments/sym_coeffs.py)
- Symmetry loss, mu >= 0.1: c1, c2, c4, c5 go to about 0. The cubic error left is c3 alone, set by the data draw
  (mu = 1: k0 -0.85, k1 -1.19, k2 -0.98; true -1). Seeds agree to about 0.01, so each cell is effectively 3 independent samples.
  c3 moves with nu_hat (dc3 about 50-70 dnu). Reason: scaling gives S = sum (k-3) c_k u^k, so c3 (and nu) are free.
  The symmetry fixes the form of g, not its size.
- Weight decay: c3 shrinks monotonically toward 0 (k0: -0.89, -0.69, -0.33, -0.06 for wd 0, 1e-6, 1e-5, 1e-4).
  c5 is cleared already at 1e-6; c4 only at >= 1e-5. Two side effects: wd creates c1 (-0.04 to -0.07) and raises
  nu_hat (to 0.103-0.106). The lost cubic damping is replaced by the cheapest substitutes: c1 (u^1 is the largest power
  for |u| < 1, so the data curvature a is large and the shrink factor a/(a+w) small) and extra viscosity (no penalty).
  Same c3-nu coupling as above: the data alone barely separate a cubic source from extra viscosity.
- g0: Galilean gives S = -g', so the symmetry loss forces g_hat = 0 exactly; weight decay only pulls toward 0 against the data
  (c3 still -0.06 for k1 at 1e-4).

### 9.3 Conclusion
The symmetry-loss gain is not "shrink g_hat". At every wd tried, both cases are worse than with mu = 0.01; one wd cannot serve
both cases (at 1e-4 g0 is still 200x worse than mu = 1 while cubic has already gone from 0.22 to 0.63). The symmetry loss improves
both with one mu. Not shown: whether wd = 1e-3 on g0 alone reaches about 3e-4 (likely, by forcing g_hat -> 0), but choosing wd
per case needs the answer in advance.
Prediction record: the user predicted "wd clears c4, c5 but pulls c3 toward 0" (c3 right; c4 only at >= 1e-5).
I predicted wd may also clear c4, c5 and bias c3 (same); neither of us predicted the c1 / nu_hat compensation.

### 9.4 Next
False-positive control on the eps datasets (8.7), then the alignment check and discovered generators (candidates from the
singular vectors, accepted by held-out data error).

## 10. False-positive control: stage 1 on near-symmetric data (2026-11-25)
Question: the detector found scaling in some stage-1 cubic runs (8.4: 11 / 54 poly at J_min = 0.5). Would it also "find"
scaling on data where scaling is broken? If yes, those detections are not evidence.

Setup (run_eps_control.sh, 144 models): stage 1 only (learned g and nu, NO symmetry loss, no weight decay), same settings
as run_lg_all.sh, on data from g = -u^3 + eps u, eps = 0.03 (e003, about 15% of the cubic term) and 0.1 (e01, about 50%).
N 200 and 1000, sigma 1 and 3%, poly and MLP, k 0-2, seeds 0-2. Data: experiments/make_near_sym.py --eps e003 e01 --n 200 1000.
The eps data use the same observation points and the same noise draws (rescaled by 5-7%) as the cubic data, so each eps run
has a twin cubic run (same arm, N, sigma, k, seed) from stage 1; the twins differ only by the eps u term.
Analysis: `python -m experiments.eps_control --arm poly|MLP`.
(Counting tip: grep 'cubic_e0' also matches 72 older oracle rows on e001-e03; count learned rows with g_model and learn_nu.)

Measures:
- gap3: log10(s_4 / s_3) of the 30 substituted columns (the SVD detector; large = scaling looks exact).
- c3@0.5, c3@1.0: fraction of runs with deflated count 3 (scaling declared) at J_min 0.5 / 1.0.
- p3, misfit: fraction with best power law u^3; relative distance of g_hat from the best c u^p.
- eps_eff = RMS(B) / (2 RMS(u)), B = u g_hat' - 3 g_hat. B = 0 is the scaling condition; for g = -u^3 + eps u, B = -2 eps u,
  so eps_eff = eps exactly: "the eps that breaks scaling by the same amount as g_hat".
- win: fraction of twin pairs where the cubic twin has the larger gap3 (1 = detector always prefers the symmetric twin,
  0.5 = coin flip).

### 10.1 Results (medians over 9 runs per cell; ranges over the 4 cells N x sigma)
| | exact equation | poly, learned | MLP, learned |
|---|---|---|---|
| gap3 cubic / e003 / e01 | 13.9 / 1.03 / 0.57 | 0.27-0.43 / 0.25-0.46 / 0.38-0.44 | 0.22-0.33 / 0.20-0.36 / 0.18-0.40 |
| win e003 / e01 | - | 0.44-0.56 / 0.11-0.44 | 0.11-0.44 / 0.22-0.44 |
| c3@0.5 cubic / e003 / e01 | 1 / 1 / 1 | 0-0.22 / 0-0.22 / 0.11-0.44 | 0 / 0-0.11 / 0 |
| eps_eff cubic / e003 / e01 | 0 / 0.030 / 0.100 | 0.024-0.036 / 0.029-0.038 / 0.087-0.095 | 0.030-0.041 / 0.032-0.036 / 0.082-0.094 |
| p3 cubic / e003 / e01 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0-0.22 |
| err_g cubic / e003 / e01 | 0 | 0.054-0.212 / 0.046-0.190 / 0.038-0.187 | 0.068-0.232 / 0.065-0.207 / 0.052-0.196 |
c3@1.0 is 0 everywhere except one cubic poly cell (0.11).

### 10.2 What it says
- The SVD gap cannot tell symmetric from broken data: win 0.11-0.56 in all 16 cells (never near 1), and scaling is declared
  as often or more often on the broken data. The earlier cubic detections (8.4) are not evidence.
- A test of the specific candidate (scaling) works above a floor: eps_eff of g_hat recovers eps = 0.1 almost exactly
  (0.08-0.095) and is clearly above the cubic value (0.024-0.041) in every cell; the power law rejects u^3 for eps = 0.1.
  eps = 0.03 is not separable: stage 1 itself adds a fake breaking of eps_eff about 0.03 (the stray c1, c4, c5 terms).
  Resolution of the candidate test: between eps 0.03 and 0.1 (15-50% of the cubic term).
- The floor is about the same at 1% and 3% noise (poly N = 1000: 0.024 vs 0.028) -> noise is probably not the limit.
- err_g is similar for all three data sets, so stage 1 learns them equally well; the comparison is fair.
- Why the two differ: the gap compares all 30 directions at once and depends on the whole spectrum; eps_eff asks one question
  about one candidate. Consequence for the method: use the SVD only to propose candidates, and decide on each candidate
  with a separate test (breaking measure or held-out fit).
- Caveat: for poly the seeds agree closely, so each cell is about 3 independent data draws; single win values are rough.
- Prediction record: I guessed "poor separation at 0.03, maybe slightly better at 0.1" for the gap; the 0.1 part was wrong
  (no better). No prediction from the user recorded.

### 10.3 Next
Noise-free stage 1 (sigma = 0, 72 models: g0 and cubic, N 200 and 1000, both arms) to see whether the eps_eff floor of about
0.03 drops without noise (my prediction: about the same). Then candidate testing with the symmetry loss (roadmap step 4).
