# Design proposal: the symmetry loop (propose from the SVD, enforce, test, repeat)

Status: PROPOSAL for review (2026-11-26). Sections 1, 2.5-2.7 and 8.3 are SUPERSEDED by section 9 (2026-11-27);
the parts 2.1-2.4 (held-out, proposing, cleaning, general loss) still apply.
Idea: the user's (one candidate at a time from the SVD, enforce it, keep it if it does not hurt, re-check, continue).
Evidence it can work: section 11 of week6_learned_g_stage1.md (the true symmetries are within the first 2-3 SVD candidates).
Evidence for the test step: section 10 (the SVD cannot reject a false symmetry, so something else has to).

---------------------------------------------------------------------------------------------------------------------

## 1. The loop in one view

```
0. split the data: training points + held-out points (fixed split, the same for every run of this data set)
1. stage 1 on the training points            -> model M0 (g_hat, nu_hat); held-out error E0
   accepted = []                              (list of accepted generator vectors)
2. ROUND: SVD of the CURRENT learned equation (network jets, current g_hat, nu_hat)
          drop d/dx, d/dt; remove the directions already accepted
          candidates = the next 2 singular vectors (A, then B)
3.   for each candidate v (A first):
         train from scratch with  L_data + lambda L_PDE + mu * sum over (accepted + [v]) of L_sym(v)
         held-out error E_v
         if E_v <= (1 + delta) * E_current:  accept v, current model := this one, go to 2 (next round)
4.   no candidate accepted in this round -> STOP
5. result: the last accepted model (its g_hat, nu_hat) and the list of accepted generators
```
Limits: at most 3 rounds (our 30-field basis has at most 3 non-trivial symmetries for these equations).

## 2. The parts, one by one (what / why / where the math comes from / is it needed)

### 2.1 Held-out points
What: before training, set aside part of the observation points. They are never used in training; at the end we compute
the mean squared difference between the network's u and the observed u at those points (same formula as L_data).
Why: the decision must come from the data, not from the SVD (circular) and not from the training loss (the network
already fits the training data down to the noise, so a true symmetry cannot lower it; section 9 / mu pilot).
Two ways to choose the held-out points:
- (a) random 20% of the points.
- (b) RECOMMENDED: a time window, all points with t > 1.5 (about 25%). The PDE loss still covers all t, so in that window
  the network's u is mainly produced by the learned equation ("predict the future"). A wrong g_hat should show there much
  more than at random points in between training points, where the data alone pin u down.
Needed: yes. Code: a --holdout option in pinn_oracle (split and stored error), small.

### 2.2 Proposing candidates
What: the same 30-column SVD as sections 8-11, but with the NETWORK's jets (u, u_x, u_xx, u_xxx by autograd) instead of
the true jets, because in real use the true solution is unknown. Drop the two numerically zero values (d/dx, d/dt).
Remove the already accepted directions (project them out), so the same symmetry is not proposed twice.
Why "remove accepted": after we enforce A, the SVD will show A as an exact symmetry, because we forced it (circular).
It carries no new information, so it is removed, and the next candidate is the next new direction.
About the idea "re-check the SVD and A gets refined": for the same reason this does not happen by itself. Enforcing v
makes v the null vector of the new equation, so the SVD gives v back unchanged. If the candidates need cleaning, see 2.3.
Needed: yes. Code: reuse the functions of stage0b_general / raw_vs_sub_learned (network jets exist there).

### 2.3 Optional: cleaning a candidate (sparsification)
What: write the candidate in the original coefficients (xi, tau, phi on 1, x, t, u, ...), scale so the largest is 1, set
the components below 0.1 to 0, renormalise.
Why: the candidates are 5-30 deg off the true symmetries (section 11), probably because of many small wrong components.
True symmetries are simple (scaling has 3 non-zero entries: x d/dx + 2t d/dt - u d/du). Enforcing a slightly wrong
generator may hurt the fit and get a true symmetry rejected.
Assumption it adds: "symmetries are sparse in the basis". Common (LieNLSD sparsifies too), but it is an assumption.
Can be checked BEFORE any training, data-free: rerun the alignment check with cleaned candidates and see if res drops.
Needed: optional. Decide after that cheap check.

### 2.4 The general symmetry loss for a candidate vector v
What: the SVD matrix has one column per basis field; each column is the substituted symmetry condition of that field,
evaluated with g_hat, g_hat', nu_hat. A generator v (30 numbers) is a symmetry of the learned equation exactly when the
combination sum_i v_i * column_i is zero at every point. So the loss is
    L_sym(v) = mean over the collocation points of ( sum_i v_i * column_i / scale_i )^2
where scale_i is the RMS size of column i, fixed at the moment the SVD was taken (the same normalisation the SVD used,
so v means the same thing in the loss as in the SVD).
Where it comes from: this is exactly the condition the SVD tests (no new math). It is linear in v, which is also why a
mixture of symmetries is a symmetry. The substituted form has no multiplier lambda, so it works for any v, not only for
the hand-written scaling and Galilean; projective is included automatically.
As in the mu pilot, the network jets are detached: the gradient reaches only g_hat and nu_hat.
Checked already: the 30 columns evaluate in torch with gradients; the scaling vector gives 3e-16 on the exact cubic
equation, the Galilean vector does not.
Validation before use (tests):
- v = d/dx gives L_sym = 0 for any g_hat.
- v = exact scaling vector: training with it must give what the hand-written scaling loss gave in the mu pilot
  (c1, c2, c4, c5 -> about 0, c3 free). This also fixes mu for the new loss (its size differs from the old one).
Needed: yes; this is the main new code.

### 2.5 Training each candidate
From scratch, same seed and same split as the baseline, so baseline and candidate differ ONLY by the added loss.
(Continuing from the previous model would be cheaper but then the comparison depends on the history.)

Saving and going back (user's point): every trained model is saved (network, g_hat, nu_hat, the split, the list of
generators it was trained with). "current model" is only replaced when a candidate is ACCEPTED. On rejection the loop
goes back to the saved current model and its accepted list, so a false candidate leaves no trace in later rounds.
With training from scratch this is automatic (the rejected model is simply not used further); with continued training
(option 5 in section 7) it is essential: save a checkpoint before trying a candidate and restore it on rejection.

### 2.6 The decision rule
R = E_v / E_current (held-out error with the candidate / without). Accept if R <= 1 + delta.
delta must be fixed before the real runs. Proposal: a small pilot on ONE data draw of cubic (k = 0) with two known
anchors: the exact scaling vector (true -> R should be about 1 or below) and the Galilean vector (false for cubic: it
forces g = 0 -> R should be clearly above 1). Set delta between them, then never change it for the other runs.
All R values are reported anyway, so the reader can see how clear the separation is.
DECIDED (user, 2026-11-26): delta is fixed after some experimental runs (pilot) and written down before the real
loop runs. Pilot = exact scaling on cubic (chance fluctuation of R for a true symmetry), Galilean on cubic as the
clearly false anchor. Caveat: the pilot does not measure nearly-true false symmetries (eps 0.1); the evaluation does.

### 2.7 Stop rules
Try at most 2 candidates per round (A, then B); stop when neither is accepted; at most 3 rounds.

## 3. How to show it works (evaluation)
Data sets with a known answer:
| data | expected accepted |
|---|---|
| g0 | 3 directions (scaling, Galilean, projective, possibly as mixtures) |
| cubic | 1 (scaling) |
| eps = 0.1 | 0 |
| eps = 0.03 | 0 by truth; probably 1 in practice (below the resolution found in section 10) |
For every run report: accepted vectors (and, for reading only, their res to the true symmetries), all R values, and the
final err_g and nu error compared with stage 1 and with the true-generator runs (mu pilot).
The thesis claim is supported if: g0 and cubic get their true symmetries, eps 0.1 gets none, and the final err_g is
close to the true-generator result.

## 4. Cell and cost
Start: poly arm (coefficients are readable), N = 1000, sigma = 1% (best candidates in section 11), k = 0-2, seed 0 only
(poly seeds agree to about 0.01). MLP and harder cells later if it works.
Trainings per (data set, k): cubic about 4, g0 about 6, eps about 3 -> about 60 trainings + about 6 for validation/pilot.
Each (data set, k) chain runs sequentially; 12 chains on 6 cores.

## 5. Order of work
1. General symmetry loss L_sym(v) in torch + tests (2.4).
2. --holdout in pinn_oracle, held-out error in metrics (2.1).
3. SVD from network jets + "remove accepted" (2.2).
4. (optional) data-free check of the cleaning (2.3).
5. Pilot: validation with the exact scaling vector (fixes mu), anchors for delta (2.6).
6. Loop driver discovery/sym_loop.py: one (data set, k) chain end to end, every round logged to a jsonl.
7. Full evaluation (3).

## 6. Risks
- A wrong symmetry may be cheap (the c3 - nu trade-off of section 9): the held-out error may hardly rise for eps 0.03 or
  even 0.1. The time-window split is chosen to make this less likely; the eps 0.1 row will show it.
- Candidates are 5-30 deg off: a true symmetry could be rejected because the candidate is not exact (2.3 helps).
- Network jets instead of true jets: candidates may be somewhat worse than in section 11.
- delta is a choice; fixed in advance on cubic k = 0 only.

## 7. Decisions for the user
1. Held-out: time window t > 1.5 (recommended) or random 20%?
2. Cleaning (2.3): check it first data-free, or leave it out?
3. Start cell: N = 1000, 1%, poly (recommended), or noise-free?
4. delta: from the two-anchor pilot (recommended), or a fixed number such as 0.1?
5. Training each candidate from scratch (recommended) or continuing from the previous model?

## 8. Decisions and implementation log
2026-11-26 (user): held-out = time window t > 1.5 (260 of 1000 points for k = 0); each candidate trained FROM SCRATCH;
delta fixed after a pilot (section 2.6), written down before the loop runs. Still open: cleaning (2.3), start cell.

Implemented (2026-11-26):
- pinn/sym_general.py: general loss L_sym(v) (section 2.4). `python -m pinn.sym_general --test`, all OK:
  d/dx, d/dt give 0 for any g (and zero gradient); on exact cubic scaling gives 1e-32, Galilean / projective > 0;
  on exact g0 all five give ~1e-33. For a perturbed polynomial g the general residual is EXACTLY proportional to the
  hand-written one (misfit 1e-15): scaling 1.90 x (u g' - 3 g), Galilean 1.31 x (-g'). So L_general = 3.6 x L_old for
  scaling: mu = 1 here is about mu = 3.6 of the mu pilot (where mu >= 0.1 already cleared c1, c2, c4, c5). mu = 1 is used.
- pinn_oracle: --holdout T (trains on t <= T; stores err_hold vs the noisy held-out observations and err_hold_clean vs
  the clean u), --gen-file (general loss); runs.load_rows(holdout=None) keeps holdout runs out of all older analyses.
- discovery/make_gen_file.py: column scales from the baseline model (network jets at its collocation points), fixed.
- run_delta_pilot.sh (12 trainings) + experiments/delta_pilot.py (summary).
Observation from a short smoke run (300 Adam steps, not a result): at sigma = 1% the held-out error against the noisy
observations is dominated by the noise (3.7e-6 vs 5.8e-8 against the clean u). So R based on noisy observations is
pulled toward 1; a wrong g_hat must change u a lot in t > 1.5 to show. The pilot will show whether this is a problem.

### 8.1 Decision rule FIXED (user, 2026-11-26, from the delta pilot, section 12 of week6_learned_g_stage1.md)
- Two settings, sigma 1% and noise-free, each with its own delta; both turned out equal:
  reference E_current = the BETTER (smaller held-out error) of 2 baseline seeds; accept a candidate if R <= 1 + delta,
  delta = 0.1 (sigma 1% and sigma 0, poly arm).
- Pilot values with this reference: scaling (true) R 0.95-0.99 (1%), 0.05-0.67 (0%); Galilean (false) R 11-12 (1%),
  689-10626 (0%). Seed-1 baseline vs seed-0 differed by up to 12x without noise: why one baseline is not enough.
- Development runs: candidates trained with 1 seed (faster; known small asymmetry against the candidate).
  FINAL evaluation runs: candidates ALSO best of 2 seeds (fair comparison, about twice the cost).
- Allowed before the real loop runs: more baseline seeds (3 would be better). NOT allowed after seeing loop results:
  any change of delta or of the reference rule.
- MLP arm: own pilot (run_delta_pilot.sh MLP), delta fixed the same way before any MLP loop run.

### 8.2 g0 one symmetry at a time (user's idea; "perfect proposals" reference for the loop)
run_g0_steps.sh, summary experiments/g0_steps.py. g0, poly, N 1000, held out t > 1.5, sigma 1% and 0, k 0-2.
Same general loss (no new math), TRUE generators chosen by hand: scaling -> scaling+Galilean -> all three (cumulative,
the order the loop would add them), then Galilean alone and projective alone (optional: which symmetry does the work).
With several generators the loss is the MEAN over them, so each gets weight mu/m. Reference = better of 2 baseline seeds.
Measures: R, err_g (true g = 0), nu error. 42 trainings. Runs after the MLP delta pilot (run_overnight.sh).

### 8.3 Decision rule UPDATED (user, 2026-11-27, option a; after section 13 of week6 notes, before any loop run)
Problem found: noise-free, once g_hat is exact the held-out error sits at the optimiser floor (1-5e-9) and moves by
up to ~50% between runs; compared with the PREVIOUS model, 4 of 6 true later steps on g0 had R 1.16-1.49 (rejected).
Rule (all rounds, sigma 1% and 0, poly and MLP): every candidate is compared with the BASELINE (better of 2 seeds,
no symmetry loss), not with the previous model:
    R = E(accepted symmetries + candidate) / E(best baseline),   accept if R <= 1.1.
Pilot / g0 values with this rule: true steps 0.914-0.992 (1%), 0.010-0.119 (0%); Galilean on cubic 11-12 (1%), >= 380 (0%).
Known weakness: if earlier true symmetries lowered the error a lot (noise-free: up to 100x), a later MILDLY false
symmetry can raise it again and still stay below 1.1 x baseline. Very wrong ones are still caught.
Deferred (user, for later): false symmetry AFTER a true one (cubic: scaling+Galilean, scaling+projective vs scaling;
12 trainings) -> shows how large this weakness is. The eps data in the final evaluation test the mild case.


## 9. THE LOOP, final version (user's design, 2026-11-27). Supersedes sections 1, 2.5-2.7 and 8.3.
Idea (user): every candidate gets its OWN training "baseline + that one symmetry" and is compared with the baseline;
the accepted ones are then trained together. A better model may reveal new candidates -> repeat in passes.

```
parameters: max_rejections (2), max_candidates (5), max_passes (2), delta (0.1), mu (1), n_base_seeds (2)

0. split: train on t <= 1.5, held out t > 1.5
1. BASELINE B: stage 1 (L_data + lambda L_PDE), n_base_seeds seeds, keep the one with the smaller held-out error E_B
   accepted = [],  model M = B
2. PASS p = 1, 2, ..., max_passes:
     a. SVD of model M (network jets, g_hat, nu_hat); drop d/dx, d/dt; remove the directions in `accepted`
        candidates = the next singular vectors, in order (smallest singular value first)
     b. for each candidate v, until max_rejections rejections or max_candidates tried:
          train B's setup + mu L_sym(v)   (from scratch, same seed and split; v ALONE)
          R = E_v / E_B;   accept (add v to `accepted`) if R <= 1 + delta, else reject
     c. if no new candidate was accepted in this pass -> STOP
     d. COMBINED training: B's setup + mu L_sym(all accepted)   -> new model M (u, g_hat, nu_hat all retrained)
        final check: R_comb = E_M / E_B <= 1 + delta; if it fails, report it (a candidate that passed alone is
        suspicious together) and keep the previous M
3. RESULT: the last M (g_hat, nu_hat) and the list `accepted`
```
Why tested alone against the baseline: this is exactly the situation the pilot calibrated (true 0.93-1.00 at 1%,
0.01-0.17 noise-free; false >= 11). No comparison between two models that are both at the optimiser floor (the
noise-free problem of section 13.2), and a false candidate cannot hide behind the gains of earlier true ones.
Why passes: the combined model has a better g_hat, so its SVD can show symmetries the baseline's SVD missed
(g0: once Galilean forces g_hat = 0, scaling and projective become exact). New candidates are still tested alone
against B: the better model only supplies better candidates, the test stays the calibrated one.
Removing accepted directions: they appear exact in the SVD of M only because they were enforced (circular).
Consequences of accepted symmetries may appear as new candidates; they are tested like any other candidate.
Expected: cubic -> scaling in pass 1, nothing in pass 2; g0 -> up to 3 directions; eps 0.1 -> none.
Saving: every training is saved (network, g_hat, nu_hat, split, generators), every decision logged (jsonl).
Development runs: 1 seed per candidate; final evaluation: candidates best of 2 seeds (section 8.1).
Deferred: false symmetry after a true one (cubic scaling+Galilean / scaling+projective) now mainly tests the combined
check of step d.

## 10. Implementation (2026-11-27, branch loop-prototype)
- discovery/sym_loop.py: one chain (name, N, sigma, k). Trainings via pinn.pinn_oracle (subprocess; existing runs are
  skipped and read back, so the pilot baselines are reused). Column scales fixed from B for the whole chain.
  Candidates: SVD of M's normalised columns restricted to the orthogonal complement of {d/dx, d/dt, accepted}
  (exact projection instead of "drop the 2 smallest"). If only one symmetry is accepted in total, its own run is the
  combined model (no duplicate training). Log: results/oracle/loop/<chain>.jsonl.
- run_loop.sh: 24 chains (cubic, g0, cubic_e01, cubic_e003 x sigma 1, 0 x k 0-2), poly, defaults of section 9.
- experiments/loop_summary.py: per chain R and decision of every candidate, combined checks, final nu error / c,
  res of the true generators to the accepted span.
Smoke tests (Mac, 50 Adam steps, not results): all paths run (baseline, SVD, accept, reject, combined, pass 2, stop).
One observation to watch in the real runs: in a 50-step cubic noise-free test the first candidate was within
res 0.02 of scaling but gave R = 3.3 (rejected). Likely the under-trained models, but an inexact candidate enforced
with mu = 1 could also be rejected in the real runs (risk in section 6; cleaning 2.3 would be the fix).

## 11. First loop results and the cleaning test (2026-11-27)
Ubuntu run (uncleaned, partial): cubic candidate 1 is close to scaling (res 0.04-0.13; 0.40 once) but REJECTED in 5 of 6
chains (R 4.3-19 noise-free, 1.7-4.3 at 1%; one accepted at 1.04). Exact scaling in the pilot: 0.05-0.15 / ~0.95.
Wrong candidates (res ~1) correctly rejected (R 1.2-5515). g0: candidate 1 is a mixture of true symmetries (span3
0.004-0.08), accepted in all 5 finished first steps (R 0.39-1.005).
Data-free cleaning check (discovery/clean_check.py, thr 0.1): cubic candidate 1 -> exactly xi:x, tau:t, phi:u kept,
res 0.009-0.018 (raw 0.035-0.133) in 5 of 6 chains; kept ratios 0.49-0.50 : 1 : -0.51..-0.53 (exact 0.5 : 1 : -0.5).
Training test on the Mac (--clean 0.1, cubic k0, pass 1; Mac retrains its own baselines, so not identical to ubuntu):
| | res raw -> cleaned | R cleaned (ubuntu raw R) | c of the candidate run |
| noise-free | 0.065 -> 0.008 | 2.37 (19.4) | c3 -1.011, c5 0.066 |
| 1% | 0.127 -> 0.017 | 2.15 (1.72), R_clean 27 | c3 -0.876, c5 0.108 |
Still rejected. Cleaning removes the extra components but not the ~4% error in phi:u. For u_t + u u_x = nu u_xx + g a
generator a x d/dx + b t d/dt + c u d/du can only be a symmetry if b = 2a and c = -a (u_t, u u_x, nu u_xx must scale
alike); with c = -1.04 a no g works, so the loss still bends g_hat (c5 not cleared, c3 off at 1%).

### 11.1 Full ubuntu run (24 chains, uncleaned, 2026-11-27)
Accepted per chain (k = 0, 1, 2):
| data | expected | noise-free | 1% |
|---|---|---|---|
| cubic | 1 (scaling) | 0, 0, 0 | 0, 0, 1 |
| g0 | up to 3 | 1, 1, 1 (mixture of Gal./proj.; scaling cand. rejected R 27-30) | 2, 3, 3 (k1, k2: all three, res 0.02-0.07) |
| eps 0.1 | 0 | 0, 0, 0 | 0, 0, 0 |
| eps 0.03 | 0 (probably 1) | 0, 0, 0 | 0, 0, 1 |
Scaling-like candidate 1, noise-free (res to scaling -> R):
  cubic (TRUE):    0.08 -> 19.4,  0.04 -> 7.0,  0.05 -> 4.3
  eps 0.03 (false): 0.11 -> 25.3, 0.04 -> 3.6,  0.17 -> 21.3
  eps 0.1 (false):  0.23 -> 77.3, 0.12 -> 12.7, 0.25 -> 44.4
-> R grows with the candidate's distance from scaling, whether scaling is true or not. A higher noise-free threshold
cannot separate them: cubic true up to 19.4, eps 0.1 false from 12.7. At this candidate accuracy (4-15 deg) the test
cannot tell "true but inexact" from "false but near".
eps 0.1: correctly 0 accepted everywhere (but cubic also mostly 0, so not yet evidence of discrimination).
g0 at 1% works best: true candidates R 0.99-1.09, wrong ones 1.9-20.
Next proposed: trainable candidate coefficients (option b): with the support fixed to the cleaned components, the
PDE terms alone force exact ratios (b = 2a, c = -a for scaling), so a true candidate should become exact (R -> pilot
0.05-0.15), while for eps 0.1 the source term cannot follow (R should stay high).

## 12. Learnable candidate coefficients + drift limit (Mac test, 2026-11-26/27, noise-free, one data draw each)
Code: pinn.sym_general.LearnableGen (non-zero components of the cleaned/sparsified candidate trained with the model,
renormalised every step); sym_loop --prep clean|sparse --learn --max-drift 10 (reject if the trained vector moved
> 10 deg: it was replaced, not corrected) + duplicate check after training. Chains run before the drift rule was
added were re-evaluated with it from their logs.
| chain | accepted (new rules) | what |
|---|---|---|
| cubic k0, clean | 1 | exact scaling (res 0.000, drift 0.5 deg, R 0.042; was R 19.4 without learning) |
| cubic k0, sparse | 0 | rotation put a wrong vector first; scaling only reached by a 41 deg drift (rejected) |
| eps 0.1 k1, clean | 0 | candidate trained to EXACT scaling (drift 1.7 deg) -> R 2325 (was 12.7) |
| eps 0.1 k1, sparse | 0 | R 49268 / 2057 |
| g0 k0, clean | 2 | a Gal./proj. mixture + exact scaling; final g_hat = 0, nu err 0.005% (base 0.136%) |
| g0 k0, sparse | 3 | exact scaling, exact Galilean, exact projective (res 0.000 each; drift 2.5, 9.4, 0.0 deg); final g_hat = 0, nu err 0.002% |
Key point: refining the candidate removes the overlap between "true but inexact" and "false but near": the true
candidate becomes exact and fits (R 0.04), the false one becomes exact scaling and cannot fit the +0.1u term (R 2325).
Drift: good corrections 0.0-9.4 deg, replacements 12-90 deg; the 10 deg limit is a choice (9.4 and 12.5 are close).
Clean is better for cubic, sparse for g0 (mixtures). Open: noise (1%), all draws, choice/union of clean and sparse.

## 13. Full run, tier 1 (ubuntu, 2026-11-26/27): poly, N 1000, noise 0 and 1%, k 0-2, learnable coefficients, drift 10 deg
(python -m experiments.loop_eval --arm poly --N 1000; 48 chains, no crashes)
| noise | prep | cubic acc (exp 1) | g0 acc (exp 3), all found | eps 0.1 false acc | eps 0.03 acc | err_g cubic | err_g g0 | nu err % cubic / g0 |
|---|---|---|---|---|---|---|---|---|
| 0 | clean | 1,1,1 | 3,3,2 (2/3) | 0/3 | 0,0,0 | 0.0228 -> 0.0024 | 0.0214 -> 0.0000 | 0.023->0.028 / 0.148->0.007 |
| 0 | sparse | 0,1,1 | 3,0,3 (2/3) | 0/3 | 0,0,0 | 0.0228 -> 0.0042 | 0.0214 -> 0.0000 | 0.023->0.025 / 0.148->0.026 |
| 1 | clean | 1,0,1 | 3,2,3 (2/3) | 0/3 | 0,0,0 | 0.0325 -> 0.0054 | 0.0201 -> 0.0001 | 0.423->0.243 / 0.361->0.126 |
| 1 | sparse | 0,0,1 | 3,3,2 (2/3) | 0/3 | 0,0,0 | 0.0325 -> 0.0325 | 0.0201 -> 0.0000 | 0.423->0.283 / 0.361->0.103 |
- No false acceptance in any of the 24 eps chains (eps 0.1 AND eps 0.03: the old SVD-gap test could not resolve 0.03).
- Clean + learn: cubic scaling found in 5/6 draws; g0 complete in 4/6 (the others 2 of 3). err_g 6-10x smaller
  for cubic, g0 g_hat -> 0; nu error smaller except cubic noise-free (already 0.02%).
- Sparse is not better overall (cubic worse; g0 k1 noise-free 0 accepted). Clean + learn is the default from here.
- To inspect: the misses (cubic s1 k1 clean, g0 s0 k2 clean, g0 s0 k1 sparse).

## 14. Future experiments (user, 2026-11-27): other PDEs
Planned for later (not started): (1) reaction-diffusion u_t = nu u_xx + g(u): g = -u^3 (scaling x d/dx + 2t d/dt - u d/du,
fixed only by g, no u u_x term -> tests co-adaptation of learnable coefficients and g_hat), eps control g = -u^3 + 0.1u,
optional heat equation g = 0 (many symmetries); (2) Fisher-KPP u_t = nu u_xx + u(1 - u): only d/dx, d/dt -> pure
false-positive test (loop should accept nothing). Needs: solver without u u_x, new criterion columns, PINN residual
form, true symmetries for evaluation. Run small tests on the Mac.
