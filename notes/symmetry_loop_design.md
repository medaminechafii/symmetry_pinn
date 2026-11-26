# Design proposal: the symmetry loop (propose from the SVD, enforce, test, repeat)

Status: PROPOSAL for review (2026-11-26). Nothing here is implemented yet.
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
