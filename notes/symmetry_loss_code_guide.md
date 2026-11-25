# Symmetry loss and weight-decay baseline: what the code does and why

This guide covers the code written for stage 2 (the symmetry loss) and the weight-decay baseline. It walks through:

- what each part computes;
- why it was built that way;
- how a run flows from the command line to the analysis tables;
- where it fits in the larger plan.

Commits: `aef0936` (symmetry loss), `e712c56` (select runs by μ), `be77af8` (sym_coeffs), `0e3dc1d` (weight decay).

---

## 0. The larger plan

The thesis question: *can a PINN use symmetries when we do not know them in advance?* We go there in steps. Each step answers one question before the next one depends on it.

| Stage | Question | Status |
|---|---|---|
| 1 | Learn ĝ and ν̂ from data alone: how good are they? | done. err_g ≈ 0.2, and detection on the learned equation fails (count 2 everywhere) |
| 0 | Why does detection fail? | done. ĝ breaks the symmetry at the 1–2 % level (mostly c5), and the SVD is sensitive at that level |
| **2a** | **If we *knew* the true generators, would forcing them improve ĝ?** | **μ pilot done; this guide** |
| **2b** | **Does a generic regulariser (weight decay) do the same?** | **running now** |
| 2c | False-positive control: does the loss "accept" a wrong symmetry on the ε datasets? | later |
| 3 | Discovered generators: take candidates from the SVD, add them one at a time, stop when held-out fit breaks | later. This is the real goal |

**Why 2a comes first, even though it uses knowledge we will not have.** It is an *upper bound*. If even the true generators do not help, then discovered generators (which are noisier) cannot help either, and stage 3 is pointless. If they help, we know how much there is to gain, and we know the loss code is correct before we feed it uncertain candidates.

**Why 2b is needed.** In g0 the true answer is ĝ = 0. *Any* penalty that shrinks ĝ moves it toward the truth. To claim the improvement comes from the *symmetry*, we must show that a plain shrink penalty does worse. Weight decay is that plain penalty.

---

## 1. The mathematics behind the loss

The equation, with a learned source and viscosity:

    Δ = u_t + u u_x − ν u_xx − g(u)

A generator `V = ξ ∂x + τ ∂t + φ ∂u` is a symmetry when its prolongation annihilates Δ on solutions:

    pr V(Δ) = λ Δ          (λ is a function, the "multiplier")

with

    pr V(Δ) = φ^t + φ u_x + u φ^x − ν φ^xx − g'(u) φ
    φ^x  = D_x φ − u_x D_x ξ − u_t D_x τ
    φ^t  = D_t φ − u_x D_t ξ − u_t D_t τ
    φ^xx = D_x φ^x − u_xx D_x ξ − u_xt D_x τ

**The residual we penalise** at a collocation point is

    S_V = pr V(Δ) − λ Δ

**Why subtract λΔ, and not just use pr V(Δ)?** The network is not an exact solution, so Δ ≠ 0 at the collocation points. For the true generators, subtracting λΔ cancels *every* term that contains network derivatives. What is left depends only on u and ĝ. Two examples, worked out by hand:

**Scaling** (ξ = x, τ = 2t, φ = −u, λ = −3):

    φ^x  = D_x(−u) − u_x·1 − u_t·0       = −2 u_x
    φ^t  = D_t(−u) − u_x·0 − u_t·2       = −3 u_t
    φ^xx = D_x(−2u_x) − u_xx·1 − 0        = −3 u_xx
    pr V(Δ) = −3u_t − u u_x − 2u u_x + 3ν u_xx + u g'
            = −3(u_t + u u_x − ν u_xx) + u g'
    −λΔ    = +3(u_t + u u_x − ν u_xx − g)
    S      = u g'(u) − 3 g(u)

For the polynomial ĝ = Σ c_k u^k this is **S = Σ (k−3) c_k u^k**. Three things follow:

- c3 has factor 0, so the scaling loss never touches c3. It cannot fix the *size* of the cubic term.
- c4 and c5 are pushed to 0 (factors +1 and +2), and c1 and c2 too (factors −2 and −1).
- ν dropped out completely, so the scaling loss never sends a gradient to ν̂.

Example: c4 = 0.2, at u = 0.3 → S = (4−3)·0.2·0.3⁴ = 0.00162.

**Galilean** (ξ = t, τ = 0, φ = 1, λ = 0):

    φ^x = 0,  φ^t = −u_x,  φ^xx = 0
    S = −u_x + 1·u_x − g'(u)·1 = −g'(u)

The loss is mean(g'(u)²): ĝ must be constant, and since ĝ(0) = 0, ĝ = 0. This is why g0 improved so strongly. It is also why g0 needs the weight-decay comparison: this loss is *almost* a shrink penalty.

**The loss** is the average over the chosen generators of the mean of S² over the 2000 collocation points:

    L_sym = (1/|gens|) Σ_V mean(S_V²)
    L     = L_data + λ_PDE · L_PDE + μ · L_sym

Here μ is the weight we swept (0.01, 0.1, 1).

---

## 2. `pinn/sym_loss.py`, function by function

### `d(f, z)`
A total derivative through the autograd graph. It returns zeros when `f` does not depend on `z`, for example D_x of φ = 1. Without this guard, `torch.autograd.grad` raises an error for constant functions.

### `GENERATORS`
A dictionary `name → (ξ, τ, φ, λ)`. Each entry is a small function of (x, t, u). It contains:

- the true symmetries: `d/dx`, `d/dt`, `galilean`, `scaling`, and `projective` (true for g0 only);
- `not_a_symmetry`, used only in a test, to show that a wrong generator gives S ≠ 0.

### `CASE_GENS`
Which generators `--sym auto` uses for each case:

- g0 → Galilean + scaling;
- cubic → scaling.

d/dx and d/dt are left out because S = 0 for them for every ĝ, so they would add nothing. Projective is left out by default; it is optional.

### `prolong(gen, model, x, t)`
Computes the network jets (u, u_x, u_t, u_xx, u_xt) at the collocation points. It then applies the three prolongation formulas above, line by line (`phi_x`, `phi_t`, `phi_xx`). It is the same formula as `discovery/criterion.py`, but computed numerically with autograd instead of symbolically with sympy.

### `sym_residual(gen, model, g_model, nu, x, t, detach=True)`
Computes S = pr V(Δ) − λΔ at each point. Two details matter:

1. **`detach=True`**: all geometric quantities (jets, φ and its prolongations) are cut out of the graph. The gradient of L_sym therefore goes **only to ĝ** (and to ν̂, but ν cancels for our generators), **never to the network**.
   *Why:* we want the symmetry to correct the *equation* (ĝ), not to distort the *solution* u to satisfy a symmetry. The data fit decides u; the symmetry decides which ĝ is admissible. This keeps the experiment clean: any change in u comes only through the PDE term.
2. **g' with `create_graph=True`**: `u_` is a detached copy of u with `requires_grad=True`. ĝ is evaluated on it, and g' = dĝ/du comes from autograd. `create_graph=True` keeps g' differentiable with respect to the coefficients c_k. Without it, ĝ would get no gradient from the g'φ term.

### `sym_loss(...)`
The average of mean(S²) over the generator list.

---

## 3. How it is hooked into training: `pinn/pinn_oracle.py`

`train_pinn` (line 14) gained four arguments: `sym`, `mu`, `sym_detach`, `g_wd`.

```text
use_sym = bool(sym) and mu > 0            # line 42
use_wd  = g_wd > 0 and g_model is not None # line 46

Adam loop (4000 steps):
    d, r = losses()                       # data MSE, PDE residual MSE
    loss = d + lam*r
    if use_sym: loss += mu * sym_term()   # line 64
    if use_wd:  loss += g_wd * wd_term()  # line 67
    step

L-BFGS closure (up to 2000 iterations): the same three additions   # lines 86, 89

final loss and final sym value are recorded; model.extras stores g_model, sym, mu, sym_hist, sym_final, g_wd
```

**Why the `use_sym` / `use_wd` switches instead of always adding `mu * sym_term()`:**

- With μ = 0 the term would still be *computed*. The extra autograd calls change nothing numerically, but they make every run slower.
- More importantly, the tests can then check **bit-for-bit** that μ = 0 or g_wd = 0 reproduces the stage-1 run exactly. That guarantees stage 1 and stage 2 differ *only* by the new term.

**Why `ĝ` and the network share one optimizer:** this is the stage-1 setup. We change only the loss, so the comparison stays fair.

### Command line → file name → metrics

```text
python -m pinn.pinn_oracle 1 --name cubic --n 200 --sig 3 --k 0 1 2 --seed 0 \
       --g-model poly --learn-nu 0.05 --sym auto --mu 0.1
```

- `--sym auto` → `CASE_GENS[name]` (line 161). You can also list generators by hand.
- The file name gets `_symscaling_mu0.1` (line 162) or `_wd1e-05` (line 163) before `_seed`. Every μ and wd setting therefore has its own model files, and nothing overwrites stage 1. A run whose file already exists is skipped.
- The metrics.jsonl line gets `"sym"`, `"mu"`, `"sym_final"`, `"g_wd"`. For poly, it also already had `"c"` (the learned coefficients), which `sym_coeffs` reads.

---

## 4. Telling runs apart: `pinn/runs.py`

**The problem it solves:** before stage 2, every learned-g analysis selected rows with roughly `learn_nu and g_model == poly`. After the pilot, metrics.jsonl holds four runs (stage 1 plus μ = 0.01, 0.1, 1) for *the same* (case, N, σ, k, seed). Without a filter, the analysis would silently mix them, or the last one would overwrite the others in a dictionary.

- `mu_of(r)`, `wd_of(r)`: read μ and wd from a row. Old rows that lack the field count as 0, so stage 1 stays selectable without rewriting old metrics.
- `load_rows(arm, mu=0, names, wd=0)`: the single place where analyses select runs. The defaults (μ = 0, wd = 0) mean that **every existing analysis keeps seeing exactly stage 1** unless asked otherwise.
- `sym_tag`, `wd_tag`, `stem(r)`: rebuild the model file name from a metrics row, the same way `pinn_oracle` wrote it, so analyses can load `..._g.pt` or the network `.pt`. Before, each script built the name by hand; now there is one function, so a future change to the naming cannot break the scripts one by one.

The analysis scripts were patched to use `load_rows` and `stem`, and gained `--mu` (and in `analyse_learned_g_all`, `--wd`):
`analyse_learned_g_all`, `stage0b_general`, `raw_vs_sub_learned`, `deflated_count_check`, `derivative_powerlaw_check`, `eps_eff_check`.

Three scripts are stage-1-only by design: `attribution_check`, `stage0b_learned`, `stage0_floor`. They got an explicit "stage 1 only" filter instead.

(You later removed the μ and wd conditions from `stage0_floor.check_vs_metrics` in 088c8c1. That is correct: it already requires `g_model is None`, which excludes all learned-g rows.)

---

## 5. The tests: `tests/test_sym_loss.py`

Each test checks one claim we rely on in the analysis:

| Test | Claim it protects |
|---|---|
| `test_translations_give_zero` | d/dx and d/dt give S = 0 for any ĝ, so leaving them out of `CASE_GENS` loses nothing |
| `test_galilean_is_minus_g_prime` | the code gives S = −ĝ'(u), the hand formula of §1 |
| `test_scaling_is_sum_k_minus_3` | the code gives S = Σ(k−3)c_k u^k, the hand formula of §1 |
| `test_prolongation_matches_sympy_criterion` | an **independent** check: the autograd prolongation equals the sympy criterion of `discovery/criterion.py` for all six generators, at random network jets. Two separate implementations agree, so the formula is not just checked against itself |
| `test_gradient_reaches_g_only_and_pure_cubic_is_free` | with detach, the network gets no gradient; c3 gets exactly 0 gradient from the scaling; a pure cubic gives loss 0 |
| `test_not_a_symmetry_is_nonzero` | the loss can actually tell: a wrong generator gives S ≠ 0 |
| `test_mu_zero_reproduces_the_run_without_sym_loss` | μ = 0 gives the identical training history and coefficients; μ = 1 changes them |
| `test_g_wd_zero_reproduces_and_wd_shrinks_g` | wd = 0 gives the identical run; a large wd shrinks Σc² by more than 10× |

Run them with `python -m pytest tests/test_sym_loss.py -q` (8 pass).

---

## 6. The weight-decay baseline

    L = L_data + λ_PDE·L_PDE + g_wd · Σ_k c_k²     (for MLP: the sum of squares of all its weights)

- It is an explicit term in the loss, not Adam's `weight_decay` option, because the L-BFGS stage has no such option. With an explicit term, both optimizers minimise the same objective.
- **Choice of values** {1e-6, 1e-5, 1e-4}: the 3 % noise variance is ≈ 3.6e-5, and for cubic Σc² ≈ 1. So the penalty goes from about 3 % of the noise level (barely felt) to about 3× the noise level (strong enough to pull c3 visibly). This brackets the range where the penalty can matter without obviously ruining the fit.

**What would separate the two penalties** (predictions written before the runs):

| | symmetry loss (scaling) | weight decay |
|---|---|---|
| c1, c2, c4, c5 | → 0 (seen: \|c\| < 0.03 at μ = 1) | shrink only as much as c3 does |
| c3 | free: stays where the data put it | pulled toward 0 |
| g0 | ĝ → 0 | ĝ → 0, probably as well, since 0 is the truth |

If weight decay matches the symmetry loss in g0, the g0 gain is not evidence for symmetry. If in cubic it cannot clear c4 and c5 without also biasing c3, that shows what a symmetry adds: it removes only the parts of ĝ that break the symmetry.

**`run_wd_pilot.sh`** is the same grid as the μ pilot (poly, N = 200, 3 %, k 0–2, seeds 0–2, ν̂ from 0.05), with `--g-wd` in place of `--sym/--mu`. 18 xargs tasks × 3 models = 54 models, 6 in parallel, `OMP_NUM_THREADS=1` so the 6 processes do not fight over cores.

---

## 7. Reading the results: `experiments/sym_coeffs.py`

`analyse_learned_g_all` gives medians of err_g. They hide *why* the error is what it is. `sym_coeffs` prints, for every run:

- the five coefficients;
- ν̂;
- data/noise: whether the extra penalty damaged the fit (≈ 0.8–0.95 means no);
- μ·sym_final: how large the symmetry term is at the end;
- wd·Σc²: how large the weight-decay term is at the end.

```text
python -m experiments.sym_coeffs                      # stage 1 and μ = 0.01, 0.1, 1
python -m experiments.sym_coeffs --wd 1e-6 1e-5 1e-4  # stage 1, then each wd
```

What the μ table showed, and what to compare the wd table against:

1. At μ ≥ 0.1, c1, c2, c4 and c5 are ≈ 0. The loss does its job.
2. The cubic error left is **c3 alone**, and it depends on the **data draw k**, not the seed:
   - k=0: c3 ≈ −0.85, ν̂ 0.102;
   - k=1: c3 ≈ −1.19, ν̂ 0.096;
   - k=2: c3 ≈ −0.98, ν̂ 0.100.
3. c3 and ν̂ trade off along one line (Δc3 ≈ 50–70·Δν). Both damp the solution, so the data can swap one for the other. A symmetry cannot fix this, because scaling holds for any c3 and any ν.
4. Seeds within one k agree to about 0.01. So each cell has **3 independent samples, not 9**. Keep this in mind when quoting medians and ranges.

---

## 8. What comes next, and why in this order

1. **Weight-decay results** (running). This decides whether the g0 result counts.
2. **Notes update**: add the μ table, the c3–ν finding and the wd comparison to `week6_learned_g_stage1.md` (or a new stage-2 file).
3. **False-positive control** on the ε datasets (g = −u³ + εu). Scaling is *not* a symmetry there (S = −2εu). With a large μ, does the loss force ĝ to a pure cubic anyway, and does the held-out fit get worse? Stage 3 needs this: its stopping rule ("add candidates until held-out fit breaks") only works if a wrong symmetry visibly hurts the fit.
4. **Alignment check**: do singular vectors 3–5 of the learned equations point near the true scaling, Galilean and projective generators? If they don't, stage 3 has nothing good to add.
5. **Stage 3, discovered generators**: replace `GENERATORS[...]` with candidate vectors from the SVD, using the on-solution residual A·c, so no multiplier λ is needed. Add candidates one by one with a large μ, and stop when the **held-out** data error rises above the noise level. Fix the stopping rule before running.

The code is set up for step 5: `sym_residual` only needs (ξ, τ, φ, λ) as functions. A discovered generator is a linear combination of the 30 basis fields, so it can be wrapped in the same interface.
