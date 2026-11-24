"""Stage 0: how well can symmetries be found when the columns are SUBSTITUTED?

Substituted columns (columns(g)): u_t and u_xt are replaced by the equation, using the true nu and g,
so only u, u_x, u_xx, u_xxx are needed.  Raw columns (column_raw(g)): u_t and u_xt are taken as given
(the network's own, or the equation's for the truth source).

For every source of jets (truth, oracle PINN, data-only surrogate) the SVD is built both ways on the SAME
points.  A swap test puts ONE jet of a model into the truth jets to see which jet limits the gap.

Run from the repo root:
    python -m discovery.stage0_floor --check                       # truth-only sanity checks (no models)
    python -m discovery.stage0_floor --cell 1000:1                 # one cell, both cases
    python -m discovery.stage0_floor --cell 1000:1 200:1 --name g0
"""
import argparse
import csv
import json

import numpy as np
import sympy as sp
import torch

from discovery.criterion import columns, u as u_sym
from discovery.fingerprint import (spectral_derivative, evaluate, normalise, count_by_gap,
                                   KNOWN, projection_residual, DATA, N_POINTS)
from discovery.noise_gap import CASES, column_raw, evaluate7, gap_at
from pinn.surrogate import jet, NU, rel_error, RESULTS, MODELS
from pinn.discover_from_surrogate import get_model

KEYS4 = ("u", "u_x", "u_xx", "u_xxx")
N_EXP = 3
REAL = {"g0": ("d/dx", "d/dt", "Galilean", "scaling", "projective"),   # generators that are true symmetries
        "cubic": ("d/dx", "d/dt", "scaling")}


# --- jets ---------------------------------------------------------------------------------------------
def truth_jets(name, k, g):
    """Spectral jets on the dense grid for experiment k, plus u_t and u_xt from the equation."""
    d = np.load(DATA / f"dense_{name}.npz")
    x, t, U = d["x"], d["t"], d["U"][k]                       # U: (n_t, n_x)
    kk = np.fft.fftfreq(len(x), d=x[1] - x[0]) * 2 * np.pi
    J = {"u": U}
    for key, m in (("u_x", 1), ("u_xx", 2), ("u_xxx", 3)):
        J[key] = spectral_derivative(U, kk, m)
    gf = sp.lambdify(u_sym, g, "numpy")
    dgf = sp.lambdify(u_sym, sp.diff(g, u_sym), "numpy")
    gv = np.broadcast_to(np.asarray(gf(U), dtype=float), U.shape)
    dgv = np.broadcast_to(np.asarray(dgf(U), dtype=float), U.shape)
    J["u_t"] = -U * J["u_x"] + NU * J["u_xx"] + gv
    J["u_xt"] = -J["u_x"] ** 2 - U * J["u_xx"] + NU * J["u_xxx"] + dgv * J["u_x"]
    return x, t, J


def model_jets(model, x, t, chunk=4000):
    """Network jets on the same grid, including u_xxx. Returns arrays of shape (n_t, n_x)."""
    X, T_ = np.meshgrid(x, t)
    xf, tf = X.ravel(), T_.ravel()
    keys = ("u", "u_x", "u_t", "u_xx", "u_xt", "u_xxx")
    out = {key: [] for key in keys}
    for i in range(0, len(xf), chunk):
        xb = torch.tensor(xf[i:i + chunk], requires_grad=True)
        tb = torch.tensor(tf[i:i + chunk], requires_grad=True)
        u, u_x, u_t, u_xx, u_xt = jet(model, xb, tb)
        u_xxx = torch.autograd.grad(u_xx, xb, grad_outputs=torch.ones_like(u_xx))[0]
        for key, v in zip(keys, (u, u_x, u_t, u_xx, u_xt, u_xxx)):
            out[key].append(v.detach().numpy())
    return {key: np.concatenate(v).reshape(X.shape) for key, v in out.items()}


def pool(J_per_exp, x, t, it, ix):
    """Pick the same grid points (it, ix: one index array per experiment) and stack the experiments."""
    out = {key: np.concatenate([J[key][i, j] for J, i, j in zip(J_per_exp, it, ix)])
           for key in J_per_exp[0]}
    out["x"] = np.concatenate([x[j] for j in ix])
    out["t"] = np.concatenate([t[i] for i in it])
    return out


def svd_stats(cols, jd, k, evaluator):
    A = np.column_stack([evaluator(c, jd) for c in cols])
    A_n, scale, _ = normalise(A)
    _, s, Vt = np.linalg.svd(A_n, full_matrices=False)
    res = {gn: projection_residual(w, Vt, k, scale) for gn, w in KNOWN.items()}
    return s, gap_at(s, k), count_by_gap(s)[0], res


def sample_indices(rng, n_t, n_x):
    per = N_POINTS // N_EXP
    it = [rng.integers(0, n_t, per) for _ in range(N_EXP)]
    ix = [rng.integers(0, n_x, per) for _ in range(N_EXP)]
    return it, ix


def load_truth(name, g):
    out = [truth_jets(name, k, g) for k in range(N_EXP)]
    x, t = out[0][0], out[0][1]
    for o in out[1:]:
        assert np.allclose(o[0], x) and np.allclose(o[1], t), "x, t differ between experiments"
    return x, t, [o[2] for o in out]


# --- sanity check 1 and 2: truth only -----------------------------------------------------------------
def check_truth(name):
    g, k_true = CASES[name]
    cols_sub, cols_raw = columns(g), column_raw(g)
    x, t, J = load_truth(name, g)
    it, ix = sample_indices(np.random.default_rng(0), len(t), len(x))
    jd = pool(J, x, t, it, ix)
    s_sub, gap_sub, cnt_sub, res = svd_stats(cols_sub, jd, k_true, evaluate)
    s_raw, gap_raw, cnt_raw, _ = svd_stats(cols_raw, jd, k_true, evaluate7)
    diff = np.max(np.abs(s_sub - s_raw)) / s_sub[0]           # absolute, relative to the largest value
    print(f"[check truth] {name}: count sub/raw = {cnt_sub}/{cnt_raw} (expected {k_true})   "
          f"gap sub/raw = {gap_sub:.2f}/{gap_raw:.2f}   max |s_sub - s_raw| / s[0] = {diff:.1e}  (round-off ~1e-15)")
    print("              residuals of true symmetries: " +
          "  ".join(f"{gn}={res[gn]:.1e}" for gn in REAL[name]))


# --- sanity check 3: recomputed errors vs the stored ones ---------------------------------------------
def check_vs_metrics(name, n_obs, sig, oracle_J, truth_J):
    path = RESULTS / "oracle" / "metrics.jsonl"
    if not path.exists():
        print("[check metrics] no metrics.jsonl here, skipped"); return
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if (r["name"] == name and r["N"] == n_obs and r["sig"] == sig and r["lam"] == 1.0 and r["seed"] == 0
                and r.get("g_model") is None and not r.get("learn_nu") and not r.get("mu") and not r.get("g_wd") and r["steps"] == 4000
                and r["ncolloc"] == 2000 and r["feat"] == 1 and not r["cos"] and r.get("loss_g") == name):
            rows[r["k"]] = r                       # later rows overwrite earlier duplicates
    if not rows:
        print("[check metrics] no matching oracle rows, skipped"); return
    worst = 0.0
    for k, r in rows.items():
        for key, mkey in (("u", "err_u"), ("u_x", "err_u_x"), ("u_xx", "err_u_xx")):
            mine = rel_error(oracle_J[k][key], truth_J[k][key])
            worst = max(worst, abs(mine - r[mkey]) / r[mkey])
    print(f"[check metrics] {name} N={n_obs} s={sig}: worst relative difference to stored err_u/err_u_x/err_u_xx "
          f"= {worst:.1e}  (should be ~1e-6 or smaller)")


# --- one cell -----------------------------------------------------------------------------------------
def try_load(name, n_obs, sig, x, t, label, **kw):
    """Models for all experiments, or None (with a message) if a file is missing. Never trains."""
    prefix, tag, suffix = kw.get("prefix", ""), kw.get("tag", ""), kw.get("suffix", "")
    if not prefix:                                         # data-only: get_model would train on demand
        for k in range(N_EXP):
            if not (MODELS / f"{name}_N{n_obs}_s{sig}_k{k}{suffix}.pt").exists():
                print(f"   {label}: model file for k={k} missing, source skipped"); return None
    try:
        return [model_jets(get_model(name, n_obs, sig, k, **kw), x, t) for k in range(N_EXP)]
    except FileNotFoundError as e:
        print(f"   {label}: {e}, source skipped"); return None


def run_cell(name, n_obs, sig, cols_sub, cols_raw, g, k_true):
    print(f"\n=== {name}  N={n_obs}  sigma={sig}%")
    x, t, truth_J = load_truth(name, g)
    sources = {"truth": truth_J}
    oracle_J = try_load(name, n_obs, sig, x, t, "oracle", prefix="oracle_", tag="_lam1", suffix="_seed0")
    dataonly_J = try_load(name, n_obs, sig, x, t, "dataonly", suffix="_seed0")
    if oracle_J is not None:
        sources["oracle"] = oracle_J
        check_vs_metrics(name, n_obs, sig, oracle_J, truth_J)
    if dataonly_J is not None:
        sources["dataonly"] = dataonly_J

    rng = np.random.default_rng(0)
    it, ix = sample_indices(rng, len(t), len(x))           # same points for every source

    rows = []
    for src in ("oracle", "dataonly"):
        if src in sources:
            e = {key: np.mean([rel_error(sources[src][k][key], truth_J[k][key]) for k in range(N_EXP)])
                 for key in KEYS4}
            print(f"   jet errors {src:9s}: " + "  ".join(f"{key}={100 * v:.2f}%" for key, v in e.items()))
            rows.append(dict(name=name, N=n_obs, sig=sig, src=src, variant="err", gap=np.nan, count=-1,
                             scaling=np.nan, **{f"err_{key}": e[key] for key in KEYS4}))

    for src, J_list in sources.items():
        jd = pool(J_list, x, t, it, ix)
        for variant, cols, ev in (("sub", cols_sub, evaluate), ("raw", cols_raw, evaluate7)):
            s, gap, cnt, res = svd_stats(cols, jd, k_true, ev)
            rows.append(dict(name=name, N=n_obs, sig=sig, src=src, variant=variant, gap=gap, count=cnt,
                             scaling=res["scaling"]))

    for src in ("oracle", "dataonly"):                     # swap test: truth jets with ONE jet replaced
        if src not in sources:
            continue
        for key in KEYS4:
            mixed = []
            for k in range(N_EXP):
                m = dict(truth_J[k])                       # a copy, never edit truth_J itself
                m[key] = sources[src][k][key]
                mixed.append(m)
            jd = pool(mixed, x, t, it, ix)
            s, gap, cnt, res = svd_stats(cols_sub, jd, k_true, evaluate)
            rows.append(dict(name=name, N=n_obs, sig=sig, src=f"swap_{src}_{key}", variant="sub", gap=gap,
                             count=cnt, scaling=res["scaling"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="truth-only sanity checks, no models needed")
    ap.add_argument("--cell", nargs="+", default=["1000:1"], help="N:sigma pairs, e.g. 1000:1 200:3")
    ap.add_argument("--name", nargs="+", choices=list(CASES), default=list(CASES))
    args = ap.parse_args()

    if args.check:
        for name in args.name:
            check_truth(name)
        raise SystemExit

    all_rows = []
    for name in args.name:
        g, k_true = CASES[name]
        cols_sub, cols_raw = columns(g), column_raw(g)     # sympy is slow: once per case
        for cell in args.cell:
            n_obs, sig = (int(v) for v in cell.split(":"))
            all_rows += run_cell(name, n_obs, sig, cols_sub, cols_raw, g, k_true)

    print(f"\n{'case':6s} {'N':>5s} {'sig':>3s} {'source':22s} {'var':4s} {'gap':>6s} {'count':>5s} {'scaling':>9s}")
    for r in all_rows:
        if r["variant"] == "err":
            continue
        print(f"{r['name']:6s} {r['N']:5d} {r['sig']:3d} {r['src']:22s} {r['variant']:4s} "
              f"{r['gap']:6.2f} {r['count']:5d} {r['scaling']:9.1e}")

    fields = sorted({k for r in all_rows for k in r}, key=lambda s: (s not in ("name", "N", "sig", "src", "variant"), s))
    out = RESULTS / "stage0_floor.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print("\nsaved", out)
