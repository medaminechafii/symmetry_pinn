import json
import numpy as np, torch
from collections import defaultdict
from pinn.surrogate import dense_truth, MODELS
from pinn.learned_g import MLPG          # if this import fails, use pinn.poly_g

KS = np.arange(1, 6)

import argparse
from pinn.runs import load_rows, true_g, stem as run_stem
ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.0, help="symmetry-loss weight of the runs to analyse (0 = stage 1)")
ap.add_argument("--wd", type=float, default=0.0, help="weight-decay baseline: g_wd of the runs to analyse (0 = off)")
args = ap.parse_args()

# one row per run of this mu; later duplicates overwrite earlier ones
rows = {}
for r in load_rows(mu=args.mu, wd=args.wd):
    rows[(r["g_model"], r["name"], r["N"], r["sig"], r["k"], r["seed"])] = r
print(len(rows), "runs found, mu =", args.mu, " wd =", args.wd)

truth_cache = {}
def true_u(name, k):
    if (name, k) not in truth_cache:
        truth_cache[(name, k)] = dense_truth(name, k)[2]["u"].ravel()
    return truth_cache[(name, k)]

def g_hat_of(r, u):
    if r["g_model"] == "poly":
        c = np.array(r["c"])
        return (c[None, :] * u[:, None] ** KS[None, :]).sum(1)
    g = MLPG()
    g.load_state_dict(torch.load(MODELS / (run_stem(r) + "_g.pt")))
    with torch.no_grad():
        return g(torch.tensor(u)).numpy()

res = defaultdict(list)
per_run = []
for key, r in rows.items():
    arm, name, N, sig, k, seed = key
    u = true_u(name, k)
    g_true = true_g(name, u)[0]
    err = np.sqrt(np.mean((g_hat_of(r, u) - g_true)**2)) / np.sqrt(np.mean((u**3)**2))
    res[(name, N, sig, arm)].append((err, r["nu_err"], r["D"]))
    per_run.append((arm, name, N, sig, k, seed, err, r["nu_err"], r["D"]))

print(f"{'case':6s} {'N':>5s} {'sig':>3s} {'arm':5s} {'n':>2s} | "
      f"{'err_g med [min-max]':26s} | {'nu_err med [min-max]':26s} | {'D med [min-max]':26s}")
for (name, N, sig, arm) in sorted(res, key=lambda t: (t[0], -t[1], t[2], t[3])):
    a = np.array(res[(name, N, sig, arm)])          # columns: err_g, nu_err, D
    cell = lambda i: f"{np.median(a[:, i]):.4f} [{a[:, i].min():.4f}-{a[:, i].max():.4f}]"
    print(f"{name:6s} {N:5d} {sig:3d} {arm:5s} {len(a):2d} | {cell(0):26s} | {cell(1):26s} | {cell(2):26s}")

np.savetxt("results/learned_g_per_run" + (f"_mu{args.mu:g}" if args.mu else "") + (f"_wd{args.wd:g}" if args.wd else "") + ".csv", np.array(per_run, dtype=object), fmt="%s", delimiter=",",
           header="arm,name,N,sig,k,seed,err_g,nu_err,D")
NU = 0.1
comp = defaultdict(list)
n = np.linalg.norm
for key, r in rows.items():
    arm, name, N, sig, k, seed = key
    _, _, tr = dense_truth(name, k)
    u, u_xx = tr["u"].ravel(), tr["u_xx"].ravel()
    g_true = true_g(name, u)[0]
    dg  = g_hat_of(r, u) - g_true            # error of the learned g, point by point
    dnu = (r["nu_hat"] - NU) * u_xx          # error of nu, as it enters the equation
    cos      = dg @ dnu / (n(dg) * n(dnu))   # -1 = the two errors cancel, 0 = unrelated, +1 = add up
    ratio    = n(dnu) / n(dg)                # size of the nu error vs the g error
    combined = n(dg + dnu) / np.sqrt(n(dg)**2 + n(dnu)**2)   # <1 = they cancel, 1 = independent
    comp[(name, N, sig, arm)].append((cos, ratio, combined))

print(f"\n{'case':6s} {'N':>5s} {'sig':>3s} {'arm':5s} | {'cos med [min,max]':24s} | {'nu/g size med':14s} | combined med")
for (name, N, sig, arm) in sorted(comp, key=lambda t: (t[0], -t[1], t[2], t[3])):
    a = np.array(comp[(name, N, sig, arm)])
    print(f"{name:6s} {N:5d} {sig:3d} {arm:5s} | "
          f"{np.median(a[:,0]):+.2f} [{a[:,0].min():+.2f},{a[:,0].max():+.2f}]   | "
          f"{np.median(a[:,1]):.3f}          | {np.median(a[:,2]):.2f}")