"""Per-run learned coefficients for the symmetry-loss pilot (poly arm, N=200, sigma=3%).

Question: when the cubic case improves only a little, is it because the symmetry loss failed to remove
c1, c2, c4, c5, or because the remaining error sits in c3 (which the scaling symmetry leaves free)?

    python -m experiments.sym_coeffs                      # mu = 0, 0.01, 0.1, 1
    python -m experiments.sym_coeffs --wd 1e-6 1e-5 1e-4  # weight-decay baseline: stage 1, then each wd
"""
import argparse
import numpy as np
from pinn.runs import load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--mu", nargs="+", type=float, default=[0.0, 0.01, 0.1, 1.0])
ap.add_argument("--wd", nargs="+", type=float, default=None, help="show the weight-decay runs instead of the mu runs")
args = ap.parse_args()
settings = [(0.0, 0.0)] + [(0.0, w) for w in args.wd] if args.wd else [(m, 0.0) for m in args.mu]

N, SIG = 200, 3
print(f"{'case':5s} {'mu/wd':>6s} {'seed':>4s} {'k':>2s} |" + "".join(f"{'c'+str(i):>8s}" for i in range(1, 6))
      + f" | {'nu_hat':>7s} | {'data/noise':>10s} | {'mu*sym':>9s} | {'wd*sum c2':>9s}")
for name in ("cubic", "g0"):
    for mu, wd in settings:
        lab = f"{wd:g}w" if wd else f"{mu:g}"
        rows = [r for r in load_rows(arm="poly", mu=mu, wd=wd, names=[name]) if r["N"] == N and r["sig"] == SIG]
        rows.sort(key=lambda r: (r["k"], r["seed"]))
        C = []
        for r in rows:
            c = np.array(r["c"]); C.append(c)
            ms = mu * r["sym_final"] if r.get("sym_final") is not None else float("nan")
            print(f"{name:5s} {lab:>6s} {r['seed']:4d} {r['k']:2d} |" + "".join(f"{x:8.3f}" for x in c)
                  + f" | {r['nu_hat']:7.4f} | {r['data_over_noise']:10.2f} | {ms:9.2e} | {wd * (c**2).sum():9.2e}")
        if C:
            C = np.array(C)
            print(f"{name:5s} {lab:>6s}  med    |" + "".join(f"{x:8.3f}" for x in np.median(C, 0)) + "\n")
