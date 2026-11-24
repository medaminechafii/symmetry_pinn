"""Per-run learned coefficients for the symmetry-loss pilot (poly arm, N=200, sigma=3%).

Question: when the cubic case improves only a little, is it because the symmetry loss failed to remove
c1, c2, c4, c5, or because the remaining error sits in c3 (which the scaling symmetry leaves free)?

    python -m experiments.sym_coeffs
"""
import numpy as np
from pinn.runs import load_rows

N, SIG = 200, 3
print(f"{'case':5s} {'mu':>5s} {'seed':>4s} {'k':>2s} |" + "".join(f"{'c'+str(i):>8s}" for i in range(1, 6))
      + f" | {'nu_hat':>7s} | {'data/noise':>10s} | {'mu*sym':>9s}")
for name in ("cubic", "g0"):
    for mu in (0.0, 0.01, 0.1, 1.0):
        rows = [r for r in load_rows(arm="poly", mu=mu, names=[name]) if r["N"] == N and r["sig"] == SIG]
        rows.sort(key=lambda r: (r["k"], r["seed"]))
        C = []
        for r in rows:
            c = np.array(r["c"]); C.append(c)
            ms = mu * r["sym_final"] if r.get("sym_final") is not None else float("nan")
            print(f"{name:5s} {mu:5g} {r['seed']:4d} {r['k']:2d} |" + "".join(f"{x:8.3f}" for x in c)
                  + f" | {r['nu_hat']:7.4f} | {r['data_over_noise']:10.2f} | {ms:9.2e}")
        if C:
            C = np.array(C)
            print(f"{name:5s} {mu:5g}  med    |" + "".join(f"{x:8.3f}" for x in np.median(C, 0)) + "\n")
