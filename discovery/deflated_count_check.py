"""Count symmetries after ignoring the built-in (numerically zero) singular values.

For a learned equation:  1. build the 30 substituted columns and take the singular values.
                         2. drop the values below 1e-9 * (largest): the built-in / exact ones (d/dx, d/dt, ...).
                         3. among the next 8 values (smallest first) look at the log10 ratios between neighbours;
                            the largest one, J1, sits after the first j* remaining values;  J2 is the second largest.
                         4. deflated count = (number dropped) + j*  if J1 >= J_MIN, else (number dropped).
J_MIN = 0.5 and 1.0 are shown (the ordinary spacing of the spectrum is about 0.1-0.3 decades).
The same rule is applied to the exact equations -u^3 + eps*u (data-free calibration) so the numbers can be compared.

    python -m discovery.deflated_count_check --calib          # data-free (works anywhere)
    python -m discovery.deflated_count_check --arm poly       # learned runs (ubuntu)
    python -m discovery.deflated_count_check --arm MLP
"""
import argparse
import json
import numpy as np
from discovery.stage0b_general import (FUNS, KEYS, points, sample_indices, g_poly, g_mlp, load_mlp)
from discovery.fingerprint import normalise
from discovery.noise_gap import CASES
from pinn.surrogate import RESULTS

J_MINS = (0.5, 1.0)
THR = 1e-9
N_NEXT = 8


def spectrum(jd, g, g1, nu):
    args = [jd[k] for k in KEYS] + [g, g1, nu]
    A = np.column_stack([np.broadcast_to(np.asarray(f(*args), dtype=float), jd["x"].shape) for f in FUNS])
    A_n, _, _ = normalise(A)
    return np.linalg.svd(A_n, compute_uv=False)


def deflated(s):
    """Returns (n_dropped, j_star, J1, J2, {J_MIN: count})."""
    keep = s >= THR * s[0]
    n_drop = int((~keep).sum())
    rem = np.sort(s[keep])[:N_NEXT]                         # smallest first
    ratios = np.log10(rem[1:] / rem[:-1])                   # ratios[j-1] = jump after the first j values
    order = np.argsort(ratios)[::-1]
    j_star, J1, J2 = int(order[0]) + 1, float(ratios[order[0]]), float(ratios[order[1]])
    counts = {jm: n_drop + (j_star if J1 >= jm else 0) for jm in J_MINS}
    return n_drop, j_star, J1, J2, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--arm", choices=["MLP", "poly"], default="poly")
    args = ap.parse_args()
    it, ix = sample_indices()

    if args.calib:
        jd = points("cubic", 0, it, ix)
        uu = jd["u"]
        print("data-free calibration, exact equations g = -u^3 + eps*u  (true count: 3 at eps = 0, else 2)")
        print(f"{'eps':>7s} | {'dropped':>7s} {'j*':>3s} {'J1':>6s} {'J2':>6s} | count @0.5  count @1.0")
        for eps in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
            s = spectrum(jd, -uu**3 + eps * uu, -3 * uu**2 + eps, 0.1)
            nd, j, J1, J2, c = deflated(s)
            print(f"{eps:7.3f} | {nd:7d} {j:3d} {J1:6.2f} {J2:6.2f} | {c[0.5]:10d}  {c[1.0]:10d}")
        raise SystemExit

    rows = [json.loads(l) for l in open(RESULTS / "oracle" / "metrics.jsonl")]
    rows = [r for r in rows if r.get("g_model") == args.arm and r.get("learn_nu")]
    print(len(rows), args.arm, "rows")
    cache, res = {}, {}
    for r in rows:
        name, k = r["name"], r["k"]
        if (name, k) not in cache:
            cache[(name, k)] = points(name, k, it, ix)
        jd = cache[(name, k)]
        g, g1 = g_poly(r["c"], jd["u"]) if args.arm == "poly" else g_mlp(load_mlp(r), jd["u"])
        res.setdefault((name, r["N"], r["sig"]), []).append(deflated(spectrum(jd, g, g1, r["nu_hat"])))

    print(f"{'case':6s} {'N':>5s} {'sig':>3s} {'n':>2s} | {'dropped':>7s} | {'j* (median)':>11s} | {'J1 med':>6s} {'J2 med':>6s} | counts @0.5            | counts @1.0   (true: cubic 3, g0 5)")
    for (name, N, sig), v in sorted(res.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        nd = sorted({a[0] for a in v})
        c05 = sorted(a[4][0.5] for a in v)
        c10 = sorted(a[4][1.0] for a in v)
        print(f"{name:6s} {N:5d} {sig:3d} {len(v):2d} | {str(nd):>7s} | {np.median([a[1] for a in v]):11.0f} | "
              f"{np.median([a[2] for a in v]):6.2f} {np.median([a[3] for a in v]):6.2f} | {str(c05):22s} | {c10}")
