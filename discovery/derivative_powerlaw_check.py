"""Two checks on a learned g_hat (poly or MLP), against the true g in {0, -u^3}.

Check 1  derivative error.  delta = g_hat - g.  The scaling breaking of the learned equation is exactly
         B = u g_hat' - 3 g_hat = u delta' - 3 delta   (the true g is homogeneous).  Is the value error small but
         the derivative error large, so that u delta' dominates B?   All sizes are divided by rms(u^3), the size of the
         cubic term, so g0 and cubic are on the same scale.
Check 2  power-law misfit.  Best c*u^p (p = 1..5) for g_hat, relative misfit ||g_hat - c u^p|| / ||g_hat||.
         Calibrated on the exact equations g = -u^3 + eps*u (misfit and SVD gap side by side).
         (Only meaningful for cubic: for g0 g_hat is about 0, so it is reported as a size.)

    python -m discovery.derivative_powerlaw_check --calib          # data-free calibration table + demo (works anywhere)
    python -m discovery.derivative_powerlaw_check --arm MLP        # the real runs (ubuntu)
    python -m discovery.derivative_powerlaw_check --arm poly
"""
import argparse
from pinn.runs import load_rows, true_g, stem as run_stem
import json
import numpy as np
from discovery.stage0b_general import (g_poly, g_mlp, load_mlp, points, sample_indices, gap_from_g)
from discovery.noise_gap import CASES
from pinn.surrogate import RESULTS

rms = lambda a: np.sqrt(np.mean(np.asarray(a) ** 2))


def error_stats(uu, g, g1, name):
    gt, g1t = true_g(name, uu)
    d, d1 = g - gt, g1 - g1t
    s0, s1 = rms(uu**3), rms(3 * uu**2)
    # B is the scaling breaking of the LEARNED equation, u g_hat' - 3 g_hat.  It equals u delta' - 3 delta only when
    # the true g is homogeneous (g0, cubic); for the eps data the true g itself breaks scaling.
    return dict(e_val=rms(d) / s0, e_der=rms(d1) / s1, u_dprime=rms(uu * d1) / s0,
                three_d=rms(3 * d) / s0, B=rms(uu * g1 - 3 * g) / s0)


def powerlaw(uu, g):
    best = None
    for p in range(1, 6):
        v = uu**p
        c = (g @ v) / (v @ v)
        mis = np.linalg.norm(g - c * v) / np.linalg.norm(g)
        if best is None or mis < best[1]:
            best = (p, mis, c)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--arm", choices=["MLP", "poly"], default="MLP")
    ap.add_argument("--mu", type=float, default=0.0, help="symmetry-loss weight of the runs to analyse (0 = stage 1)")
    args = ap.parse_args()
    it, ix = sample_indices()

    if args.calib:
        jd = points("cubic", 0, it, ix)
        uu = jd["u"]
        print(f"rms(u^3) = {rms(uu**3):.4f}   rms(u) = {rms(uu):.3f}\n")
        print("calibration: exact g = -u^3 + eps*u")
        print(f"{'eps':>7s} | {'best p':>6s} {'misfit':>8s} | {'gap@3':>6s}")
        for eps in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
            g, g1 = -uu**3 + eps * uu, -3 * uu**2 + eps
            p, mis, c = powerlaw(uu, g)
            print(f"{eps:7.3f} | {p:6d} {mis:8.3f} | {gap_from_g(jd, g, g1, 0.1, 3)[0]:6.2f}")
        c = (2.9e-4, 2.9e-3, -0.940, 9.0e-3, -0.0806)
        g, g1 = g_poly(c, uu)
        print("\ndemo: the learned vector of the earlier Mac run (c3 = -0.94, c5 = -0.081)")
        print("  ", {k: round(float(v), 4) for k, v in error_stats(uu, g, g1, "cubic").items()})
        p, mis, cc = powerlaw(uu, g)
        print(f"   best power law: p = {p}, misfit = {mis:.3f};   gap@3 = {gap_from_g(jd, g, g1, 0.1, 3)[0]:.2f}")
        raise SystemExit

    rows = load_rows(args.arm, args.mu)
    print(len(rows), args.arm, "rows, mu = " + str(args.mu) + "  (sizes in units of rms(u^3))")
    cache, res = {}, {}
    for r in rows:
        name, k = r["name"], r["k"]
        if (name, k) not in cache:
            cache[(name, k)] = points(name, k, it, ix)
        uu = cache[(name, k)]["u"]
        g, g1 = g_poly(r["c"], uu) if args.arm == "poly" else g_mlp(load_mlp(r), uu)
        st = error_stats(uu, g, g1, name)
        p, mis, _ = powerlaw(uu, g)
        st.update(p=p, mis=mis)
        res.setdefault((name, r["N"], r["sig"]), []).append(st)

    keys = ("e_val", "e_der", "u_dprime", "three_d", "B")
    print(f"{'case':10s} {'N':>5s} {'sig':>3s} | " + " ".join(f"{k:>9s}" for k in keys) + " | best p (counts)  misfit med")
    for (name, N, sig), v in sorted(res.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        med = [np.median([s[k] for s in v]) for k in keys]
        ps = np.bincount([s["p"] for s in v], minlength=6)[1:]
        mis = np.median([s["mis"] for s in v]) if name != "g0" else float("nan")
        print(f"{name:10s} {N:5d} {sig:3d} | " + " ".join(f"{m:9.3f}" for m in med) +
              f" | p=1..5: {ps.tolist()}  {mis:8.3f}")
