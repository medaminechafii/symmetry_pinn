"""Summary of run_g0_steps.sh: g0 data, TRUE symmetries added one at a time (general symmetry loss, mu = 1), poly.

Per sigma (1%, 0%) and data draw k:
    E, E_clean   held-out error (t > 1.5) against the noisy / clean u
    R            E / reference, reference = the better of the 2 baseline seeds (the rule fixed in the design, 8.1)
    nu err       |nu_hat - 0.1| / 0.1
    err_g        RMS(g_hat) / RMS(u^3) at the data's u values (true g = 0)

    python -m experiments.g0_steps
"""
import numpy as np
from pinn.runs import load_rows
from experiments.delta_pilot import err_g

HO = 1.5
SETS = ["scaling", "scaling+Galilean", "scaling+Galilean+projective", "Galilean", "projective"]


def table(SIG):
    sel = lambda r: r["N"] == 1000 and r["sig"] == SIG
    base = {(r["k"], r["seed"]): r for r in load_rows("poly", names=["g0"], holdout=HO) if sel(r)}
    sym = {(r["k"], r["gen_file"].split("__")[-1].removesuffix(".npz")): r
           for r in load_rows("poly", mu=1.0, names=["g0"], holdout=HO) if sel(r) and r.get("gen_file")}
    print(f"\n===== g0, poly, sigma = {SIG}%: {len(base)} baselines, {len(sym)} symmetry runs\n")
    print(f"{'k':>2s} {'run':28s} | {'E':>9s} {'E_clean':>9s} | {'R':>8s} {'R_clean':>8s} | {'nu err':>7s} {'err_g':>7s} | c1..c5")
    for k in sorted({k for k, _ in base}):
        b = [base[(k, s)] for s in (0, 1) if (k, s) in base]
        ref = min(b, key=lambda r: r["err_hold"])
        rows = [(f"base s{r['seed']}", r) for r in b] + [(s, sym.get((k, s))) for s in SETS]
        for lab, r in rows:
            if r is None:
                print(f"{k:2d} {lab:28s} | missing"); continue
            R, Rc = r["err_hold"] / ref["err_hold"], r["err_hold_clean"] / ref["err_hold_clean"]
            print(f"{k:2d} {lab:28s} | {r['err_hold']:9.3e} {r['err_hold_clean']:9.3e} | {R:8.3f} {Rc:8.3f} | "
                  f"{100 * r['nu_err']:6.2f}% {err_g(r):7.4f} | " + " ".join(f"{c:6.3f}" for c in r["c"]))
        print()


if __name__ == "__main__":
    for SIG in (1, 0):
        table(SIG)
