"""Delta pilot summary (run_delta_pilot.sh; design section 2.6). Cell: cubic, N 1000, held-out t > 1.5,
sigma 1% and noise-free (sigma 0; there E = E_clean).

Per data draw k:
    E          held-out error (mean squared difference to the NOISY held-out observations; what the loop can use)
    E_clean    the same against the clean u (analysis only: in real use the clean u is unknown)
    seed1/0    E of baseline seed 1 / baseline seed 0: how much E changes when NOTHING changes but the seed
    R          E with the symmetry loss / E of the seed-0 baseline. scaling is true for cubic, Galilean is false.

    python -m experiments.delta_pilot --arm poly      (c1..c5 are printed for poly only)
    python -m experiments.delta_pilot --arm MLP
"""
import argparse
import numpy as np, torch
from pinn.runs import load_rows, true_g

HO = 1.5
rms = lambda a: np.sqrt(np.mean(np.asarray(a) ** 2))


def err_g(r):
    """RMS of g_hat - g_true over RMS(u^3), at the clean u values of the observation points of this data draw
    (so only the u range the data actually cover counts; |u| < 0.75 for cubic)."""
    from pinn.surrogate import DATA
    s = np.load(DATA / f"sparse_{r['name']}_N{r['N']}_s{r['sig']}.npz")
    u = s["u_clean"][s["exp_id"] == r["k"]]
    if r["g_model"] == "poly":
        g = sum(ck * u ** (i + 1) for i, ck in enumerate(r["c"]))
    else:
        from discovery.stage0b_general import load_mlp
        with torch.no_grad():
            g = load_mlp(r)(torch.tensor(u)).numpy()
    return rms(g - true_g(r["name"], u)[0]) / rms(u ** 3)


def table(SIG, arm):
    sel = lambda r: r["N"] == 1000 and r["sig"] == SIG
    base = {(r["k"], r["seed"]): r for r in load_rows(arm, names=["cubic"], holdout=HO) if sel(r)}
    sym = {}
    for r in load_rows(arm, mu=1.0, names=["cubic"], holdout=HO):
        if sel(r) and r.get("gen_file"):
            sym[(r["k"], r["gen_file"].split("__")[-1].removesuffix(".npz"))] = r
    print(f"\n===== {arm}, sigma = {SIG}%: {len(base)} baselines, {len(sym)} symmetry runs\n")
    print(f"{'k':>2s} {'run':10s} | {'E':>9s} {'E_clean':>9s} | {'R':>6s} {'R_clean':>8s} | {'nu_hat':>7s} {'err_g':>6s} | c1..c5")
    for k in sorted({k for k, _ in base}):
        b0 = base.get((k, 0))
        if b0 is None:
            continue
        runs = [("base s0", b0), ("base s1", base.get((k, 1))), ("scaling", sym.get((k, "scaling"))),
                ("Galilean", sym.get((k, "Galilean")))]
        for lab, r in runs:
            if r is None:
                print(f"{k:2d} {lab:10s} | missing"); continue
            R = r["err_hold"] / b0["err_hold"]
            Rc = r["err_hold_clean"] / b0["err_hold_clean"]
            print(f"{k:2d} {lab:10s} | {r['err_hold']:9.3e} {r['err_hold_clean']:9.3e} | {R:6.3f} {Rc:8.3f} | "
                  f"{r['nu_hat']:7.4f} {err_g(r):6.3f} | " + " ".join(f"{c:6.3f}" for c in (r["c"] or [])))
        print()
    print("noise variance (floor of E):", ", ".join(f"k{k}: {base[(k, 0)]['noise_var']:.2e}" for k in sorted({k for k, _ in base})
                                                    if (k, 0) in base))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    arm = ap.parse_args().arm
    for SIG in (1, 0):
        table(SIG, arm)
