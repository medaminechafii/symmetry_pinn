"""False-positive control: can the detector tell a learned equation from exactly symmetric data (cubic, g = -u^3)
apart from one learned on near-symmetric data (cubic_e003, cubic_e01: g = -u^3 + eps u, scaling broken)?

The eps data use the same observation points and the same noise draws as the cubic data (experiments/make_near_sym.py),
so each eps run has a PAIRED cubic run with the same arm, N, sigma, k and seed; only the eps term differs.

Per arm, N, sigma and data set (stage-1 runs: mu = 0, wd = 0):
    gap3     log10(s_4 / s_3) of the 30 substituted columns of the learned equation (large = scaling looks exact)
    c3@0.5, c3@1.0   fraction of runs with deflated count 3 (scaling accepted) at J_min = 0.5 / 1.0
    p3       fraction with best power law p = 3, and the median misfit (experiments 8.3)
    eps_eff  RMS(u g' - 3 g) / (2 RMS(u)) of the learned g; for the exact eps equation it equals eps
    err_g    function error of g_hat against its own true g, in units of RMS(u^3)
    win      paired: fraction of pairs where the cubic run has the LARGER gap3 than its eps partner (0.5 = no separation)
The first block is the same quantities for the exact equations (data-free reference).

    python -m experiments.eps_control --arm poly
    python -m experiments.eps_control --arm MLP
"""
import argparse
import numpy as np
from pinn.runs import load_rows, true_g
from discovery.stage0b_general import points, sample_indices, g_poly, g_mlp, load_mlp, gap_from_g, eps_eff
from discovery.deflated_count_check import spectrum, deflated
from discovery.derivative_powerlaw_check import powerlaw
from discovery.noise_gap import EPS

NAMES = ["cubic", "cubic_e003", "cubic_e01"]
EPS_OF = {"cubic": 0.0, "cubic_e003": EPS["e003"], "cubic_e01": EPS["e01"]}
rms = lambda a: np.sqrt(np.mean(np.asarray(a) ** 2))


def measures(jd, g, g1, nu, name):
    uu = jd["u"]
    s = spectrum(jd, g, g1, nu)
    c = deflated(s)[4]
    p, mis, _ = powerlaw(uu, g)
    return dict(gap3=gap_from_g(jd, g, g1, nu, 3)[0], c05=c[0.5] == 3, c10=c[1.0] == 3, p3=p == 3, mis=mis,
                eps_eff=eps_eff(uu, g, g1), err_g=rms(g - true_g(name, uu)[0]) / rms(uu**3))


def line(label, v):
    a = lambda k: np.array([m[k] for m in v], dtype=float)
    return (f"{label:22s} {len(v):3d} | {np.median(a('gap3')):5.2f} [{a('gap3').min():5.2f},{a('gap3').max():5.2f}] | "
            f"{a('c05').mean():5.2f} {a('c10').mean():5.2f} | {a('p3').mean():4.2f} {np.median(a('mis')):6.3f} | "
            f"{np.median(a('eps_eff')):7.3f} | {np.median(a('err_g')):6.3f}")


HEAD = (f"{'':22s} {'n':>3s} | {'gap3 med [min,max]':19s} | c3@0.5 c3@1.0 | {'p3':>4s} {'misfit':>6s} | "
        f"{'eps_eff':>7s} | {'err_g':>6s}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    args = ap.parse_args()
    it, ix = sample_indices()
    cache = {}

    def jets(name, k):
        if (name, k) not in cache:
            cache[(name, k)] = points(name, k, it, ix)
        return cache[(name, k)]

    print("exact equations (data-free, k = 0, 1, 2; true nu)")
    print(HEAD)
    for name in NAMES:
        v = []
        for k in range(3):
            jd = jets(name, k)
            g, g1 = true_g(name, jd["u"])
            v.append(measures(jd, g, g1, 0.1, name))
        print(line(f"{name} (eps {EPS_OF[name]:g})", v))

    rows = [r for r in load_rows(args.arm, names=NAMES)]
    res = {}
    for r in rows:
        jd = jets(r["name"], r["k"])
        g, g1 = g_poly(r["c"], jd["u"]) if args.arm == "poly" else g_mlp(load_mlp(r), jd["u"])
        res[(r["N"], r["sig"], r["name"], r["k"], r["seed"])] = measures(jd, g, g1, r["nu_hat"], r["name"])
    cells = sorted({(N, sig) for (N, sig, name, _, _) in res if name != "cubic"}, key=lambda c: (-c[0], c[1]))
    print(f"\nlearned equations, arm {args.arm}, stage 1 ({len(rows)} rows)")
    for N, sig in cells:
        print(f"\nN = {N}, sigma = {sig}%\n" + HEAD + " | win vs cubic")
        for name in NAMES:
            keys = sorted(key for key in res if key[:3] == (N, sig, name))
            v = [res[key] for key in keys]
            if not v:
                continue
            txt = line(f"{name} (eps {EPS_OF[name]:g})", v)
            if name != "cubic":
                pairs = [(res[(N, sig, "cubic") + key[3:]]["gap3"], res[key]["gap3"])
                         for key in keys if (N, sig, "cubic") + key[3:] in res]
                win = np.mean([a > b for a, b in pairs]) if pairs else float("nan")
                txt += f" | {win:4.2f} ({len(pairs)} pairs)"
            print(txt)
