"""Stage 0b for ANY learned g (polynomial or MLP): the columns only need g(u) and g'(u) (no g'').

The 30 substituted columns are built once with symbols G, G1 (= g and g') and nu_hat, then evaluated with the
numeric values of the learned g and its derivative at the sample points (for the MLP: autograd).
For the cubic case it also gives the equivalent epsilon of the scaling breaking B = u g' - 3 g and the gap predicted
from it (same idea as eps_eff_check.py), so that MLP and polynomial runs are comparable.

    python -m discovery.stage0b_general --check                 # route equivalence + MLP smoke test (works anywhere)
    python -m discovery.stage0b_general --arm MLP               # the learned-g MLP runs of results/oracle/metrics.jsonl
    python -m discovery.stage0b_general --arm poly
"""
import argparse
from pinn.runs import load_rows, stem as run_stem
import json
import numpy as np, sympy as sp, torch
from discovery.criterion import columns, u, JET
from discovery.fingerprint import normalise, count_by_gap, N_POINTS
from discovery.noise_gap import CASES, gap_at
from discovery.stage0_floor import truth_jets
from pinn.surrogate import RESULTS, MODELS
from pinn.learned_g import MLPG

G_F = sp.Function("G")
G_S, G1_S, NU_S = sp.symbols("G G1 nu_hat")
_cols = columns(G_F(u), NU_S)
COLS = [c.subs(sp.Derivative(G_F(u), u), G1_S).subs(G_F(u), G_S) for c in _cols]
assert not any(c.has(G_F) for c in COLS), "unexpected dependence on g'' or other derivatives"
FUNS = [sp.lambdify(JET + (G_S, G1_S, NU_S), c, "numpy") for c in COLS]
KEYS = ("x", "t", "u", "u_x", "u_xx", "u_xxx")
KS = np.arange(1, 6)


def gap_from_g(jd, g_val, g1_val, nu, k):
    args = [jd[key] for key in KEYS] + [g_val, g1_val, nu]
    A = np.column_stack([np.broadcast_to(np.asarray(f(*args), dtype=float), jd["x"].shape) for f in FUNS])
    A_n, _, _ = normalise(A)
    s = np.linalg.svd(A_n, compute_uv=False)
    return gap_at(s, k), count_by_gap(s)[0]


def g_poly(c, uu):
    c = np.asarray(c, dtype=float)
    return (c[None, :] * uu[:, None] ** KS[None, :]).sum(1), (KS * c[None, :] * uu[:, None] ** (KS - 1)[None, :]).sum(1)


def g_mlp(model, uu):
    ut = torch.tensor(uu, requires_grad=True)
    g = model(ut)
    g1 = torch.autograd.grad(g.sum(), ut)[0]
    return g.detach().numpy(), g1.detach().numpy()


def eps_eff(uu, g, g1):
    B = uu * g1 - 3 * g
    return np.sqrt(np.mean(B**2)) / (2 * np.sqrt(np.mean(uu**2)))


def points(name, k, it, ix):
    x, t, J = truth_jets(name, k, CASES[name][0])
    jd = {key: J[key][it, ix] for key in ("u", "u_x", "u_xx", "u_xxx")}
    jd["x"], jd["t"] = x[ix], t[it]
    return jd


def sample_indices():
    x0, t0, _ = truth_jets("cubic", 0, CASES["cubic"][0])
    rng = np.random.default_rng(0)
    return rng.integers(0, len(t0), N_POINTS), rng.integers(0, len(x0), N_POINTS)


def load_mlp(r):
    m = MLPG()
    m.load_state_dict(torch.load(MODELS / (run_stem(r) + "_g.pt")))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arm", choices=["MLP", "poly"], default="MLP")
    ap.add_argument("--mu", type=float, default=0.0, help="symmetry-loss weight of the runs to analyse (0 = stage 1)")
    args = ap.parse_args()
    it, ix = sample_indices()

    if args.check:
        from discovery.eps_eff_check import gap3
        jd = points("cubic", 0, it, ix)
        rng = np.random.default_rng(1)
        print("route equivalence (g, g' numeric)  vs  (symbolic polynomial route of eps_eff_check):")
        worst = 0.0
        for _ in range(5):
            c = np.array([0, 0, -1, 0, 0]) + 0.05 * rng.normal(size=5)
            g, g1 = g_poly(c, jd["u"])
            a = gap_from_g(jd, g, g1, 0.1, 3)[0]
            b = gap3(jd, c, 0.1)
            worst = max(worst, abs(a - b))
            print(f"   gap@3 new = {a:.6f}   old = {b:.6f}")
        print(f"   worst difference = {worst:.1e}")
        torch.manual_seed(0)
        m = MLPG()                                    # random weights: only a smoke test of the code path
        g, g1 = g_mlp(m, jd["u"])
        print("MLP smoke test (random weights): shapes", g.shape, g1.shape,
              " gap@3 =", round(gap_from_g(jd, g, g1, 0.1, 3)[0], 2))
        raise SystemExit

    rows = load_rows(args.arm, args.mu)
    print(len(rows), args.arm, "rows, mu =", args.mu)
    cache, res = {}, {}
    for r in rows:
        name, k = r["name"], r["k"]
        k_true = CASES[name][1]
        if (name, k) not in cache:
            cache[(name, k)] = points(name, k, it, ix)
        jd = cache[(name, k)]
        g, g1 = g_poly(r["c"], jd["u"]) if args.arm == "poly" else g_mlp(load_mlp(r), jd["u"])
        gap, cnt = gap_from_g(jd, g, g1, r["nu_hat"], k_true)
        if name == "cubic":
            e = eps_eff(jd["u"], g, g1)
            pg, _ = gap_from_g(jd, -jd["u"]**3 + e * jd["u"], -3 * jd["u"]**2 + e, 0.1, 3)
        else:
            e, pg = np.nan, np.nan
        res.setdefault((name, r["N"], r["sig"]), []).append((gap, cnt, e, pg))

    print(f"{'case':6s} {'N':>5s} {'sig':>3s} {'n':>2s} | {'gap@k_true med [min,max]':26s} | {'eps_eff med':>11s} | {'predicted':>9s} | counts")
    for (name, N, sig), v in sorted(res.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        a = np.array(v)
        print(f"{name:6s} {N:5d} {sig:3d} {len(a):2d} | {np.median(a[:,0]):6.2f} [{a[:,0].min():6.2f},{a[:,0].max():6.2f}]    | "
              f"{np.median(a[:,2]):11.4f} | {np.median(a[:,3]):9.2f} | {sorted(a[:,1].astype(int).tolist())}")
