"""Learned equations: substituted columns vs RAW columns (u_t and u_xt taken from the network).

For every learned-g run (poly or MLP) the trained PINN gives the jets at the sample points.  Two matrices are built
on the SAME points with the SAME learned g_hat, g_hat' and nu_hat:
    sub : u_t and u_xt replaced by the learned equation  (needs u, u_x, u_xx, u_xxx from the network)
    raw : u_t and u_xt taken from the network            (needs u, u_x, u_t, u_xx, u_xt from the network)
The raw criterion off the solutions is   pr V(Delta) = (substituted part) + (multiple of the residual), so the raw gap
also sees how well the network satisfies its own learned equation.  That residual is printed too:
    res = rms(u_t + u u_x - nu_hat u_xx - g_hat(u)) / rms(u_t)   on the network jets.

    python -m discovery.raw_vs_sub_learned --check            # truth jets, exact g: sub and raw must agree (works anywhere)
    python -m discovery.raw_vs_sub_learned --arm poly         # learned runs (ubuntu: needs the PINN .pt files)
    python -m discovery.raw_vs_sub_learned --arm MLP
"""
import argparse
from pinn.runs import load_rows, stem as run_stem
import json
import numpy as np, sympy as sp, torch
from discovery.criterion import criterion, basis_fields, u
from discovery.noise_gap import JET7, JET7_KEY, CASES, gap_at
from discovery.fingerprint import normalise
from discovery.stage0b_general import (FUNS as FUNS_SUB, KEYS as KEYS_SUB, G_F, G_S, G1_S, NU_S,
                                       g_poly, g_mlp, load_mlp, sample_indices)
from discovery.deflated_count_check import deflated
from discovery.stage0_floor import truth_jets
from pinn.surrogate import Surrogate, jet, RESULTS, MODELS

_raw = [criterion(*f, G_F(u), NU_S) for f in basis_fields()[0]]
RAW = [c.subs(sp.Derivative(G_F(u), u), G1_S).subs(G_F(u), G_S) for c in _raw]
assert not any(c.has(G_F) for c in RAW), "unexpected dependence on other derivatives of g"
FUNS_RAW = [sp.lambdify(tuple(JET7) + (G_S, G1_S, NU_S), c, "numpy") for c in RAW]


def spectrum(funs, keys, jd, g, g1, nu):
    args = [jd[k] for k in keys] + [g, g1, nu]
    A = np.column_stack([np.broadcast_to(np.asarray(f(*args), dtype=float), jd["x"].shape) for f in funs])
    return np.linalg.svd(normalise(A)[0], compute_uv=False)


def net_jets(model, xs, ts):
    xb = torch.tensor(xs, requires_grad=True)
    tb = torch.tensor(ts, requires_grad=True)
    u_, ux, ut, uxx, uxt = jet(model, xb, tb)
    uxxx = torch.autograd.grad(uxx, xb, grad_outputs=torch.ones_like(uxx))[0]
    vals = dict(u=u_, u_x=ux, u_t=ut, u_xx=uxx, u_xt=uxt, u_xxx=uxxx)
    out = {k: v.detach().numpy() for k, v in vals.items()}
    out["x"], out["t"] = xs, ts
    return out


def load_net(r):
    m = Surrogate(K=r.get("feat", 1))
    m.load_state_dict(torch.load(MODELS / (run_stem(r) + ".pt")))
    return m


def both(jd, g, g1, nu, k_true):
    s_sub = spectrum(FUNS_SUB, KEYS_SUB, jd, g, g1, nu)
    s_raw = spectrum(FUNS_RAW, JET7_KEY, jd, g, g1, nu)
    return (gap_at(s_sub, k_true), deflated(s_sub)[4][0.5], gap_at(s_raw, k_true), deflated(s_raw)[4][0.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    ap.add_argument("--mu", type=float, default=0.0, help="symmetry-loss weight of the runs to analyse (0 = stage 1)")
    args = ap.parse_args()
    it, ix = sample_indices()

    if args.check:
        for name in ("cubic", "g0"):
            g_true, k_true = CASES[name]
            x, t, J = truth_jets(name, 0, g_true)
            jd = {key: J[key][it, ix] for key in ("u", "u_x", "u_t", "u_xx", "u_xt", "u_xxx")}
            jd["x"], jd["t"] = x[ix], t[it]
            uu = jd["u"]
            g, g1 = (-uu**3, -3 * uu**2) if name == "cubic" else (0 * uu, 0 * uu)
            gs, cs, gr, cr = both(jd, g, g1, 0.1, k_true)
            print(f"[check] {name}, truth jets, exact g:  sub gap = {gs:.2f}, count {cs}   raw gap = {gr:.2f}, count {cr}"
                  f"   (expected about 14 and {k_true} for both)")
        raise SystemExit

    rows = load_rows(args.arm, args.mu)
    print(len(rows), args.arm, "rows, mu =", args.mu)
    grid, res = {}, {}
    for r in rows:
        name, k = r["name"], r["k"]
        k_true = CASES[name][1]
        if name not in grid:
            x, t, _ = truth_jets(name, 0, CASES[name][0])
            grid[name] = (x[ix], t[it])
        xs, ts = grid[name]
        jd = net_jets(load_net(r), xs, ts)
        uu = jd["u"]
        g, g1 = g_poly(r["c"], uu) if args.arm == "poly" else g_mlp(load_mlp(r), uu)
        resid = np.sqrt(np.mean((jd["u_t"] + uu * jd["u_x"] - r["nu_hat"] * jd["u_xx"] - g) ** 2)) / np.sqrt(np.mean(jd["u_t"] ** 2))
        res.setdefault((name, r["N"], r["sig"]), []).append((resid,) + both(jd, g, g1, r["nu_hat"], k_true))

    print(f"{'case':6s} {'N':>5s} {'sig':>3s} {'n':>2s} | {'residual':>8s} | {'sub gap':>7s} {'sub: count=true':>15s} | "
          f"{'raw gap':>7s} {'raw: count=true':>15s}   (gaps are medians, counts with the deflated rule, J_min = 0.5)")
    for (name, N, sig), v in sorted(res.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        a = np.array(v, dtype=float)
        kt = CASES[name][1]
        print(f"{name:6s} {N:5d} {sig:3d} {len(a):2d} | {np.median(a[:,0]):8.4f} | {np.median(a[:,1]):7.2f} "
              f"{int((a[:,2]==kt).sum()):>9d} / {len(a):<3d} | {np.median(a[:,3]):7.2f} {int((a[:,4]==kt).sum()):>9d} / {len(a):<3d}")
