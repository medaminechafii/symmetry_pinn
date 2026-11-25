"""Alignment check: are the first non-trivial singular vectors of a LEARNED equation close to the true symmetries?

Needed before the iterative "propose candidate A from the SVD, enforce it, test it" loop: if candidate A (the singular
vector right after d/dx and d/dt) has nothing to do with scaling, enforcing it would be meaningless.

Same matrix as in section 8 / deflated_count_check: the 30 substituted columns, evaluated at the true jets with the
learned g_hat, g_hat' and nu_hat; columns normalised; SVD. Every generator is a vector of 30 coefficients (xi, tau, phi
on the monomials 1, x, t, u, x^2, t^2, u^2, xt, xu, tu); the singular vectors are vectors of the same kind.

    res(j) = how much of the true generator w is NOT a combination of the 2 + j smallest singular vectors
             (2 = the trivial d/dx, d/dt; j = 1 is "trivial + candidate A", j = 2 adds candidate B, ...).
             Least-squares projection (fingerprint.projection_residual), in normalised coordinates.
             res = sin(angle between w and that span): 0 = w lies in it exactly, 0.1 ~ 6 deg, 0.5 = 30 deg, 1 = unrelated.

True symmetries: cubic -> scaling;  g0 -> scaling, Galilean, projective.  For the eps data scaling is NOT a symmetry;
its row shows whether the SVD would still propose scaling there.

    python -m discovery.alignment_check --calib       # exact equations (data-free reference)
    python -m discovery.alignment_check --arm poly    # stage-1 learned runs (ubuntu)
    python -m discovery.alignment_check --arm MLP
"""
import argparse
import numpy as np
from pinn.runs import load_rows, true_g
from discovery.stage0b_general import FUNS, KEYS, points, sample_indices, g_poly, g_mlp, load_mlp
from discovery.fingerprint import normalise, projection_residual, KNOWN
from discovery.deflated_count_check import THR

GENS = {"cubic": ["scaling"], "g0": ["scaling", "Galilean", "projective"]}
J_MAX = 3


def gens_of(name):
    return GENS.get(name, ["scaling"])            # eps data: scaling is the (false) candidate of interest


def svd(jd, g, g1, nu):
    args = [jd[k] for k in KEYS] + [g, g1, nu]
    A = np.column_stack([np.broadcast_to(np.asarray(f(*args), dtype=float), jd["x"].shape) for f in FUNS])
    A_n, scale, _ = normalise(A)
    _, s, Vt = np.linalg.svd(A_n, full_matrices=False)
    return s, Vt, scale


def residuals(jd, g, g1, nu, name):
    """n_dropped (values < THR * largest) and {generator: [res(1), ..., res(J_MAX)]}."""
    s, Vt, scale = svd(jd, g, g1, nu)
    n_drop = int((s < THR * s[0]).sum())
    return n_drop, {gn: [projection_residual(KNOWN[gn], Vt, 2 + j, scale) for j in range(1, J_MAX + 1)]
                    for gn in gens_of(name)}


def fmt(vals):
    v = np.array(vals)
    return f"{np.median(v):5.2f} [{v.min():4.2f},{v.max():4.2f}]"


HEAD = (f"{'case':11s} {'N':>5s} {'sig':>3s} {'n':>2s} {'dropped':>8s} | {'generator':10s} | "
        + " | ".join(f"{'res(' + str(j) + ') med [min,max]':20s}" for j in range(1, J_MAX + 1)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    args = ap.parse_args()
    it, ix = sample_indices()
    cache = {}

    def jets(name, k):
        if (name, k) not in cache:
            cache[(name, k)] = points(name, k, it, ix)
        return cache[(name, k)]

    if args.calib:
        print("exact equations, true nu (k = 0, 1, 2).  Exact symmetries give values ~1e-16 that are all 'dropped',")
        print("so here res(j) uses the 2 + j smallest vectors in the same way as for the learned runs.")
        print(HEAD)
        for name in ("cubic", "g0", "cubic_e003", "cubic_e01"):
            out = [residuals(jets(name, k), *true_g(name, jets(name, k)["u"]), 0.1, name) for k in range(3)]
            for gn in gens_of(name):
                print(f"{name:11s} {'-':>5s} {'-':>3s} {3:2d} {str(sorted({o[0] for o in out})):>8s} | {gn:10s} | "
                      + " | ".join(fmt([o[1][gn][j] for o in out]) for j in range(J_MAX)))
        raise SystemExit

    rows = load_rows(args.arm)                    # stage 1: mu = 0, wd = 0
    print(len(rows), args.arm, "stage-1 rows")
    res = {}
    for r in rows:
        jd = jets(r["name"], r["k"])
        g, g1 = g_poly(r["c"], jd["u"]) if args.arm == "poly" else g_mlp(load_mlp(r), jd["u"])
        res.setdefault((r["name"], r["N"], r["sig"]), []).append(residuals(jd, g, g1, r["nu_hat"], r["name"]))
    print(HEAD)
    order = {"g0": 0, "cubic": 1}
    for (name, N, sig), v in sorted(res.items(), key=lambda kv: (order.get(kv[0][0], 2), kv[0][0], -kv[0][1], kv[0][2])):
        for gn in gens_of(name):
            print(f"{name:11s} {N:5d} {sig:3d} {len(v):2d} {str(sorted({o[0] for o in v})):>8s} | {gn:10s} | "
                  + " | ".join(fmt([o[1][gn][j] for o in v]) for j in range(J_MAX)))
