"""Does the size of the scaling breaking of a learned g_hat explain the gap of its SVD?  (cubic case)

For a learned polynomial c = (c1..c5) the scaling breaking is  B(u) = u g' - 3 g = sum_k (k-3) c_k u^k.
The pure-eps equation g = -u^3 + eps*u has B = -2 eps u.  Matching the RMS of B gives an equivalent eps:
    eps_eff = rms(B) / (2 rms(u)).
Prediction: gap@3 of the learned equation  ~  gap@3 of  c = (eps_eff, 0, -1, 0, 0).

    python -m discovery.eps_eff_check               # synthetic coefficient vectors (works anywhere)
    python -m discovery.eps_eff_check --real        # the poly rows of results/oracle/metrics.jsonl (ubuntu)
(Own copy of the matrix builder: discovery/stage0b_learned.py runs its table at import.)
"""
import argparse
import json
import numpy as np, sympy as sp
from discovery.criterion import columns, u, JET
from discovery.fingerprint import normalise, N_POINTS
from discovery.noise_gap import CASES, gap_at
from discovery.stage0_floor import truth_jets
from pinn.surrogate import RESULTS

C = sp.symbols("c1:6")
NU_S = sp.Symbol("nu_hat")
FUNS = [sp.lambdify(JET + C + (NU_S,), c, "numpy") for c in columns(sum(C[k-1] * u**k for k in range(1, 6)), NU_S)]
KS = np.arange(1, 6)
KEYS = ("x", "t", "u", "u_x", "u_xx", "u_xxx")


def gap3(jd, c, nu=0.1):
    args = [jd[k] for k in KEYS] + list(c) + [nu]
    A = np.column_stack([np.broadcast_to(np.asarray(f(*args), dtype=float), jd["x"].shape) for f in FUNS])
    A_n, _, _ = normalise(A)
    return gap_at(np.linalg.svd(A_n, compute_uv=False), 3)


def eps_eff(uu, c):
    B = ((KS - 3) * np.asarray(c)[None, :] * uu[:, None] ** KS[None, :]).sum(1)
    return np.sqrt(np.mean(B**2)) / (2 * np.sqrt(np.mean(uu**2)))


def points(name, k, it, ix):
    x, t, J = truth_jets(name, k, CASES[name][0])
    jd = {key: J[key][it, ix] for key in ("u", "u_x", "u_xx", "u_xxx")}
    jd["x"], jd["t"] = x[ix], t[it]
    return jd


def report(jd, c, label=""):
    e = eps_eff(jd["u"], c)
    actual = gap3(jd, c)
    pred = gap3(jd, (e, 0, -1, 0, 0))
    return e, actual, pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    x0, t0, _ = truth_jets("cubic", 0, CASES["cubic"][0])
    rng = np.random.default_rng(0)
    it, ix = rng.integers(0, len(t0), N_POINTS), rng.integers(0, len(x0), N_POINTS)

    if not args.real:
        jd = points("cubic", 0, it, ix)
        print("a learned example seen earlier (Mac run, c3 = -0.94, c5 = -0.081):")
        e, a, p = report(jd, (2.9e-4, 2.9e-3, -0.940, 9.0e-3, -0.0806))
        print(f"   eps_eff = {e:.4f}   actual gap@3 = {a:.2f}   predicted from eps_eff = {p:.2f}\n")

        print("random perturbations of the pure cubic (each of c1, c2, c4, c5 and a shift of c3):")
        print(f"{'eps_eff':>9s} {'actual':>8s} {'predicted':>10s} {'diff':>7s}")
        rows = []
        for _ in range(30):
            scale = 10 ** rng.uniform(-3, -1)
            c = np.array([0, 0, -1, 0, 0], dtype=float) + scale * rng.normal(size=5) * np.array([1, 1, 0.3, 1, 1])
            rows.append(report(jd, c))
        rows.sort()
        for e, a, p in rows[::3]:
            print(f"{e:9.4f} {a:8.2f} {p:10.2f} {a - p:+7.2f}")
        d = np.array([a - p for _, a, p in rows])
        print(f"\nall 30: median |actual - predicted| = {np.median(np.abs(d)):.2f}, max = {np.max(np.abs(d)):.2f}")
    else:
        rows = [json.loads(l) for l in open(RESULTS / "oracle" / "metrics.jsonl")]
        rows = [r for r in rows if r.get("g_model") == "poly" and r.get("learn_nu") and r["name"] == "cubic"]
        print(len(rows), "cubic poly rows")
        cache, res = {}, {}
        for r in rows:
            if r["k"] not in cache:
                cache[r["k"]] = points("cubic", r["k"], it, ix)
            e, a, p = report(cache[r["k"]], r["c"])
            res.setdefault((r["N"], r["sig"]), []).append((e, a, p))
        print(f"{'N':>5s} {'sig':>3s} {'n':>2s} | {'eps_eff med':>11s} | {'actual gap med':>14s} | {'predicted med':>13s} | {'median |diff|':>13s}")
        for (N, sig), v in sorted(res.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            a = np.array(v)
            print(f"{N:5d} {sig:3d} {len(a):2d} | {np.median(a[:, 0]):11.4f} | {np.median(a[:, 1]):14.2f} | "
                  f"{np.median(a[:, 2]):13.2f} | {np.median(np.abs(a[:, 1] - a[:, 2])):13.2f}")
