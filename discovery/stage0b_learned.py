import json,numpy as np, sympy as sp
from discovery.criterion import columns,u,JET
from discovery.fingerprint import normalise,count_by_gap,KNOWN,projection_residual,N_POINTS
from discovery.noise_gap import CASES,gap_at 
from discovery.stage0_floor import truth_jets
from pinn.surrogate import RESULTS

C = sp.symbols("c1:6")
NU_S = sp.Symbol("nu_hat")
G_SYM = sum(C[k-1] * u**k for k in range(1,6))
COLS = columns(G_SYM,NU_S)
FUNS = [sp.lambdify(JET + C + (NU_S,), c, "numpy") for c in COLS] #numeric evaluators

def build_matrix_learned(jd, c, nu_hat):
    """jd: dict with x, t, u, u_x, u_xx, u_xxx (1-D arrays). c: 5 coefficients."""
    args = [jd[k] for k in ("x","t","u","u_x","u_xx","u_xxx")] + list(c) + [nu_hat]
    cols = [np.broadcast_to(np.asarray(f(*args), dtype=float),jd["x"].shape) for f in FUNS]
    return np.column_stack(cols)

rows = []
for line in open(RESULTS / "oracle" / "metrics.jsonl"):
    r = json.loads(line)
    if r.get("g_model") == "poly" and r.get("learn_nu") and not r.get("mu") and not r.get("g_wd"):   # stage 1 only
        rows.append(r)
print(len(rows),"poly rows")

# jets cache 
cache = {}
x0, t0, _ = truth_jets("g0",0,CASES["g0"][0])
rng = np.random.default_rng(0)
it = rng.integers(0,len(t0),N_POINTS)
ix = rng.integers(0,len(x0),N_POINTS)

# one svd per run
res = {}
for r in rows:
    name, k = r["name"], r["k"]
    g_true, k_true = CASES[name]
    if (name, k) not in cache:
        cache[(name, k)] = truth_jets(name,k,g_true)
    x, t, J = cache[(name, k)]
    jd = {key: J[key][it,ix] for key in ("u","u_x","u_xx","u_xxx")}
    jd["x"], jd["t"] = x[ix],t[it]

    A = build_matrix_learned(jd, r["c"],r["nu_hat"])
    A_n, scale, _ = normalise(A)
    _, s, Vt = np.linalg.svd(A_n,full_matrices=False)
    res.setdefault((name,r["N"], r["sig"]),[]).append(
        (gap_at(s,k_true),gap_at(s,2),count_by_gap(s)[0])
    )

fmt = lambda a: f"{np.median(a):6.2f} [{a.min():6.2f},{a.max():6.2f}]"
print(f"{'case':6s} {'N':>5s} {'sig':>3s} {'n':>2s} {'gap @ k_true':22s} | {'gap @ 2':22s} | counts")
for (name, N, sig), v in sorted(res.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
    a = np.array(v)
    print(f"{name:6s} {N:5d} {sig:3d} {len(a):2d} | {fmt(a[:,0]):22s} | {fmt(a[:,1]):22s} | "
          f"{sorted(a[:,2].astype(int).tolist())}")


if __name__ == "__main__":
    x, t, J = truth_jets("cubic", 0, CASES["cubic"][0])
    jd = {key: J[key][it, ix] for key in ("u", "u_x", "u_xx", "u_xxx")}; jd["x"], jd["t"] = x[ix], t[it]
    for c in ((0, 0, -1, 0, 0), (0.01, 0, -1, 0, 0), (0.1, 0, -1, 0, 0)):
        A_n, scale, _ = normalise(build_matrix_learned(jd, c, 0.1))
        s = np.linalg.svd(A_n, compute_uv=False)
        print(c, "gap@3 =", round(gap_at(s, 3), 2))




