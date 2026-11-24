"""Which learned coefficient is responsible for the low scaling gap?  (cubic case, polynomial arm)

For every learned c = (c1..c5):
    full      the learned equation as it is
    c3 only   keep c3, zero the others: a homogeneous equation, scaling is exact -> gap should be about 13.5
    +ck       c3 plus ONE other coefficient (k = 1, 2, 4, 5)       -> how much does that coefficient alone lower the gap?
    -ck       the full equation with ONE coefficient set to zero   -> how much of the gap does removing it restore?
Also the median |c_k|, so the size of each coefficient can be compared with its effect.

    python -m discovery.attribution_check            # real poly rows if metrics.jsonl has them, else a demo vector
(Own imports from discovery/eps_eff_check.py, which has a main guard.)
"""
import json
import numpy as np
from discovery.eps_eff_check import gap3, points
from discovery.fingerprint import N_POINTS
from discovery.noise_gap import CASES
from discovery.stage0_floor import truth_jets
from pinn.surrogate import RESULTS

OTHERS = (0, 1, 3, 4)                                  # indices of c1, c2, c4, c5


def variants(c):
    c = np.asarray(c, dtype=float)
    base = np.zeros(5); base[2] = c[2]
    out = {"full": c, "c3 only": base}
    for k in OTHERS:
        add = base.copy(); add[k] = c[k]; out[f"+c{k+1}"] = add
    for k in OTHERS:
        drop = c.copy(); drop[k] = 0.0; out[f"-c{k+1}"] = drop
    return out


def gaps_of(jd, c, nu):
    return {name: gap3(jd, vec, nu) for name, vec in variants(c).items()}


if __name__ == "__main__":
    x0, t0, _ = truth_jets("cubic", 0, CASES["cubic"][0])
    rng = np.random.default_rng(0)
    it, ix = rng.integers(0, len(t0), N_POINTS), rng.integers(0, len(x0), N_POINTS)   # same points as eps_eff_check

    path = RESULTS / "oracle" / "metrics.jsonl"
    rows = []
    if path.exists():
        rows = [r for r in map(json.loads, open(path))
                if r.get("g_model") == "poly" and r.get("learn_nu") and r["name"] == "cubic" and not r.get("mu")]   # stage 1 only

    if not rows:
        print("no cubic poly rows here: demo with the coefficient vector of the earlier Mac run")
        jd = points("cubic", 0, it, ix)
        c = (2.9e-4, 2.9e-3, -0.940, 9.0e-3, -0.0806)
        for name, g in gaps_of(jd, c, 0.1).items():
            print(f"  {name:8s} gap@3 = {g:6.2f}")
        raise SystemExit

    print(len(rows), "cubic poly rows")
    cache, res, coef = {}, {}, {}
    for r in rows:
        if r["k"] not in cache:
            cache[r["k"]] = points("cubic", r["k"], it, ix)
        key = (r["N"], r["sig"])
        res.setdefault(key, []).append(gaps_of(cache[r["k"]], r["c"], r["nu_hat"]))
        coef.setdefault(key, []).append(np.abs(r["c"]))

    names = ["full", "c3 only", "+c1", "+c2", "+c4", "+c5", "-c1", "-c2", "-c4", "-c5"]
    print("\nmedian gap@3 over the runs of each cell ('+ck': c3 plus only ck;  '-ck': full equation without ck)")
    print(f"{'N':>5s} {'sig':>3s} | " + " ".join(f"{n:>8s}" for n in names))
    for key in sorted(res, key=lambda kk: (-kk[0], kk[1])):
        med = [np.median([g[n] for g in res[key]]) for n in names]
        print(f"{key[0]:5d} {key[1]:3d} | " + " ".join(f"{m:8.2f}" for m in med))

    print("\nmedian |c_k| over the runs of each cell")
    print(f"{'N':>5s} {'sig':>3s} | " + " ".join(f"{'c'+str(k):>8s}" for k in range(1, 6)))
    for key in sorted(coef, key=lambda kk: (-kk[0], kk[1])):
        print(f"{key[0]:5d} {key[1]:3d} | " + " ".join(f"{m:8.3f}" for m in np.median(coef[key], axis=0)))
