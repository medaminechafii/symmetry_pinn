"""Data-free cleaning check (design notes/symmetry_loop_design.md 2.3): does setting the small components of an SVD
candidate to zero bring it closer to a true symmetry?  No training: it reads the candidate files that the loop already
wrote (results/oracle/gens/L_<chain>_p<p>_c<j>.npz: V = unit vector in normalised coordinates, scale, W = V / scale).

Cleaning (in the ORIGINAL coefficients, i.e. the 30 numbers of xi, tau, phi on 1, x, t, u, x^2, t^2, u^2, xt, xu, tu):
    divide by the largest |component|, set every component with |value| < thr to 0, renormalise.
Assumption: true symmetries are sparse in this basis (scaling has 3 non-zero entries).

Printed per candidate (for reading only; the true generators are never used by the loop):
    res      sin(angle) of the closest true symmetry to span{d/dx, d/dt, candidate}: before / after cleaning, for
             thr = 0.05, 0.1, 0.2   (0 = exactly that symmetry, 0.1 ~ 6 deg, 1 = unrelated)
    kept     the components left after cleaning with thr = 0.1 (normalised so the largest is 1)
For cubic the true symmetry is scaling; for g0 any of scaling / Galilean / projective (candidates may be mixtures,
then the res to each single one stays large; see the column 'span3' = distance from span of all three);
for the eps data scaling is NOT a symmetry (the row shows whether cleaning would make the false candidate look exact).

    python -m discovery.clean_check                    # all candidates of all chains
    python -m discovery.clean_check --chain cubic      # only chains whose name starts with cubic_N
"""
import argparse, json
import numpy as np
from pinn.surrogate import RESULTS
from discovery.fingerprint import KNOWN
from discovery.criterion import MONO_NAMES
from discovery.sym_loop import res_true, orth_basis, TRIVIAL

GENS_DIR = RESULTS / "oracle" / "gens"
LOOP_DIR = RESULTS / "oracle" / "loop"
NAMES = [f"{c}:{m}" for c in ("xi", "tau", "phi") for m in MONO_NAMES]
THRS = (0.05, 0.1, 0.2)
TRUE3 = ("scaling", "Galilean", "projective")


def clean(w, thr):
    w = w / np.abs(w).max()
    return np.where(np.abs(w) < thr, 0.0, w)


def res_vec(w_true, scale, w_cand):
    """res of a true generator to span{d/dx, d/dt, candidate}; candidate in ORIGINAL coefficients."""
    v = w_cand * scale
    return res_true(w_true, scale, [v / np.linalg.norm(v)])


def res_span3(w_cand, scale):
    """distance of the candidate from span{d/dx, d/dt, scaling, Galilean, projective} (0 = a mixture of true g0 ones)."""
    v = w_cand * scale; v = v / np.linalg.norm(v)
    K = orth_basis([np.eye(30)[i] for i in TRIVIAL] + [KNOWN[g] * scale for g in TRUE3])
    return float(np.linalg.norm(v - K @ (K.T @ v)))


def kept(w):
    w = clean(w, 0.1)
    return "  ".join(f"{NAMES[i]} {w[i]:+.2f}" for i in np.flatnonzero(w))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="", help="only chains whose name starts with this (e.g. cubic_N, g0, cubic_e01)")
    args = ap.parse_args()
    files = sorted(GENS_DIR.glob(f"L_{args.chain}*_c*.npz"))
    if not files:
        raise SystemExit(f"no candidate files in {GENS_DIR}")
    decided = {}
    for f in LOOP_DIR.glob("*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            if d["step"] == "candidate":
                decided[(f.stem, d["p"], d["j"])] = (d["R"], d["accepted"])
    print(f"{'chain / candidate':42s} {'R':>8s}  res to closest true symmetry: raw | thr " + " / ".join(map(str, THRS))
          + "   span3 raw/0.1")
    for f in files:
        chain, rest = f.stem[2:].rsplit("_p", 1)
        p, j = rest.split("_c")
        g = np.load(f)
        w, scale = g["W"][0], g["scale"]
        name = chain.split("_N")[0]
        gens = ["scaling"] if name != "g0" else list(TRUE3)
        best = lambda wc: min((res_vec(KNOWN[gn], scale, wc), gn) for gn in gens)
        r0, g0n = best(w)
        rc = [best(clean(w, t)) for t in THRS]
        R, acc = decided.get((chain, int(p), int(j)), (float("nan"), None))
        tag = "" if acc is None else ("ACC" if acc else "rej")
        print(f"{chain + f'  p{p} c{j}':42s} {R:8.3f} {tag:3s} {g0n:10s} {r0:5.3f} | "
              + " / ".join(f"{r:5.3f}" + ("" if gn == g0n else f" ({gn})") for r, gn in rc)
              + f"   {res_span3(w, scale):5.3f}/{res_span3(clean(w, 0.1), scale):5.3f}")
        print(f"{'':46s} kept (thr 0.1): {kept(w)}")
