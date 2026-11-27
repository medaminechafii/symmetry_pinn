"""Evaluation of the symmetry loop over all finished chains (results/oracle/loop/*.jsonl).

One row per cell (arm, N, sigma, preparation, data set); the 3 data draws k = 0, 1, 2 are summarised:
    acc        number of accepted symmetries per draw (expected: cubic 1, g0 3, eps 0.1 0, eps 0.03 probably 1)
    true found how many draws have ALL true symmetries inside the accepted span (res < 0.1 for each true generator);
               worst res = median over draws of the largest res of the true generators (0 = all found exactly)
    false acc  eps data only: draws where scaling (NOT a symmetry there) ended up in the accepted span (res < 0.1)
    err_g      median over draws, baseline -> final model: RMS(g_hat - g_true) / RMS(u^3) at the data's u values
    nu err     median over draws, baseline -> final model, in %
Only chains of the new method (learnable coefficients, *_lg) unless --all. Unfinished chains are counted separately.
Chains that ran before the drift rule existed are evaluated as they were logged. Chains run with another drift limit
(sym_loop --max-drift X, X != 10) carry "_mdX" in their name and are shown as separate cells ("clean+learn md180").

    python -m experiments.loop_eval
    python -m experiments.loop_eval --arm poly --N 1000
"""
import argparse, json, re
import numpy as np
from pinn.surrogate import RESULTS

PAT = re.compile(r"^(?P<name>.+)_N(?P<N>\d+)_s(?P<sig>\d+)_k(?P<k>\d)_(?P<arm>poly|MLP)_ho(?P<ho>[\d.]+)"
                 r"(?P<ad>_ad\d+)?(?P<prep>_cl[\d.]+|_sp[\d.]+)?(?P<md>_md[\d.]+)?(?P<lg>_lg)?$")
EXPECTED = {"cubic": "1", "g0": "3", "cubic_e01": "0", "cubic_e003": "0/1"}
ORDER = {"cubic": 0, "g0": 1, "cubic_e01": 2, "cubic_e003": 3}
PREP = lambda p: "none" if not p else ("clean" if p.startswith("_cl") else "sparse")


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also the chains without learnable coefficients")
    ap.add_argument("--arm", choices=["poly", "MLP"])
    ap.add_argument("--N", type=int)
    a = ap.parse_args()

    cells, unfinished = {}, 0
    for f in sorted((RESULTS / "oracle" / "loop").glob("*.jsonl")):
        m = PAT.match(f.stem)
        if not m or m["ad"] or (not m["lg"] and not a.all):
            continue
        if (a.arm and m["arm"] != a.arm) or (a.N and int(m["N"]) != a.N):
            continue
        res = [json.loads(l) for l in open(f) if '"step": "result"' in l]
        if not res:
            unfinished += 1
            continue
        key = (m["arm"], int(m["N"]), int(m["sig"]), PREP(m["prep"]) + ("+learn" if m["lg"] else "") + (f" md{m['md'][3:]}" if m["md"] else ""), m["name"])
        cells.setdefault(key, []).append((int(m["k"]), res[0]))

    head = (f"{'arm':4s} {'N':>4s} {'sig':>3s} {'prep':18s} {'data':10s} | {'acc (k0,k1,k2)':14s} exp | "
            f"{'true found':10s} {'worst res':>9s} | {'false acc':9s} | {'err_g base -> final':19s} | nu err % base -> final")
    print(head); print("-" * len(head))
    last = None
    for key in sorted(cells, key=lambda c: (c[0], -c[1], c[2], c[3], ORDER.get(c[4], 9))):
        arm, N, sig, prep, name = key
        if last and last != key[:4]:
            print()
        last = key[:4]
        runs = sorted(cells[key], key=lambda x: x[0])
        acc = ",".join(str(r["n_accepted"]) for _, r in runs)
        if name in ("cubic", "g0"):
            worst = [max(r["res_true_in_accepted"].values()) for _, r in runs]
            found = f"{sum(w < 0.1 for w in worst)}/{len(runs)}"
            worst_txt, false_txt = f"{med(worst):9.3f}", f"{'-':9s}"
        else:
            found, worst_txt = "-", f"{'-':>9s}"
            false_txt = f"{sum(r['res_scaling_in_accepted'] < 0.1 for _, r in runs)}/{len(runs)}"
            false_txt = f"{false_txt:9s}"
        eg = (f"{med([r.get('err_g_base') for _, r in runs]):7.4f} -> {med([r.get('err_g') for _, r in runs]):7.4f}"
              if all("err_g" in r for _, r in runs) else f"{'(not logged)':19s}")
        nu = f"{100 * med([r['nu_err_base'] for _, r in runs]):.3f} -> {100 * med([r['nu_err'] for _, r in runs]):.3f}"
        print(f"{arm:4s} {N:4d} {sig:3d} {prep:18s} {name:10s} | {acc:14s} {EXPECTED.get(name, '?'):3s} | "
              f"{found:10s} {worst_txt} | {false_txt} | {eg:19s} | {nu}")
    print(f"\n{sum(len(v) for v in cells.values())} finished chains in {len(cells)} cells; {unfinished} unfinished")
