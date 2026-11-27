"""Offline re-evaluation of the drift limit (POST-HOC, no training): what would pass 1 of each learnable-coefficient
chain have accepted with a different acceptance rule?

    rule "logged"   as run:  R <= 1 + delta  and  drift <= 10 deg  and  not a duplicate
    rule "nodrift"  proposed: R <= 1 + delta  and  not a duplicate        (drift ignored)

Only the chains run with the 10 deg limit (chains with "_mdX" in their name already ran without it).
Uses only the logged pass-1 candidates (trained vector v_final, R, drift). Limits:
  - pass 2 and the combined training depend on what was accepted, so they are NOT re-evaluated;
  - the loop stops a pass after 2 rejections; if rule "nodrift" would have accepted one of those, the pass would have
    continued with candidates that were never trained -> the chain is marked "cut" (its nodrift count is a lower bound).
Columns per cell (3 data draws): accepted per draw under both rules; for cubic / g0 how many draws have all true
symmetries in the accepted span (res < 0.1); for the eps data how many draws accept scaling (FALSE acceptance).

    python -m experiments.reeval_drift                 # all *_lg chains
    python -m experiments.reeval_drift --arm poly --N 1000
"""
import argparse, json
import numpy as np
from pinn.surrogate import RESULTS
from discovery.fingerprint import KNOWN
from discovery.sym_loop import orth_basis, TRIVIAL, res_true, GENS_DIR
from experiments.loop_eval import PAT, PREP, ORDER, EXPECTED

TRUE = {"cubic": ["scaling"], "g0": ["scaling", "Galilean", "projective"]}


def decide(cands, rule, delta=0.1, max_drift=10.0):
    """Pass-1 decisions under a rule. Returns the accepted (trained) vectors and 'cut': the logged pass ended before
    this rule would have stopped it (fewer than 2 rejections and fewer than 5 candidates tried), so untrained
    candidates are missing and the count is a lower bound."""
    triv = [np.eye(30)[i] for i in TRIVIAL]
    acc, n_rej = [], 0
    for d in cands:
        if n_rej >= 2:                      # the loop would have stopped here under this rule
            break
        v = np.array(d["v_final"])
        K = orth_basis(triv + acc)
        dup = float(np.linalg.norm(v - K @ (K.T @ v))) < 0.05
        if d["R"] > 1 + delta or (rule == "logged" and d["drift_deg"] > max_drift):
            n_rej += 1
        elif not dup:
            acc.append(v)
    cut = n_rej < 2 and max(d["j"] for d in cands) < 5
    return acc, cut


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["poly", "MLP"])
    ap.add_argument("--N", type=int)
    a = ap.parse_args()
    cells = {}
    for f in sorted((RESULTS / "oracle" / "loop").glob("*_lg.jsonl")):
        m = PAT.match(f.stem)
        if not m or m["ad"] or m["md"] or (a.arm and m["arm"] != a.arm) or (a.N and int(m["N"]) != a.N):
            continue
        steps = [json.loads(l) for l in open(f)]
        cands = [d for d in steps if d["step"] == "candidate" and d["p"] == 1 and "v_final" in d]
        if not any(d["step"] == "result" for d in steps) or not cands:
            continue
        scale = np.load(GENS_DIR / f"L_{f.stem}_p1_c1.npz")["scale"]
        out = {}
        for rule in ("logged", "nodrift"):
            acc, cut = decide(cands, rule)
            name = m["name"]
            if name in TRUE:
                hit = max(res_true(KNOWN[g], scale, acc) for g in TRUE[name]) < 0.1 if acc else False
            else:
                hit = res_true(KNOWN["scaling"], scale, acc) < 0.1 if acc else False   # FALSE acceptance
            out[rule] = (len(acc), hit, cut)
        key = (m["arm"], int(m["N"]), int(m["sig"]), PREP(m["prep"]), m["name"])
        cells.setdefault(key, []).append((int(m["k"]), out))

    print("pass 1 only, POST-HOC.  acc = accepted per draw;  ok = cubic/g0: all true symmetries found, "
          "eps: scaling FALSELY accepted;  * = pass was cut short (lower bound)")
    print(f"{'arm':4s} {'N':>4s} {'sig':>3s} {'prep':6s} {'data':10s} exp | {'logged (10 deg)':22s} | {'no drift limit':22s}")
    for key in sorted(cells, key=lambda c: (c[0], -c[1], c[2], c[3], ORDER.get(c[4], 9))):
        runs = sorted(cells[key])
        txt = []
        for rule in ("logged", "nodrift"):
            acc = ",".join(f"{o[rule][0]}{'*' if o[rule][2] else ''}" for _, o in runs)
            hits = sum(o[rule][1] for _, o in runs)
            lab = "false" if key[4] not in TRUE else "ok"
            txt.append(f"acc {acc:9s} {lab} {hits}/{len(runs)}")
        arm, N, sig, prep, name = key
        print(f"{arm:4s} {N:4d} {sig:3d} {prep:6s} {name:10s} {EXPECTED.get(name, '?'):3s} | {txt[0]:22s} | {txt[1]:22s}")
