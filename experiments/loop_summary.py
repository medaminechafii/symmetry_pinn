"""Summary of the symmetry-loop chains (results/oracle/loop/*.jsonl, written by discovery/sym_loop.py).

Per chain: number accepted (expected: cubic 1, g0 up to 3, eps 0), every candidate's R and decision, the final model's
nu error and coefficients against the baseline, and (for reading only) how far the true generators are from the
accepted span (res: 0 = inside, 1 = unrelated; for eps data the row shows scaling, which is NOT a true symmetry there).

    python -m experiments.loop_summary
"""
import json
from pinn.surrogate import RESULTS

EXPECTED = {"cubic": "1 (scaling)", "g0": "up to 3", "cubic_e01": "0", "cubic_e003": "0 (probably 1)"}


def fmt_c(c):
    return " ".join(f"{x:6.3f}" for x in c) if c else "-"


if __name__ == "__main__":
    files = sorted((RESULTS / "oracle" / "loop").glob("*.jsonl"))
    order = {"cubic": 0, "g0": 1, "cubic_e01": 2, "cubic_e003": 3}
    chains = []
    for f in files:
        steps = [json.loads(l) for l in open(f)]
        name = f.stem.split("_N")[0]
        chains.append((order.get(name, 9), f.stem, name, steps))
    for _, stem, name, steps in sorted(chains):
        res = [s for s in steps if s["step"] == "result"]
        print(f"\n=== {stem}   (expected accepted: {EXPECTED.get(name, '?')})" + ("" if res else "   [not finished]"))
        for s in steps:
            if s["step"] == "baseline":
                print(f"  baseline  E seeds {['%.2e' % e for e in s['E']]} -> E_B {s['E_B']:.2e}   c {fmt_c(s['c'])}")
            elif s["step"] == "candidate":
                r = s["res"]
                print(f"  pass {s['p']} cand {s['j']}:  R {s['R']:9.3f}  R_clean {s['R_clean']:9.3f}  "
                      f"{'ACCEPT' if s['accepted'] else 'reject'}   res scal/Gal/proj "
                      f"{r['scaling']:.2f}/{r['Galilean']:.2f}/{r['projective']:.2f}")
            elif s["step"] == "combined":
                print(f"  pass {s['p']} combined ({s['n_accepted']} accepted):  R {s['R']:.3f}  "
                      f"{'ok' if s['ok'] else 'FAILS check'}   c {fmt_c(s['c'])}")
            elif s["step"] == "stop":
                print(f"  stop after pass {s['p']}: {s['reason']}")
        if res:
            r = res[0]
            print(f"  RESULT: {r['n_accepted']} accepted, final {r['final']};  nu err {100*r['nu_err']:.2f}% "
                  f"(baseline {100*r['nu_err_base']:.2f}%);  c {fmt_c(r['c'])}")
            txt = ", ".join(f"{g} {v:.2f}" for g, v in r["res_true_in_accepted"].items())
            print(f"  true generators vs accepted span (res): {txt or '-'};  scaling {r['res_scaling_in_accepted']:.2f}")
