"""The symmetry loop (design notes/symmetry_loop_design.md, section 9; figure notes/figures/symmetry_loop.png).

One chain = one data set (name, N, sigma, k). Every training is a normal run of pinn.pinn_oracle (subprocess), so all
models and metrics rows are saved as usual; existing runs are skipped by pinn_oracle and simply read back.

    1. baseline B: stage 1 with the held-out split, seeds --base-seeds, keep the smaller held-out error E_B
    2. pass p: SVD of the current model M (network jets at its collocation points, its g_hat, nu_hat), with the column
       scales of B (fixed). Search only orthogonal to d/dx, d/dt and the accepted directions. Candidates = smallest
       singular vectors there.
    3. every candidate v ALONE: B's setup (same seed and split) + mu L_sym(v); R = E_v / E_B; accept if R <= 1 + delta.
       Stop the pass after --max-rej rejections or --max-cand candidates.
    4. nothing new accepted -> stop. Otherwise COMBINED training with all accepted (if only one is accepted in total,
       its own run is the combined model); check E_M / E_B <= 1 + delta (if not: reported, previous M kept).
       Next pass with the new M, at most --max-passes.
Every step is appended to results/oracle/loop/<chain>.jsonl; the end prints a summary with, for reading only, the
distance of the true generators from the accepted span (res, as in alignment_check: 0 = inside, 1 = unrelated).

    python -m discovery.sym_loop --name cubic --N 1000 --sig 1 --k 0
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from pinn.runs import load_rows
from pinn.surrogate import RESULTS, NU
from pinn.sym_general import rms_scale
from discovery.make_gen_file import baseline_columns, GENS_DIR
from discovery.fingerprint import KNOWN
from discovery.alignment_check import GENS

LOOP_DIR = RESULTS / "oracle" / "loop"
TRIVIAL = [0, 10]                      # columns xi:1 (d/dx) and tau:1 (d/dt)


def train(a, seed, gen_file=None):
    """Run pinn_oracle for one model (skipped there if it exists) and return its metrics row."""
    cmd = [sys.executable, "-m", "pinn.pinn_oracle", "1", "--name", a.name, "--n", str(a.N), "--sig", str(a.sig),
           "--k", str(a.k), "--seed", str(seed), "--g-model", a.arm, "--learn-nu", "0.05",
           "--holdout", str(a.holdout), "--steps", str(a.steps)]
    if gen_file:
        cmd += ["--gen-file", str(gen_file), "--mu", str(a.mu)] + (["--learn-gen"] if a.learn else [])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if a.quiet else None)
    rows = [r for r in load_rows(a.arm, mu=a.mu if gen_file else 0.0, names=[a.name], holdout=a.holdout)
            if (r["N"], r["sig"], r["k"], r["seed"], r.get("steps", 4000)) == (a.N, a.sig, a.k, seed, a.steps)
            and r.get("gen_file") == (str(gen_file) if gen_file else None)
            and bool(r.get("learn_gen")) == bool(gen_file and a.learn)]
    assert rows, f"no metrics row after training {cmd}"
    return rows[-1]


def orth_basis(vecs, n=30):
    """Orthonormal basis (columns) of the span of the given vectors."""
    if not vecs:
        return np.zeros((n, 0))
    U, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return U[:, s > 1e-10 * s[0]]


def candidates(M, scale, known, n_cand):
    """Smallest singular vectors of M's normalised columns, searched orthogonal to the known directions.
    Returns unit vectors in normalised coordinates and their singular values (smallest first)."""
    A_n = baseline_columns(M) / scale
    K = orth_basis(known)
    P = np.eye(30) - K @ K.T                               # projector onto the complement of the known span
    U, s, _ = np.linalg.svd(P)
    Q = U[:, s > 0.5]                                      # orthonormal basis of the complement
    _, sv, Vt = np.linalg.svd(A_n @ Q, full_matrices=False)
    order = np.argsort(sv)[:n_cand]
    V = (Q @ Vt[order].T).T
    return V / np.linalg.norm(V, axis=1, keepdims=True), sv[order] / np.sqrt(len(A_n)), A_n


def sparse_rotate(Vs, scale, n_starts=10, seed=0):
    """LieNLSD-style sparsification (their section 4.4): rotate the k given candidates inside their span so that they
    are as sparse as possible, measured as sum over vectors of L1/L2 of the ORIGINAL coefficients (scale-free).
    The span is unchanged. k <= 3 here, so the rotation has k(k-1)/2 angles: Nelder-Mead from several starts."""
    from scipy.linalg import expm
    from scipy.optimize import minimize
    k = len(Vs)
    if k < 2:
        return Vs
    Q, iu = Vs.T, np.triu_indices(k, 1)

    def rot(th):
        S = np.zeros((k, k)); S[iu] = th
        return expm(S - S.T)

    def cost(th):
        W = (Q @ rot(th)) / scale[:, None]
        return float(np.sum(np.abs(W).sum(0) / np.linalg.norm(W, axis=0)))

    rng = np.random.default_rng(seed)
    best = min((minimize(cost, rng.uniform(-np.pi, np.pi, len(iu[0])), method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12}) for _ in range(n_starts)),
               key=lambda o: o.fun)
    V = (Q @ rot(best.x)).T
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def clean(v, scale, thr):
    """Cleaning (design 2.3): in ORIGINAL coefficients, divide by the largest |component|, set components below thr to 0;
    back to a unit vector in normalised coordinates. Assumption: true symmetries are sparse in the basis."""
    w = v / scale
    w = w / np.abs(w).max()
    w = np.where(np.abs(w) < thr, 0.0, w)
    v = w * scale
    return v / np.linalg.norm(v)


def err_g(r):
    """RMS(g_hat - g_true) / RMS(u^3) at the clean u values of this data draw (as experiments/delta_pilot.py)."""
    from pinn.surrogate import DATA
    from pinn.runs import true_g
    from discovery.stage0b_general import g_poly, g_mlp, load_mlp
    d = np.load(DATA / f"sparse_{r['name']}_N{r['N']}_s{r['sig']}.npz")
    u = d["u_clean"][d["exp_id"] == r["k"]]
    g = g_poly(r["c"], u)[0] if r["g_model"] == "poly" else g_mlp(load_mlp(r), u)[0]
    return float(np.sqrt(np.mean((g - true_g(r["name"], u)[0]) ** 2)) / np.sqrt(np.mean(u ** 6)))


def write_gen(path, V, scale):
    V = np.atleast_2d(V)
    np.savez(path, V=V, scale=scale, W=V / scale)
    return path


def res_true(w, scale, accepted):
    """sin(angle) between a true generator and span{d/dx, d/dt, accepted}, in normalised coordinates."""
    w_n = w * scale
    K = orth_basis([np.eye(30)[i] for i in TRIVIAL] + list(accepted))
    return float(np.linalg.norm(w_n - K @ (K.T @ w_n)) / np.linalg.norm(w_n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--sig", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    ap.add_argument("--holdout", type=float, default=1.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--base-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-rej", type=int, default=2)
    ap.add_argument("--max-cand", type=int, default=5)
    ap.add_argument("--max-passes", type=int, default=2)
    ap.add_argument("--clean", type=float, default=None,
                    help="old form of --prep clean --thr X")
    ap.add_argument("--prep", choices=["none", "clean", "sparse"], default="none",
                    help="clean: zero the small components of each candidate; sparse: first rotate the first 3 candidates "
                         "to the sparsest basis of their span (LieNLSD 4.4), then clean")
    ap.add_argument("--thr", type=float, default=0.1, help="cleaning threshold (relative to the largest component)")
    ap.add_argument("--max-drift", type=float, default=10.0,
                    help="with --learn: a candidate whose trained vector moved more than this many degrees from its start "
                         "is rejected (it was replaced by another direction, not corrected)")
    ap.add_argument("--learn", action="store_true",
                    help="train the non-zero components of each candidate together with the model (pinn_oracle --learn-gen)")
    ap.add_argument("--steps", type=int, default=4000, help="Adam steps (smaller only for quick tests)")
    ap.add_argument("--quiet", action="store_true", help="hide the training output")
    a = ap.parse_args()
    if a.clean is not None:
        a.prep, a.thr = "clean", a.clean

    chain = f"{a.name}_N{a.N}_s{a.sig}_k{a.k}_{a.arm}_ho{a.holdout:g}" + (f"_ad{a.steps}" if a.steps != 4000 else "") \
        + {"none": "", "clean": f"_cl{a.thr:g}", "sparse": f"_sp{a.thr:g}"}[a.prep] \
        + (f"_md{a.max_drift:g}" if a.learn and a.max_drift != 10 else "") + ("_lg" if a.learn else "")
    LOOP_DIR.mkdir(parents=True, exist_ok=True); GENS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOOP_DIR / f"{chain}.jsonl"
    log_path.unlink(missing_ok=True)

    def log(**d):
        d["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(json.dumps(d) + "\n")
        print(" ", {k: v for k, v in d.items() if k not in ("v", "time")}, flush=True)

    # 1. baseline
    base = [train(a, s) for s in a.base_seeds]
    B = min(base, key=lambda r: r["err_hold"])
    E_B, seed = B["err_hold"], B["seed"]
    log(step="baseline", E=[r["err_hold"] for r in base], seeds=a.base_seeds, best_seed=seed, E_B=E_B,
        nu_hat=B["nu_hat"], c=B.get("c"))
    scale = rms_scale(baseline_columns(B))                   # fixed for the whole chain
    trivial = [np.eye(30)[i] for i in TRIVIAL]
    accepted, acc_rows, M = [], [], B

    for p in range(1, a.max_passes + 1):
        # 2. candidates from the current model
        V, sv, A_n = candidates(M, scale, trivial + accepted, a.max_cand)
        log(step="svd", p=p, model="baseline" if M is B else "combined", sing_values=sv.tolist())
        V_raw = V.copy()
        if a.prep == "sparse":
            n_rot = min(3, len(V))
            Vr = sparse_rotate(V[:n_rot], scale)
            Vr = np.array([clean(v, scale, a.thr) for v in Vr])
            Vr = Vr[np.argsort(np.linalg.norm(A_n @ Vr.T, axis=0))]      # best-satisfied first
            V = np.vstack([Vr] + [clean(v, scale, a.thr)[None] for v in V[n_rot:]])
        elif a.prep == "clean":
            V = np.array([clean(v, scale, a.thr) for v in V])
        new, n_rej = [], 0
        # 3. each candidate alone
        three = ("scaling", "Galilean", "projective")
        for j, v in enumerate(V, start=1):
            v_raw = V_raw[j - 1]
            K = orth_basis(trivial + accepted + new)
            dup = float(np.linalg.norm(v - K @ (K.T @ v)))
            if dup < 0.05:                                        # (almost) inside what is already accepted
                log(step="duplicate", p=p, j=j, dist=dup)
                continue
            gf = write_gen(GENS_DIR / f"L_{chain}_p{p}_c{j}.npz", v, scale)
            r = train(a, seed, gf)
            v_fin = np.array(r["gen_final"][0]) if a.learn and r.get("gen_final") else v
            R = r["err_hold"] / E_B
            drift = float(np.degrees(np.arccos(min(1.0, abs(float(v @ v_fin))))))
            K = orth_basis(trivial + accepted + new)
            dup_after = float(np.linalg.norm(v_fin - K @ (K.T @ v_fin)))
            reason = ("R" if R > 1 + a.delta else "drift" if drift > a.max_drift
                      else "duplicate after training" if dup_after < 0.05 else None)
            ok = reason is None
            log(step="candidate", p=p, j=j, R=R, R_clean=r["err_hold_clean"] / B["err_hold_clean"],
                accepted=ok, reason=reason, dup_after=dup_after,
                nu_hat=r["nu_hat"], c=r.get("c"), v=v.tolist(), v_final=v_fin.tolist(), drift_deg=drift,
                res={gn: res_true(KNOWN[gn], scale, [v]) for gn in three},
                res_raw={gn: res_true(KNOWN[gn], scale, [v_raw]) for gn in three},
                res_final={gn: res_true(KNOWN[gn], scale, [v_fin]) for gn in three})
            if ok:
                new.append(v_fin); acc_rows.append(r)
            elif reason == "duplicate after training":
                pass                                              # not a rejection: nothing new, just skipped
            else:
                n_rej += 1
                if n_rej >= a.max_rej:
                    break
        # 4. stop or combined training
        if not new:
            log(step="stop", p=p, reason="nothing new accepted")
            break
        accepted += new
        if len(accepted) == 1:
            M_new = acc_rows[0]                               # the single accepted run is the combined model
        else:
            M_new = train(a, seed, write_gen(GENS_DIR / f"L_{chain}_p{p}_comb.npz", np.array(accepted), scale))
        R_c = M_new["err_hold"] / E_B
        log(step="combined", p=p, n_accepted=len(accepted), R=R_c, ok=R_c <= 1 + a.delta, nu_hat=M_new["nu_hat"],
            c=M_new.get("c"))
        if R_c <= 1 + a.delta:
            M = M_new
        else:
            print("  WARNING: combined model fails the check; previous model kept")
        if p == a.max_passes:
            log(step="stop", p=p, reason="max_passes reached")

    # summary
    true_gens = GENS.get(a.name, [])
    log(step="result", n_accepted=len(accepted), final="baseline" if M is B else "combined",
        nu_err=abs(M["nu_hat"] - NU) / NU, nu_err_base=abs(B["nu_hat"] - NU) / NU, c=M.get("c"), c_base=B.get("c"),
        err_g=err_g(M), err_g_base=err_g(B),
        res_true_in_accepted={gn: res_true(KNOWN[gn], scale, accepted) for gn in true_gens},
        res_scaling_in_accepted=res_true(KNOWN["scaling"], scale, accepted))
    print(f"\n{chain}: {len(accepted)} accepted -> log {log_path}")
