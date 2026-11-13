import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from discovery.noise_gap import ALL_CASES, EPS, column_raw, evaluate7
from discovery.fingerprint import normalise, projection_residual, KNOWN,RESULTS
from pinn.discover_from_surrogate import load_models, surrogate_jets, N_COLLOC
from pinn.surrogate import variant_tag

matplotlib.use("Agg")

NAMES = ["cubic"] + [f"cubic_{t}" for t in EPS]
EPS_OF = {"cubic": 0.0, **{f"cubic_{t}": e for t, e in EPS.items()}}
SIGS, SEEDS, M_MAX = [1, 3], [0, 1, 2], 8

def analyse(name, cols, sig, seed):
    """Return the 8 smallest singular values and the scaling residual for m = 1..8."""
    models = load_models(name, 1000, sig, prefix="oracle_", tag="_lam1",
                         suffix=variant_tag() + f"_seed{seed}", seed=seed)
    jd = surrogate_jets(models, N_COLLOC, np.random.default_rng(0))     # same points as discover_from_surrogate
    A = np.column_stack([evaluate7(c,jd) for c in cols])
    A_n, scale, _ = normalise(A)
    _, s, Vt = np.linalg.svd(A_n, full_matrices=False)
    w = KNOWN["scaling"]
    res = [projection_residual(w, Vt, m, scale) for m in range(1, M_MAX + 1)]
    return s[-M_MAX:][::-1], np.array(res)           # s_{-1}, s_{-2}, ..., s_{-8}

if __name__ == "__main__":
    out = {}
    for name in NAMES:
        g, _ = ALL_CASES[name]
        cols = column_raw(g)                          # sympy is slow: once per name
        for sig in SIGS:
            runs = [analyse(name, cols, sig, seed) for seed in SEEDS]
            out[(name, sig)] = (np.array([r[0] for r in runs]), np.array([r[1] for r in runs]))
    print("\nTABLE 1: third smallest singular values and the ratio r to the eps=0 baseline")
    print(f"{'case':<11}{'eps':>6}{'sig%':>5}{'s_-3 mean':>12}{'std':>10}{'r':>7}{'max s_-1':>11}{'max s_-2':>11}")
    for sig in SIGS:
        base = out[("cubic",sig)][0][:,2].mean()
        for name in NAMES:
            S = out[(name,sig)][0]
            s3 = S[:,2]
            print(f"{name:<11}{EPS_OF[name]:>6.2f}{sig:>5}{s3.mean():>12.2e}{s3.std():>10.1e}"
                  f"{s3.mean()/base:>7.2f}{S[:,0].max():>11.1e}{S[:,1].max():>11.1e}")
    print("\nTABLE 2: scaling residual (mean over seeds) when the null space has dimension m")
    print(f"{'case':<11}{'eps':>6}{'sig%':>5}{'m=2':>8}{'m=3':>8}{'m=4':>8}")

    for sig in SIGS:
        for name in NAMES:
            R = out[(name,sig)][1]
            m2,m3,m4 = (R[:,m-1].mean() for m in (2,3,4))
            print(f"{name:<11}{EPS_OF[name]:>6.2f}{sig:>5}{m2:>8.3f}{m3:>8.3f}{m4:>8.3f}")

    print("\nDECISION RULE: smallest eps with r >= 2")
    for sig in SIGS:
        base = out[("cubic",sig)][0][:,2].mean()
        detected = [EPS_OF[n] for n in NAMES
        if EPS_OF[n]>0 and out[(n,sig)][0][:,2].mean() / base >= 2]
        print(f"sigma={sig}%: " + (f"{min(detected)}" if detected else "none"))

    np.savez(RESULTS / "near_symmetry.npz",
             **{f"S_{n}_s{sig}": out[(n, sig)][0] for n in NAMES for sig in SIGS},
             **{f"R_{n}_s{sig}": out[(n, sig)][1] for n in NAMES for sig in SIGS})

    fig, ax = plt.subplots(figsize=(6, 4))
    eps_grid = np.array([EPS_OF[n] for n in NAMES[1:]])
    for sig, marker in zip(SIGS, ["o", "s"]):
        mean = [out[(n, sig)][0][:, 2].mean() for n in NAMES[1:]]
        std = [out[(n, sig)][0][:, 2].std() for n in NAMES[1:]]
        ax.errorbar(eps_grid, mean, yerr=std, marker=marker, label=f"oracle, sigma={sig}%")
        ax.axhline(out[("cubic", sig)][0][:, 2].mean(), ls=":", lw=1)   # dotted line: noise floor
    ax.plot(eps_grid, 0.076*eps_grid, "k--", lw=1, label="noise-free: 0.076 eps")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("eps"); ax.set_ylabel("third-smallest singular value s_-3"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "near_symmetry.png", dpi=150)
    print("saved")
    # TODO 1: print a table, one row per (eps, sig): mean +- std over seeds of s_{-3},
    #         the ratio r = mean s_{-3}(eps) / mean s_{-3}(0), and max over seeds of s_{-1}, s_{-2}.
    # TODO 2: print the scaling residual for m = 2, 3, 4 (mean over seeds) per (eps, sig).
    # TODO 3: per sig, print the smallest eps with r >= 2 (the decision rule), or "none".
    # TODO 4: np.savez("results/near_symmetry.npz", ...) with the arrays, so you can plot later.