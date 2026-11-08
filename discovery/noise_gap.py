from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from discovery.criterion import columns, basis_fields, criterion,NU, x, t, u, ux, ut, uxx, uxt
from discovery.fingerprint import (sample_jet, normalise, count_by_gap, KNOWN, projection_residual,DATA,RESULTS,SEED,N_POINTS)

CASES = {"g0":(sp.Integer(0),5), "cubic":(-u**3,3)}
SIGMAS = [0,0.001,0.003,0.01,0.03,0.1]
NOISY = ["u", "u_x", "u_t", "u_xx", "u_xt"]
JET7 = [x,t,u,ux,ut,uxx,uxt]
JET7_KEY = ["x","t","u","u_x","u_t","u_xx","u_xt"]
def column_raw(g):
    return [criterion(*f,g) for f in basis_fields()[0]]
def evaluate7(expr, jet):
    """F4: lambdify expr in the jet variables JET and evaluate it at the sampled points.
    Constant columns (the two zero columns!) need the broadcast_to fix from toy lesson 2."""
    fun = sp.lambdify(JET7, expr, "numpy")
    val = fun(*(jet[k] for k in JET7_KEY))
    return np.broadcast_to(np.asarray(val,dtype=float),jet["x"].shape).copy()    

def clean_jet(d, g, rng):
    jet = sample_jet(d, N_POINTS, rng)
    g_fun = sp.lambdify(u, g, "numpy")
    dg_fun = sp.lambdify(u, sp.diff(g,u), "numpy")
    uu = jet["u"]
    g_val = np.broadcast_to(np.asarray(g_fun(uu),dtype=float),uu.shape)
    dg_val = np.broadcast_to(np.asarray(dg_fun(uu),dtype=float),uu.shape)
    nu = float(NU)
    jet["u_t"] = -uu*jet["u_x"] + nu*jet["u_xx"] + g_val
    jet["u_xt"] = -jet["u_x"]**2 - uu*jet["u_xx"] + nu*jet["u_xxx"] + dg_val*jet["u_x"]
    return jet
def add_noise(jet,sigma,rng):
    noisy = dict(jet)
    for key in NOISY:
        std = sigma*np.std(jet[key])
        noisy[key] += rng.normal(0,std,size = jet[key].shape)
    return noisy
def gap_at(s,k):
    s = np.maximum(s,s[0]*1e-18)
    return np.log10(s[-k-1]/s[-k])
if __name__ == "__main__":
    results = {}
    for name, (g, k) in CASES.items():
        d    = np.load(DATA / f"dense_{name}.npz")
        cols = column_raw(g)                                  # sympy is slow: do it ONCE per g
        jet0 = clean_jet(d, g, np.random.default_rng(SEED))    # same points for every sigma

        res = {"s": [], "gap": [], "count": [], "resid": {gn: [] for gn in KNOWN}}
        for sigma in SIGMAS:
            jet = add_noise(jet0, sigma, np.random.default_rng(1))
            A   = np.column_stack([evaluate7(c, jet) for c in cols])        # use evaluate7, not Part A's build_matrix
            A_n, scale, _ = normalise(A)
            _, s, Vt      = np.linalg.svd(A_n, full_matrices=False)
            res["s"].append(s)
            res["gap"].append(gap_at(s, k))
            res["count"].append(count_by_gap(s)[0])
            for gn, w in KNOWN.items():
                res["resid"][gn].append(projection_residual(w, Vt, k, scale))   # expected k, not the count
            print(f"{name}  sigma={sigma:<6} count={res['count'][-1]}  gap@k={res['gap'][-1]:5.1f}  "
                  + "  ".join(f"{gn}={res['resid'][gn][-1]:.1e}" for gn in KNOWN))
        results[name] = res
    labels = {s: f"{100*s:g}%" for s in SIGMAS}

    # Figure 1: singular values, one panel per g, one curve per noise level
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, (name, (g, k)) in zip(axes, CASES.items()):
        for sigma, s in zip(SIGMAS, results[name]["s"]):
            s = np.maximum(s, s[0] * 1e-18)
            ax.semilogy(np.arange(1, 31), s, "o-", ms=3, label=labels[sigma])
        ax.axvline(30 - k + 0.5, ls=":", color="gray")
        ax.set_title(f"{name}: g = {sp.sstr(g)}  (expected {k})")
        ax.set_xlabel("index"); ax.grid(alpha=0.3); ax.legend(title="noise", fontsize=8)
    axes[0].set_ylabel("singular value (normalised)")
    fig.savefig(RESULTS / "noise_singular_values.png", dpi=150)

    # Figure 2: generator residuals vs noise (sigma = 0 cannot go on a log axis, so skip it)
    sig = np.array(SIGMAS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, CASES):
        for gn, r in results[name]["resid"].items():
            ax.loglog(100 * sig, np.maximum(r[1:], 1e-17), "o-", label=gn)
        ax.set_title(name); ax.set_xlabel("noise level (%)"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("projection residual")
    fig.savefig(RESULTS / "noise_residuals.png", dpi=150)
    plt.show()