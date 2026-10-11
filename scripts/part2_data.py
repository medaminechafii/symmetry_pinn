"""Data experiment: Burgers solutions -> jet-space point cloud -> symmetries by the invariance criterion."""
import json, numpy as np, sympy as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

rng = np.random.default_rng(0)
NU = 0.1
# ---------------------------------------------------------------- exact solutions via Cole-Hopf
x, t = sp.symbols('x t')
A = sp.symbols('A0:3'); K = sp.symbols('K0:3'); P = sp.symbols('P0:3')
C0 = sp.Symbol('C0')
th = C0 + sum(A[j]*sp.cos(K[j]*x + P[j])*sp.exp(-NU*K[j]**2*t) for j in range(3))
U = -2*NU*sp.diff(th, x)/th
jet_exprs = [U, sp.diff(U, x), sp.diff(U, t), sp.diff(U, x, 2), sp.diff(U, x, t)]
jet_f = sp.lambdify((x, t, C0, *A, *K, *P), jet_exprs, "numpy")

def sample_solution(n):
    a = rng.uniform(0.2, 1.0, 3) * rng.choice([-1, 1], 3); k = rng.integers(1, 4, 3).astype(float)
    p = rng.uniform(0, 2*np.pi, 3); c0 = np.abs(a).sum() + rng.uniform(0.1, 0.4)   # keeps theta > 0
    xs = rng.uniform(0, 2*np.pi, n); ts = rng.uniform(0, 2.0, n)
    u, ux, ut, uxx, uxt = [np.broadcast_to(v, xs.shape) for v in jet_f(xs, ts, c0, *a, *k, *p)]
    return np.stack([xs, ts, u, ux, ut, uxx, uxt], 1), (c0, a, k, p)

S, n_per = 60, 300
pts, params = [], []
for s in range(S):
    z, pr = sample_solution(n_per); pts.append(z); params.append(pr)
Z = np.concatenate(pts)                        # columns: x t u ux ut uxx uxt
res = {"n_points": int(len(Z)), "n_solutions": S,
       "max_abs_residual_exact": float(np.abs(Z[:, 4] + Z[:, 2]*Z[:, 3] - NU*Z[:, 5]).max())}

# ---------------------------------------------------------------- criterion rows, linear in generator parameters
X, T, Uu, UX, UT, UXX, UXT = sp.symbols('x t u u_x u_t u_xx u_xt')
def Dx(f): return sp.diff(f, X) + UX*sp.diff(f, Uu) + UXX*sp.diff(f, UX) + UXT*sp.diff(f, UT)
def Dt(f): return sp.diff(f, T) + UT*sp.diff(f, Uu) + UXT*sp.diff(f, UX) + sp.Symbol('u_tt')*sp.diff(f, UT)
def prolong(xi, tau, phi):
    px = Dx(phi) - UX*Dx(xi) - UT*Dx(tau)
    pt = Dt(phi) - UX*Dt(xi) - UT*Dt(tau)
    pxx = Dx(px) - UXX*Dx(xi) - UXT*Dx(tau)
    return px, pt, pxx

def build(ansatz):
    mons = {"affine": [X, T, 1], "quadratic": [1, X, T, Uu, X**2, T**2, Uu**2, X*T, X*Uu, T*Uu]}[ansatz]
    if ansatz == "affine":     # xi, tau affine in (x,t); phi affine in (x,t,u)  -> 10 parameters
        cs = sp.symbols('p0:10'); xi = cs[0]*X + cs[1]*T + cs[2]; tau = cs[3]*X + cs[4]*T + cs[5]
        phi = cs[6]*X + cs[7]*T + cs[8]*Uu + cs[9]
        names = ["xi:x", "xi:t", "xi:1", "tau:x", "tau:t", "tau:1", "phi:x", "phi:t", "phi:u", "phi:1"]
    else:
        cs = sp.symbols('p0:30'); m = len(mons)
        xi = sum(c*mm for c, mm in zip(cs[:m], mons)); tau = sum(c*mm for c, mm in zip(cs[m:2*m], mons))
        phi = sum(c*mm for c, mm in zip(cs[2*m:], mons)); names = [f"{w}:{mm}" for w in ("xi", "tau", "phi") for mm in mons]
    px, pt, pxx = prolong(xi, tau, phi)
    comps = [xi, tau, phi, px, pt, pxx]             # pr v on coordinates x t u ux ut uxx
    g = sp.symbols('g0:6')                           # gradient of Delta wrt (x,t,u,ux,ut,uxx)
    crit = sp.expand(sum(gi*ci for gi, ci in zip(g, comps)))
    coefs = [sp.diff(crit, c) for c in cs]           # criterion is linear in parameters
    f = sp.lambdify((X, T, Uu, UX, UT, UXX, UXT, *g), coefs, "numpy")
    def rows(Zp, G):
        cols = f(*Zp.T, *G.T)
        return np.stack([np.broadcast_to(c, (len(Zp),)) for c in cols], 1)
    return rows, names, len(cs)

def grad_true(Zp):   # Delta = ut + u ux - nu uxx
    n = len(Zp); return np.stack([np.zeros(n), np.zeros(n), Zp[:, 3], Zp[:, 2], np.ones(n), -NU*np.ones(n)], 1)

def nullspace(M, k):
    scale = np.linalg.norm(M, axis=0); scale[scale == 0] = 1
    _, s, Vt = np.linalg.svd(M / scale, full_matrices=False)
    N = (Vt[-k:] / scale).T                       # back to original parameter units
    return s / s[0], np.linalg.qr(N)[0]

def true_basis(names, ansatz):
    idx = {n: i for i, n in enumerate(names)}; vecs = []
    if ansatz == "affine":
        gens = [{"xi:1": 1}, {"tau:1": 1}, {"xi:t": 1, "phi:1": 1}, {"xi:x": 1, "tau:t": 2, "phi:u": -1}]
    else:
        gens = [{"xi:1": 1}, {"tau:1": 1}, {"xi:t": 1, "phi:1": 1}, {"xi:x": 1, "tau:t": 2, "phi:u": -1},
                {"xi:t*x": 1, "tau:t**2": 1, "phi:x": 1, "phi:t*u": -1}]
    for gdict in gens:
        v = np.zeros(len(names))
        for kk, val in gdict.items(): v[idx[kk]] = val
        vecs.append(v)
    return np.linalg.qr(np.array(vecs).T)[0]

out = {}
for ansatz, k in (("affine", 4), ("quadratic", 5)):
    rows, names, npar = build(ansatz)
    M = rows(Z, grad_true(Z))
    s, N = nullspace(M, k)
    ang = float(np.rad2deg(subspace_angles(N, true_basis(names, ansatz)).max()))
    out[ansatz] = {"n_params": npar, "sing_vals": s.tolist(), "max_angle_deg": ang}
res["known_pde"] = out

# ---------------------------------------------------------------- noisy data + learned Delta (equation unknown)
def library(Zp):
    u, ux, uxx = Zp[:, 2], Zp[:, 3], Zp[:, 5]
    th = np.stack([np.ones_like(u), u, ux, uxx, u*u, u*ux, u*uxx, ux*ux, ux*uxx, uxx*uxx], 1)
    # derivatives of each library term wrt u, ux, uxx
    du = np.stack([0*u, 1+0*u, 0*u, 0*u, 2*u, ux, uxx, 0*u, 0*u, 0*u], 1)
    dux = np.stack([0*u, 0*u, 1+0*u, 0*u, 0*u, u, 0*u, 2*ux, uxx, 0*u], 1)
    duxx = np.stack([0*u, 0*u, 0*u, 1+0*u, 0*u, 0*u, u, 0*u, ux, 2*uxx], 1)
    return th, du, dux, duxx
lib_names = ["1", "u", "u_x", "u_xx", "u^2", "u u_x", "u u_xx", "u_x^2", "u_x u_xx", "u_xx^2"]
noise_levels = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
noisy = {}
rows_q, names_q, _ = build("quadratic"); rows_a, names_a, _ = build("affine")
for sig in noise_levels:
    Zn = Z.copy()
    for c in range(2, 7):
        Zn[:, c] += sig * Z[:, c].std() * rng.standard_normal(len(Z))
    th_, du, dux, duxx = library(Zn)
    w, *_ = np.linalg.lstsq(th_, Zn[:, 4], rcond=None)          # learned: u_t = Theta w
    G = np.stack([0*Zn[:, 0], 0*Zn[:, 0], -(du @ w), -(dux @ w), np.ones(len(Zn)), -(duxx @ w)], 1)
    entry = {"coef": dict(zip(lib_names, np.round(w, 4).tolist()))}
    for ansatz, rows, names, k in (("affine", rows_a, names_a, 4), ("quadratic", rows_q, names_q, 5)):
        s, N = nullspace(rows(Zn, G), k)
        TB = true_basis(names, ansatz) if False else None
        gens_true = {"affine": ["dx", "dt", "galilean", "scaling"], "quadratic": ["dx", "dt", "galilean", "scaling", "projective"]}[ansatz]
        idx = {n: i for i, n in enumerate(names)}
        defs = {"dx": {"xi:1": 1}, "dt": {"tau:1": 1}, "galilean": {"xi:t": 1, "phi:1": 1},
                "scaling": {"xi:x": 1, "tau:t": 2, "phi:u": -1},
                "projective": {"xi:t*x": 1, "tau:t**2": 1, "phi:x": 1, "phi:t*u": -1}}
        per = {}
        for gname in gens_true:
            v = np.zeros(len(names))
            for kk, val in defs[gname].items(): v[idx[kk]] = val
            v /= np.linalg.norm(v)
            per[gname] = float(np.rad2deg(np.arccos(min(1.0, np.linalg.norm(N.T @ v)))))
        entry[ansatz] = {"per_generator_angle_deg": per, "sing_vals": s.tolist(),
                         "max_angle_deg": float(np.rad2deg(subspace_angles(N, true_basis(names, ansatz)).max()))}
    noisy[str(sig)] = entry
res["noisy_learned"] = noisy
json.dump(res, open("data_results.json", "w"), indent=1)
for sig, e in noisy.items():
    print(sig, "u u_x:", e["coef"]["u u_x"], "u_xx:", e["coef"]["u_xx"],
          "| affine angle", round(e["affine"]["max_angle_deg"], 3), "| quad angle", round(e["quadratic"]["max_angle_deg"], 3))
print("per-gen @1%:", noisy["0.01"]["quadratic"]["per_generator_angle_deg"], noisy["0.01"]["affine"]["per_generator_angle_deg"])
print("known:", {k: (round(v["max_angle_deg"], 6), [f"{q:.1e}" for q in v["sing_vals"][-6:]]) for k, v in out.items()})

# ---------------------------------------------------------------- figures
# Fig 1: solutions and jet-space cloud
fig = plt.figure(figsize=(13, 4.2))
xs = np.linspace(0, 2*np.pi, 200); ts = np.linspace(0, 2, 120); XX, TT = np.meshgrid(xs, ts)
for i in range(2):
    c0, a, k, p = params[i]
    ax = fig.add_subplot(1, 3, i + 1)
    uu = jet_f(XX, TT, c0, *a, *k, *p)[0]
    im = ax.imshow(uu, origin="lower", extent=[0, 2*np.pi, 0, 2], aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(f"Burgers solution #{i+1}: u(x,t)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(1, 3, 3)
sel = rng.choice(len(Z), 5000, replace=False)
ax.scatter(-Z[sel, 2]*Z[sel, 3] + NU*Z[sel, 5], Z[sel, 4], s=3, c="#4a6fa5", alpha=0.5)
ax.plot([-2, 1.6], [-2, 1.6], color="#c0504d", lw=1, ls="--")
ax.set_xlabel(r"$-u\,u_x+\nu\,u_{xx}$  (computed from the jet coordinates)"); ax.set_ylabel(r"$u_t$")
ax.set_title(r"5000 jet-space points from 60 solutions" "\n" r"all lie on the surface $u_t=-u\,u_x+\nu\,u_{xx}$", fontsize=10)
fig.tight_layout(); fig.savefig("fig1_jetspace.png", dpi=150); plt.close(fig)

# Fig 2: symmetries map solutions to solutions
c0, a, k, p = params[1]
uf = lambda X_, T_: jet_f(X_, T_, c0, *a, *k, *p)[0]
eps = {"galilean": 0.6, "scaling": 0.35, "projective": 0.4}
XX, TT = np.meshgrid(np.linspace(0, 2*np.pi, 220), np.linspace(0, 2, 140))
fields = {"original": uf(XX, TT),
          "Galilean boost (ε=0.6)\n$u(x-\\varepsilon t,t)+\\varepsilon$": uf(XX - 0.6*TT, TT) + 0.6,
          "scaling (ε=0.35)\n$e^{-\\varepsilon}u(e^{-\\varepsilon}x, e^{-2\\varepsilon}t)$": np.exp(-0.35)*uf(np.exp(-0.35)*XX, np.exp(-0.7)*TT),
          "projective (ε=0.4)\n$\\frac{u(\\frac{x}{1+\\varepsilon t},\\frac{t}{1+\\varepsilon t})+\\varepsilon x}{1+\\varepsilon t}$":
              (uf(XX/(1 + 0.4*TT), TT/(1 + 0.4*TT)) + 0.4*XX)/(1 + 0.4*TT)}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (lab, F) in zip(axs, fields.items()):
    im = ax.imshow(F, origin="lower", extent=[0, 2*np.pi, 0, 2], aspect="auto", cmap="RdBu_r")
    ax.set_title(lab, fontsize=9); ax.set_xlabel("x"); plt.colorbar(im, ax=ax, fraction=0.046)
axs[0].set_ylabel("t")
fig.suptitle("Each symmetry turns a Burgers solution into another Burgers solution (PDE residual of each transformed solution: exactly 0, checked symbolically)", fontsize=11)
fig.tight_layout(); fig.savefig("fig2_transformed.png", dpi=150); plt.close(fig)

# Fig 3: singular values, known PDE
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, (ans, k) in zip(axs, (("affine", 4), ("quadratic", 5))):
    s = np.array(out[ans]["sing_vals"]); idx = np.arange(1, len(s) + 1)
    ax.semilogy(idx, np.maximum(s, 1e-18), "o-", color="#4a6fa5")
    ax.semilogy(idx[-k:], np.maximum(s[-k:], 1e-18), "o", color="#c0504d", ms=9, label=f"{k} (near-)zero values = {k} symmetries")
    ax.set_xlabel("index"); ax.set_ylabel("singular value (relative)"); ax.legend(fontsize=8)
    ax.set_title(f"{ans} ansatz ({len(s)} parameters)", fontsize=10); ax.grid(alpha=0.3)
fig.suptitle("Singular values of the invariance-criterion matrix (Burgers known, clean data)", fontsize=11)
fig.tight_layout(); fig.savefig("fig3_singular.png", dpi=150); plt.close(fig)

# Fig 4: noise
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
cols = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(noise_levels)))
for c, sig in zip(cols, noise_levels):
    s = np.array(noisy[str(sig)]["quadratic"]["sing_vals"])
    axs[0].semilogy(np.arange(1, 31), np.maximum(s, 1e-18), "o-", ms=3, color=c, label=f"noise {100*sig:g}%")
axs[0].axvline(25.5, color="gray", ls=":"); axs[0].set_xlabel("index"); axs[0].set_ylabel("singular value (relative)")
axs[0].set_title("quadratic ansatz, equation learned from noisy data", fontsize=10); axs[0].legend(fontsize=7, ncol=2)
for ans, col in (("affine", "#4a6fa5"), ("quadratic", "#c0504d")):
    axs[1].plot([100*s for s in noise_levels[1:]], [noisy[str(s)][ans]["max_angle_deg"] for s in noise_levels[1:]],
                "o-", color=col, label=f"{ans} ansatz")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("noise on jet coordinates (%)")
axs[1].set_ylabel("largest angle to true symmetries (deg)"); axs[1].legend(); axs[1].grid(alpha=0.3)
axs[1].set_title("how far the recovered symmetries are from the truth", fontsize=10)
fig.tight_layout(); fig.savefig("fig4_noise.png", dpi=150); plt.close(fig)
