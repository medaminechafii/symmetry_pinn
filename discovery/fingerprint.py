"""Week 3 Part C, steps 3-4: symmetry fingerprint on dense clean data.

Builds the matrix A (points x 30 basis fields), runs the SVD, counts the symmetries andchecks the known generators. 
Run from the repo root:
    python -m discovery.fingerprint
Needs: data/dense_g0.npz and data/dense_cubic.npz (from your data generator),
       discovery/criterion.py (all its self-checks passing), discovery/__init__.py (empty file).
"""
from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from discovery.criterion import columns, basis_fields, JET, u, MONO_NAMES

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CASES = {"g0": sp.Integer(0), "cubic": -u**3}     # file name -> g(u) used in the criterion
N_POINTS = 3000
SEED = 0


# --- step 3a: spectral derivatives of all rows at once ------------------------------
def spectral_derivative(U, k, order):
    """d^order/dx^order of every row of U (FFT along the last axis = x).

    F1: same idea as your dx, but for any order: multiply by (i k)^order in Fourier space.
        Use axis=-1 in both np.fft.fft and np.fft.ifft so it works on 2D and 3D arrays,
        and keep the real part."""
    f_hat = np.fft.fft(U, axis=-1)
    df_hat = (1j * k) ** order * f_hat
    return np.fft.ifft(df_hat, axis = -1).real                                  


# --- step 3b: jet values at random points --------------------------------------------
def sample_jet(d, n_points, rng):
    """Return a dict of 1D arrays: x, t, u, u_x, u_xx, u_xxx at n_points random points."""
    x, t, U = d["x"], d["t"], d["U"]                  # U: (K, n_t, N)
    K, n_t, N = U.shape
    k = np.fft.fftfreq(N, d=x[1] - x[0]) * 2 * np.pi  # integer wavenumbers, as in your solver
    Ux, Uxx, Uxxx = (spectral_derivative(U, k, m) for m in (1, 2, 3))

    # F2: draw n_points random integer indices for experiment, time and space.
    #     Hint: rng.integers(low, high, size) gives integers in [low, high).
    e  = rng.integers(0, K, size=n_points)          # experiment index, in [0, K)
    it = rng.integers(0, n_t, size=n_points)        # time index, in [0, n_t)
    ix = rng.integers(0, N, size=n_points)          # space index, in [0, N)

    # F3: pick the values at those points. U[e, it, ix] picks one value per point
    #     (NumPy "fancy indexing": the three index arrays are read in parallel).
    return {"x": x[ix], "t": t[it], "u": U[e, it, ix],
            "u_x": Ux[e,it,ix], "u_xx": Uxx[e,it,ix], "u_xxx": Uxxx[e,it,ix]}    


# --- step 3c: evaluate the symbolic columns -------------------------------------------
def evaluate(expr, jet):
    """F4: lambdify expr in the jet variables JET and evaluate it at the sampled points.
    Constant columns (the two zero columns!) need the broadcast_to fix from toy lesson 2."""
    fun = sp.lambdify(JET, expr, "numpy")
    val = fun(jet["x"], jet["t"], jet["u"], jet["u_x"], jet["u_xx"], jet["u_xxx"])
    return np.broadcast_to(np.asarray(val,dtype=float),jet["x"].shape).copy()   


def build_matrix(cols, jet):
    return np.column_stack([evaluate(c, jet) for c in cols])


# --- step 4: normalise, SVD, count -----------------------------------------------------
def normalise(A):
    """Divide each column by its norm. Zero columns are left at zero: their 'norm' is
    replaced by 1, so there is no division by zero and they simply give zero singular
    values (a simpler alternative to removing them, with the same result).

    F5: compute the column norms, detect zero columns with a RELATIVE test
        (norm < 1e-12 * largest norm), build `scale` = norms with 1 in the zero slots
        (np.where is useful), and divide."""
    norms   = np.linalg.norm(A, axis=0)                                 
    is_zero = norms < 1e-12 * np.max(norms)                                 
    scale   = np.where(is_zero, 1, norms)                                 
    return A / scale, scale, is_zero


def count_by_gap(s):
    """Number of near-zero singular values = position of the largest jump (in log10).

    Given (no blank), but read it: exact zeros from zero columns would make log10 = -inf,
    and a jump from round-off (1e-16) to an exact 0 would look 'infinitely large'. So all
    values are first floored at 1e-18 * s[0]."""
    floor = s[0] * 1e-18
    logs = np.log10(np.maximum(s, floor))
    jumps = logs[:-1] - logs[1:]
    j = int(np.argmax(jumps))
    return len(s) - (j + 1), jumps[j]


# --- known generators as 30-vectors ----------------------------------------------------
def vec(xi=None, tau=None, phi=None):
    """Coefficient vector from dicts {monomial name: coefficient} for each component.
    Example: the Galilean boost t d/dx + d/du is vec(xi={"t": 1}, phi={"1": 1})."""
    v = np.zeros(30)
    for comp, terms in enumerate([xi, tau, phi]):
        for name, c in (terms or {}).items():
            v[10 * comp + MONO_NAMES.index(name)] = c
    return v


# F6: write the other three generators with vec(...). Monomial names: see MONO_NAMES.
KNOWN = {
    "d/dx":       vec(xi={"1": 1}),
    "d/dt":       vec(tau={"1": 1}),
    "Galilean":   vec(xi={"t": 1}, phi={"1": 1}),                            #   t d/dx + d/du
    "scaling":    vec(xi={"x": 1}, tau={"t": 2}, phi={"u": -1}),             #   x d/dx + 2t d/dt - u d/du
    "projective": vec(xi={"xt": 1}, tau={"t^2": 1}, phi={"x": 1,"tu": -1}),   #   xt d/dx + t^2 d/dt + (x - tu) d/du
}


def projection_residual(w, Vt, n_null, scale):
    """Relative distance of the generator w (ORIGINAL coefficients) from the found null space.

    F7: the null rows of Vt live in the coordinates of the NORMALISED matrix. Since
        A c = A_n (scale * c), first convert w to those coordinates, then project onto the
        last n_null rows of Vt as in toy lesson 2, Blank 6."""
    w_n = w * scale                                     
    N   = Vt[-n_null:]                                  
    return np.linalg.norm(w_n - N.T @ (N @ w_n)) / np.linalg.norm(w_n)


# --- main ------------------------------------------------------------------------------
if __name__ == "__main__":
    _, names = basis_fields()
    fig, axes = plt.subplots(1, len(CASES), figsize=(11, 4), sharey=True, constrained_layout=True)

    for ax, (name, g) in zip(axes, CASES.items()):
        d = np.load(DATA / f"dense_{name}.npz")
        jet = sample_jet(d, N_POINTS, np.random.default_rng(SEED))

        # F8: build A from columns(g), normalise it, take the SVD (full_matrices=False),
        #     and count with count_by_gap.
        A = build_matrix(columns(g),jet)                       
        A_n, scale, is_zero = normalise(A)                     
        _, s, Vt = np.linalg.svd(A_n, full_matrices=False)     
        n_null, jump = count_by_gap(s)                         

        print(f"\n=== {name}:  A shape {A.shape},  zero columns: {[names[i] for i in np.where(is_zero)[0]]}")
        print("smallest 8 singular values:", np.array2string(s[-8:], precision=2))
        print(f"count = {n_null}   (largest jump = {jump:.1f} orders of magnitude)")
        for gname, w in KNOWN.items():
            r = projection_residual(w, Vt, n_null, scale)
            print(f"   {gname:11s} residual = {r:.1e}")

        # plot (given)
        s_plot = np.maximum(s, s[0] * 1e-18)
        idx = np.arange(1, len(s) + 1)
        ax.semilogy(idx, s_plot, "o-", ms=4)
        ax.semilogy(idx[-n_null:], s_plot[-n_null:], "o", ms=8, color="C3",
                    label=f"{n_null} near-zero values")
        ax.axvline(len(s) - n_null + 0.5, ls=":", color="gray")
        ax.set_title(f"{name}: g = {sp.sstr(g)}")
        ax.set_xlabel("index")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("singular value (normalised columns)")
    fig.savefig(RESULTS / "fingerprint_singular_values.png", dpi=150)
    print("\nsaved", RESULTS / "fingerprint_singular_values.png")
    plt.show()
