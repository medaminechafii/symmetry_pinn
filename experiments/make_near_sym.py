import numpy as np
from pathlib import Path

# ---- copied unchanged from tests/test_cole_hopf.py ----
x = np.linspace(0, 2*np.pi, 256, endpoint=False)
nu, N, T = 0.1, 256, 2.0
k = np.fft.fftfreq(N, 2*np.pi/N)*2*np.pi

def dx(u):
    return np.fft.ifft(1j*k*np.fft.fft(u)).real
def d_xx(u):
    return np.fft.ifft(-k**2*np.fft.fft(u)).real

def rhs_burgers(u, g):
    return -u*dx(u) + nu*d_xx(u) + g(u)

def rk4_step(u, dt, g):
    k1 = rhs_burgers(u, g)
    k2 = rhs_burgers(u + 0.5*dt*k1, g)
    k3 = rhs_burgers(u + 0.5*dt*k2, g)
    k4 = rhs_burgers(u + dt*k3, g)
    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# time step for stability of RK4
lambda_rk = -nu*(N/2)**2
dt_max = 2.785/np.abs(lambda_rk)
steps = int(np.ceil(T/(dt_max*0.5)))
dt = T/steps
t = np.linspace(0, T, steps+1)

save_every = 10
def solve_burgers_saved(u0, steps, g):
    dt = T/steps
    u = u0.copy()
    U = [u.copy()]
    for n in range(1, steps+1):
        u = rk4_step(u, dt, g)
        if n % save_every == 0:
            U.append(u.copy())
    return np.array(U)

def random_ic(rng):
    A = rng.uniform(-0.5, 0.5, size=3)
    phi = rng.uniform(0, 2*np.pi, size=3)
    u0 = np.zeros_like(x)
    for j in range(3):
        u0 += A[j]*np.sin((j+1)*x + phi[j])
    return u0
# ---- end of the copied block ----

from discovery.noise_gap import EPS
from experiments.make_sparse import sample_experiment, N_OBS, SIGMAS

DATA = Path(__file__).resolve().parent.parent / "data"

def g_eps(eps):
    return lambda u: -u**3 + eps*u

def make_dense(name, eps):
    g = g_eps(eps)
    U_all = []
    for i in range(3):
        rng = np.random.default_rng(i)               # same seeds as the original data
        u0 = random_ic(rng)
        U_all.append(solve_burgers_saved(u0, steps, g))
    np.savez(DATA / f"dense_{name}.npz", x=x, t=t[::save_every], U=U_all, nu=nu, seeds=np.arange(3))

def make_sparse(name, n_list=(1000,)):
    d = np.load(DATA / f"dense_{name}.npz")
    xg, tg, U, nu_ = d["x"], d["t"], d["U"], d["nu"]
    K = U.shape[0]
    for n_obs in n_list:
        for sigma in SIGMAS:
            seed = n_obs + round(100*sigma)          # same rule as make_sparse.py
            rng = np.random.default_rng(seed)
            parts = [sample_experiment(xg, tg, U[kk], n_obs, sigma, rng) for kk in range(K)]
            x_obs   = np.concatenate([p[0] for p in parts])
            t_obs   = np.concatenate([p[1] for p in parts])
            u_clean = np.concatenate([p[2] for p in parts])
            u_obs   = np.concatenate([p[3] for p in parts])
            exp_id  = np.repeat(np.arange(K), [len(p[0]) for p in parts])
            out = DATA / f"sparse_{name}_N{n_obs}_s{round(100*sigma)}.npz"
            np.savez(out, x_obs=x_obs, t_obs=t_obs, u_obs=u_obs, u_clean=u_clean,
                     exp_id=exp_id, nu=nu_, g=name, N_obs=n_obs, noise=sigma, seed=seed)
            print(f"{out.name}: {len(u_obs)} points, "
                  f"rms noise / std(u) = {np.std(u_obs - u_clean) / np.std(U):.3f}")

if __name__ == "__main__":
    # python -m experiments.make_near_sym                              # all eps, N = 1000 (as before)
    # python -m experiments.make_near_sym --eps e003 e01 --n 200 1000  # false-positive control data
    # Same seeds as make_sparse.py: same observation points and noise draws as the cubic files.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", choices=list(EPS), default=list(EPS))
    ap.add_argument("--n", nargs="+", type=int, default=[1000])
    args = ap.parse_args()
    for tag in args.eps:
        make_dense(f"cubic_{tag}", EPS[tag])
        make_sparse(f"cubic_{tag}", args.n)
        