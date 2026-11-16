from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CASES  = ["g0", "cubic"]
N_OBS  = [50, 200, 1000]
SIGMAS = [0.0, 0.01, 0.03]


def sample_experiment(x, t, Uk, n_obs, sigma, rng):
    """Uk: one experiment, shape (n_t, N). Return x_obs, t_obs, u_clean, u_obs (each length n_obs)."""
    n_t, N = Uk.shape
    # S1: pick n_obs DIFFERENT grid points. Draw flat indices in [0, n_t*N) without replacement,
    #     then turn each flat index into (time index, space index).
    flat   = rng.choice(n_t*N, size=n_obs, replace=False)
    it, ix = np.divmod(flat, N)                          # hint: divmod, or np.unravel_index(flat, Uk.shape)

    u_clean = Uk[it, ix]                               # true values at those points
    # S2: noise, width = sigma * std of the WHOLE experiment (not just the sampled points)
    u_obs = u_clean + rng.normal(0.0, sigma*np.std(Uk), size=n_obs)
    return x[ix], t[it], u_clean, u_obs


if __name__ == "__main__":
    for name in CASES:
        d = np.load(DATA / f"dense_{name}.npz")
        x, t, U, nu = d["x"], d["t"], d["U"], d["nu"]
        K = U.shape[0]

        for n_obs in N_OBS:
            for sigma in SIGMAS:
                seed = n_obs + round(100*sigma)                      # S3: a fixed seed per setting (see hint)
                rng  = np.random.default_rng(seed)

                parts = [sample_experiment(x, t, U[k], n_obs, sigma, rng) for k in range(K)]
                # S4: join the K experiments into single arrays, and record which experiment each point is from
                x_obs   = np.concatenate([p[0] for p in parts])
                t_obs   = np.concatenate([p[1] for p in parts])
                u_clean = np.concatenate([p[2] for p in parts])
                u_obs   = np.concatenate([p[3] for p in parts])
                exp_id  = np.repeat(np.arange(K), [len(p[0]) for p in parts])

                out = DATA / f"sparse_{name}_N{n_obs}_s{100*sigma:g}.npz"
                np.savez(out, x_obs=x_obs, t_obs=t_obs, u_obs=u_obs, u_clean=u_clean,
                         exp_id=exp_id, nu=nu, g=name, N_obs=n_obs, noise=sigma, seed=seed)
                print(f"{out.name}: {len(u_obs)} points, "
                      f"rms noise / std(u) = {np.std(u_obs - u_clean) / np.std(U):.3f}")
    d = np.load(DATA / "dense_g0.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    for ax, n_obs in zip(axes, N_OBS):
        s = np.load(DATA / f"sparse_g0_N{n_obs}_s1.npz")
        m = s["exp_id"] == 0                                   # experiment 0 only
        pc = ax.pcolormesh(d["x"], d["t"], d["U"][0], shading="auto", cmap="RdBu_r")
        ax.scatter(s["x_obs"][m], s["t_obs"][m], s=8, c="k")
        ax.set_title(f"N_obs = {n_obs}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(pc, ax=axes, label="u")
    fig.savefig(RESULTS / "sparse_observations.png", dpi=150)
    plt.show()