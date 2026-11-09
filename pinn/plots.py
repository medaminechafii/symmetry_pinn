import numpy as np
import matplotlib.pyplot as plt

KEYS = ["u", "u_x", "u_t", "u_xx"]


def plot_training(history, adam_steps, noise_var=None, path=None):
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    ax.semilogy(history, lw=1.0, color="C0")
    ax.axvline(adam_steps, color="C3", ls=":", lw=1.5)
    ax.text(adam_steps, history[0], " L-BFGS", color="C3", va="top", fontsize=9)
    if noise_var is not None:
        ax.axhline(noise_var, color="C2", ls="--", lw=1.3)
        ax.text(0, noise_var, " noise variance", color="C2", va="bottom", fontsize=9)
    ax.set_xlabel("loss evaluation")
    ax.set_ylabel("MSE on the observations")
    ax.set_title("training")
    ax.grid(alpha=0.3, which="both")
    if path: fig.savefig(path, dpi=150)
    return fig


def plot_snapshots(x, t, truth, pred, ti=None, path=None):
    """Surrogate vs truth for all four quantities, at one time."""
    if ti is None: ti = len(t) // 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for ax, key in zip(axes.ravel(), KEYS):
        a, b = truth[key][ti], pred[key][ti]
        ax.plot(x, a, "k--", lw=1.4, label="truth")
        ax.plot(x, b, lw=1.8, color="C0", label="surrogate")
        e = 100 * np.linalg.norm(b - a) / np.linalg.norm(a)
        ax.set_title(f"{key}    error here: {e:.1f} %")
        ax.set_xlabel("x"); ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"t = {t[ti]:.2f}")
    if path: fig.savefig(path, dpi=150)
    return fig


def plot_error_maps(x, t, truth, pred, x_obs=None, t_obs=None, path=None):
    """Where in the domain each quantity goes wrong, with the observations on top."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True)
    for ax, key in zip(axes.ravel(), KEYS):
        err = np.abs(pred[key] - truth[key])
        pc = ax.pcolormesh(x, t, err, shading="auto", cmap="magma")
        fig.colorbar(pc, ax=ax)
        if x_obs is not None:
            ax.plot(x_obs, t_obs, ".", ms=3.5, color="cyan")
        ax.set_title(f"|error| in {key}")
        ax.set_xlabel("x"); ax.set_ylabel("t")
    if path: fig.savefig(path, dpi=150)
    return fig

def plot_training_oracle(history, adam_steps, lam, noise_var=None, path=None):
    """history = [(total, data, pde), ...] from pinn_oracle.train_pinn."""
    h = np.array(history)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    ax.semilogy(h[:, 1], lw=1.0, color="C0", label="data MSE")
    ax.semilogy(h[:, 2], lw=1.0, color="C1", label="PDE residual MSE")
    ax.semilogy(h[:, 0], lw=1.0, color="k", alpha=0.6, label=f"total (lam={lam:g})")
    ax.axvline(adam_steps, color="C3", ls=":", lw=1.5)
    ax.text(adam_steps, h[0, 1], " L-BFGS", color="C3", va="top", fontsize=9)
    if noise_var is not None:
        ax.axhline(noise_var, color="C2", ls="--", lw=1.3)
        ax.text(0, noise_var, " noise variance", color="C2", va="bottom", fontsize=9)
    ax.set_xlabel("loss evaluation"); ax.set_ylabel("loss")
    ax.set_title("oracle PINN training")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    if path: fig.savefig(path, dpi=150)
    return fig
