import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

cases = {"g0": "g = 0", "cubic": "g = -u³"}
data  = {name: np.load(DATA / f"dense_{name}.npz") for name in cases}
x, t  = data["g0"]["x"], data["g0"]["t"]
K     = data["g0"]["U"].shape[0]
print("t ends at", t[-1], "| shapes:", {n: d["U"].shape for n, d in data.items()})

# 1 --- heat maps: one row per g, one column per experiment ---
vmax = max(np.abs(d["U"]).max() for d in data.values())   # same colour scale everywhere
fig, axes = plt.subplots(len(cases), K, figsize=(4*K, 3.4*len(cases)),
                         sharex=True, sharey=True, constrained_layout=True)
for r, (name, label) in enumerate(cases.items()):
    U = data[name]["U"]
    for k in range(K):
        ax = axes[r, k]
        pc = ax.pcolormesh(x, t, U[k], shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label}, experiment {k}")
        if r == len(cases) - 1: ax.set_xlabel("x")
        if k == 0:              ax.set_ylabel("t")
fig.colorbar(pc, ax=axes, label="u", shrink=0.8)
fig.savefig(RESULTS / "dense_heatmaps.png", dpi=150)

# 2 --- resolution check: Fourier spectrum at the final time ---
fig, axes = plt.subplots(1, len(cases), figsize=(11, 4), sharey=True, constrained_layout=True)
kk = np.arange(len(x) // 2)                           # wavenumbers 0 .. N/2-1
for ax, (name, label) in zip(axes, cases.items()):
    U = data[name]["U"]
    for k in range(K):
        spec = np.abs(np.fft.fft(U[k, -1])) / len(x)  # amplitude of each Fourier mode
        ax.semilogy(kk, spec[:len(kk)] + 1e-20, label=f"experiment {k}")
    ax.set_title(f"Spectrum at t = {t[-1]:.2f}, {label}")
    ax.set_xlabel("wavenumber k")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("|û_k|")
fig.savefig(RESULTS / "dense_spectrum.png", dpi=150)

# 3 --- decay: max|u| over time, both g on the same axes ---
fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
styles = {"g0": "-", "cubic": "--"}
for name, label in cases.items():
    U = data[name]["U"]
    for k in range(K):
        ax.plot(t, np.abs(U[k]).max(axis=1), styles[name], color=f"C{k}",
                label=f"{label}, exp {k}")
ax.set_xlabel("t")
ax.set_ylabel("max |u|")
ax.set_title("Decay of the solution")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.savefig(RESULTS / "dense_decay.png", dpi=150)

plt.show()