"""Paper-style figure of the symmetry loop (design notes/symmetry_loop_design.md, section 9).

    python notes/figures/make_loop_figure.py      -> notes/figures/symmetry_loop.png and .pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})

C = {  # (fill, edge) per region
    "input": ("#FDF3D7", "#D9B44A"),
    "base": ("#E8F0FA", "#5F8CC2"),
    "svd": ("#E1F5EE", "#3FA27F"),
    "test": ("#EEEDFE", "#7F77DD"),
    "comb": ("#FBEAF0", "#C0607F"),
    "out": ("#E6F2DD", "#6E9F45"),
    "param": ("#F3F2EE", "#9A9890"),
}
DARK = "#333333"

fig, ax = plt.subplots(figsize=(19, 9.6))
ax.set_xlim(0, 37); ax.set_ylim(-0.7, 17.6); ax.axis("off")


def region(x, y, w, h, kind, label):
    f, e = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.45", fc=f, ec=e, lw=1.5, zorder=0))
    ax.text(x + 0.3, y + h - 0.3, label, ha="left", va="top", fontsize=13, fontweight="bold", color=e, zorder=1)


def box(x, y, w, h, title, lines=(), size=11.5, ec=DARK, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.1, zorder=2))
    rows = [(title, True)] + [(s, False) for s in lines]
    n = len(rows)
    for i, (s, bold) in enumerate(rows):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * 0.33 - i * 0.66, s, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", zorder=3)


def path(pts, color=DARK, ls="-"):
    """Polyline with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=1.4, ls=ls, zorder=4)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, lw=1.4, color=color,
                                 ls=ls, zorder=4, shrinkA=0, shrinkB=1))


def label(x, y, s, color=DARK, size=11, ha="left", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", style=style, zorder=5)


# ======================= top row: input -> baseline -> propose =======================
region(0.4, 9.6, 6.6, 7.6, "input", "Input")
box(0.9, 13.3, 5.6, 2.4, "observations", [r"$\mathcal{D}=\{(x_i,t_i,u_i)\}_{i=1}^{N}$"])
box(0.9, 10.2, 5.6, 2.4, "fixed split", [r"train:  $t \leq 1.5$", r"held out:  $t > 1.5$"])
path([(3.7, 13.3), (3.7, 12.6)])

region(8.0, 9.6, 7.6, 7.6, "base", "1. Baseline")
box(8.5, 13.0, 6.6, 2.8, "PINN, learned equation",
    [r"$u_t + u\,u_x = \hat\nu\,u_{xx} + \hat g(u)$", r"$\mathcal{L}_{data} + \lambda\,\mathcal{L}_{PDE}$"])
box(8.5, 10.1, 6.6, 2.4, "baseline  $B$", ["2 seeds, keep the better", r"held-out error  $E_B$"], size=11)
path([(11.8, 13.0), (11.8, 12.5)])
path([(7.0, 14.5), (8.5, 14.5)])

region(16.6, 9.6, 8.2, 7.6, "svd", "2. Propose candidates")
box(17.1, 13.1, 7.2, 2.6, "symmetry condition",
    [r"$A(\hat u^{(3)};\,\hat g,\hat g',\hat\nu)\; v = 0$", r"30 basis fields $(\xi,\tau,\phi)$"], size=11)
box(17.1, 10.2, 7.2, 2.0, "candidates  $v_1, v_2, \\ldots$", [r"drop $\partial_x,\partial_t$ and accepted"], size=11)
path([(20.7, 13.1), (20.7, 12.2)])
label(20.9, 12.65, "SVD", size=10.5)
path([(15.6, 14.5), (17.1, 14.5)])

# ======================= right column: test every candidate alone =======================
region(25.8, 0.4, 10.8, 16.8, "test", "3. Test each candidate alone")
box(26.3, 13.0, 9.3, 2.7, "train  $B$ + one candidate $v_j$",
    [r"$\mathcal{L}_{data} + \lambda\mathcal{L}_{PDE} + \mu\,\mathcal{L}_{sym}(v_j)$",
     r"$\mathcal{L}_{sym}(v) = \mathrm{mean}\,(A_n v)^2$"], size=11)
box(26.3, 10.4, 9.3, 1.8, "held-out test", [r"$R_j = E_{v_j}\,/\,E_B$"], size=11)
path([(30.95, 13.0), (30.95, 12.2)])
path([(24.3, 11.2), (25.3, 11.2), (25.3, 14.35), (26.3, 14.35)])

dx, dy = 29.0, 8.0
ax.add_patch(Polygon([(dx - 2.4, dy), (dx, dy + 1.4), (dx + 2.4, dy), (dx, dy - 1.4)], closed=True,
                     fc="white", ec=DARK, lw=1.1, zorder=2))
ax.text(dx, dy, r"$R_j \leq 1+\delta$ ?", ha="center", va="center", fontsize=12.5, zorder=3)
path([(dx, 10.4), (dx, dy + 1.4)])
box(32.6, 8.7, 3.0, 1.5, "accept", ["keep $v_j$"], size=11, ec="#3B6D11", fc="#F2F8EC")
box(32.6, 5.8, 3.0, 1.5, "reject", ["discard"], size=11, ec="#A32D2D", fc="#FCEFEF")
path([(dx + 2.4, dy), (31.9, dy), (31.9, 9.45), (32.6, 9.45)], color="#3B6D11")
path([(31.9, dy), (31.9, 6.55), (32.6, 6.55)], color="#A32D2D")
label(31.45, 8.85, "yes", color="#3B6D11", size=10.5, ha="right")
label(31.45, 7.15, "no", color="#A32D2D", size=10.5, ha="right")
# next candidate loop (right edge)
path([(35.6, 9.45), (36.15, 9.45), (36.15, 14.35), (35.6, 14.35)], color="#5A52B0", ls="--")
ax.plot([35.6, 36.15, 36.15], [6.55, 6.55, 9.45], color="#5A52B0", lw=1.4, ls="--", zorder=4)
label(26.5, 5.2, "repeat with the next candidate (dashed)", color="#5A52B0", size=10.5, style="italic")
label(26.5, 4.6, "until 2 rejections or 5 candidates", color="#5A52B0", size=10.5, style="italic")

box(26.3, 1.0, 9.3, 2.6, "end of pass", ["anything new accepted?"], size=11.5)
path([(34.1, 5.8), (34.1, 3.6)])

# ======================= bottom row: combined training, parameters, output =======================
region(16.6, 0.4, 8.2, 8.2, "comb", "4. Combined training")
box(17.1, 1.0, 7.2, 3.1, "train  $B$ + all accepted",
    [r"$\mu\sum_{v\in\mathrm{acc.}}\mathcal{L}_{sym}(v)$", r"new model $M$: $\hat u,\hat g,\hat\nu$"], size=11)
box(17.1, 4.9, 7.2, 1.9, "check together", [r"$E_M / E_B \leq 1+\delta$"], size=11)
path([(26.3, 2.3), (24.3, 2.3)])
label(25.3, 2.75, "yes", color="#3B6D11", size=10.5, ha="center")
path([(20.7, 4.1), (20.7, 4.9)])
path([(23.5, 6.8), (23.5, 10.2)], color="#993556", ls="--")
label(23.3, 9.1, "next pass:  SVD of the new $M$", color="#993556", size=10.5, ha="right")

region(8.0, 0.4, 7.6, 8.2, "param", "Parameters (fixed)")
for i, s in enumerate([r"$\delta = 0.1$,   $\mu = 1$", "baseline seeds: 2", "max_rejections: 2",
                       "max_candidates: 5", "max_passes: 2", "all trainings from scratch,", "same seed and split"]):
    label(8.6, 6.9 - i * 0.82, s, size=11.5)

region(0.4, 0.4, 6.6, 8.2, "out", "Output")
box(0.9, 4.2, 5.6, 2.9, "learned equation", [r"$\hat g(u),\ \hat\nu$ of the last $M$"], size=11)
box(0.9, 1.0, 5.6, 2.6, "found symmetries", [r"accepted $\{v_i\}_{i=1}^{d}$"], size=11)
path([(30.95, 1.0), (30.95, -0.25), (3.7, -0.25), (3.7, 0.4)], color="#6E9F45")
label(17.0, -0.5, "no:  stop", color="#6E9F45", size=10.5)

fig.savefig(OUT / "symmetry_loop.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "symmetry_loop.pdf", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "symmetry_loop.png")
