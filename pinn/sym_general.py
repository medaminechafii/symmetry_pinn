"""General symmetry loss for ANY generator vector (design notes/symmetry_loop_design.md, section 2.4).

No new math: it is the same condition the SVD tests. The 30 substituted columns (discovery.criterion, evaluated as in
discovery.stage0b_general) form a matrix A with one row per point; a generator with original coefficients w
(30 numbers: xi, tau, phi on 1, x, t, u, x^2, t^2, u^2, xt, xu, tu) is a symmetry of the learned equation exactly
when A w = 0 at every point.  The loss is

    L_sym(w) = mean over points of ( sum_i v_i * A_i / scale_i )^2,      v = w * scale / |w * scale|

scale_i = RMS of column i, FIXED (computed once from the baseline model, like the SVD normalisation), so v means the
same in the loss as in the SVD: if v is a right singular vector with value s (SVD on the same points), L_sym = s^2 / n.

Jets (u, u_x, u_xx, u_xxx of the network) are detached: the gradient reaches only g_hat and nu_hat, as in the mu pilot.

    python -m pinn.sym_general --test       # data-free tests (exact cubic / g0 jets)
"""
import numpy as np, torch
from discovery.stage0b_general import FUNS
from discovery.fingerprint import KNOWN


def d(f, z):
    return torch.autograd.grad(f, z, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def network_jets(model, x, t):
    """x, t, u, u_x, u_xx, u_xxx of the network at (x, t), detached.  x, t need requires_grad."""
    u = model(x, t)
    u_x = d(u, x); u_xx = d(u_x, x); u_xxx = d(u_xx, x)
    return [z.detach() for z in (x, t, u, u_x, u_xx, u_xxx)]


def columns(jets, g_model, nu):
    """(n, 30) tensor of the substituted columns. Gradient flows to g_model's parameters and to nu (if a tensor)."""
    u = jets[2].clone().requires_grad_(True)
    g = g_model(u)
    g1 = torch.autograd.grad(g, u, grad_outputs=torch.ones_like(g), create_graph=True)[0]
    args = list(jets[:2]) + [jets[2]] + list(jets[3:]) + [g, g1, nu]
    ones = torch.ones_like(jets[0])
    return torch.stack([f(*args) * ones for f in FUNS], dim=1)       # "* ones": constant columns -> full length


def rms_scale(A):
    """RMS of each column; zero columns (e.g. d/dx, d/dt) get 1, as in fingerprint.normalise."""
    s = np.sqrt(np.mean(np.asarray(A, dtype=float) ** 2, axis=0))
    return np.where(s < 1e-12 * s.max(), 1.0, s)


def unit(W, scale):
    """Generator vectors in original coefficients (m, 30) -> unit vectors in normalised coordinates."""
    V = np.atleast_2d(W) * scale
    return V / np.linalg.norm(V, axis=1, keepdims=True)


class LearnableGen(torch.nn.Module):
    """Learnable candidate generators (loop option --learn): the components that are non-zero in the starting vectors
    (the support, fixed by cleaning / sparsification) are trainable, all others stay exactly 0; every row is
    renormalised to length 1 at each call, so the loss cannot shrink v to 0."""
    def __init__(self, V0):
        super().__init__()
        V0 = torch.as_tensor(np.atleast_2d(V0), dtype=torch.float64)
        self.register_buffer("mask", (V0 != 0).double())
        self.P = torch.nn.Parameter(V0.clone())

    def forward(self):
        V = self.P * self.mask
        return V / V.norm(dim=1, keepdim=True)


def sym_loss_general(jets, g_model, nu, V, scale):
    """mean over the m vectors of mean((A_n v)^2).  V: (m, 30) unit vectors in normalised coordinates."""
    A_n = columns(jets, g_model, nu) / torch.as_tensor(scale)
    R = A_n @ torch.as_tensor(V).T                                     # (n, m)
    return torch.mean(R ** 2)


if __name__ == "__main__":
    import argparse
    from pinn.learned_g import PolyG
    from discovery.stage0b_general import points, sample_indices
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.parse_args()
    it, ix = sample_indices()
    ok = lambda b: "OK " if b else "FAIL"
    for name, c in (("cubic", [0, 0, -1, 0, 0]), ("g0", [0, 0, 0, 0, 0])):
        jd = points(name, 0, it, ix)
        jets = [torch.tensor(jd[k]) for k in ("x", "t", "u", "u_x", "u_xx", "u_xxx")]
        g = PolyG(); g.c.data = torch.tensor(c, dtype=torch.float64)
        nu = torch.tensor(0.1, dtype=torch.float64)
        A = columns(jets, g, nu).detach().numpy()
        scale = rms_scale(A)
        print(f"\n{name}: exact jets, exact g, nu = 0.1")
        for gn in ("d/dx", "d/dt", "scaling", "Galilean", "projective"):
            L = sym_loss_general(jets, g, nu, unit(KNOWN[gn], scale), scale).item()
            true = gn in ("d/dx", "d/dt", "scaling") or name == "g0"
            print(f"  {gn:10s} L_sym = {L:.2e}   expected {'~0' if true else '> 0'}  {ok((L < 1e-20) == true)}")

        # proportionality to the hand-written residuals (fixes the conversion of mu), with a perturbed g
        g.c.data = torch.tensor([0.05, -0.1, -0.9, 0.08, -0.04], dtype=torch.float64)
        u = jets[2].clone().requires_grad_(True); gv = g(u)
        g1 = torch.autograd.grad(gv.sum(), u)[0]
        for gn, S in (("scaling", (u * g1 - 3 * gv).detach()), ("Galilean", (-g1).detach())):
            r = (columns(jets, g, nu) / torch.as_tensor(scale) @ torch.as_tensor(unit(KNOWN[gn], scale)[0])).detach()
            f = float((r @ S) / (S @ S))                                   # least-squares factor r ~ f * S
            rel = float(torch.linalg.norm(r - f * S) / torch.linalg.norm(r))
            print(f"  {gn:10s} general residual = {f:.4f} * hand-written S  (misfit {rel:.1e}, 0 = proportional)"
                  f"  -> L_general = {f*f:.4f} * L_old")

        # gradient check: d/dx gives zero gradient, scaling gives a gradient on c
        g.zero_grad()
        sym_loss_general(jets, g, nu, unit(KNOWN["d/dx"], scale), scale).backward()
        print(f"  d/dx gradient on c: {g.c.grad.abs().max():.1e} (expected 0) {ok(g.c.grad.abs().max() < 1e-12)}")
