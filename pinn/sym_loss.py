"""Symmetry loss L_sym for the hybrid PINN (stage 2), with KNOWN generators first.

For a generator V = xi d/dx + tau d/dt + phi d/du with multiplier lam, the residual at a point is
    S_V = pr V(Delta) - lam * Delta,    Delta = u_t + u u_x - nu u_xx - g(u),
    pr V(Delta) = phi^t + phi u_x + u phi^x - nu phi^xx - g'(u) phi,
with phi^x, phi^t, phi^xx from the prolongation formulas (total derivatives through the network by autograd).
For the true generators of our equations S_V does not depend on the network jets at all:
    d/dx, d/dt : S = 0 for any g
    Galilean   : S = -g'(u)                    (forces g constant, i.e. g = 0 with g(0) = 0)
    scaling    : S = u g'(u) - 3 g(u) = sum_k (k-3) c_k u^k for a polynomial g
L_sym = mean over generators of mean(S^2) at the collocation points.

detach=True (default): the geometric quantities (jets, phi and its prolongation) are detached, so the gradient of L_sym
reaches only g and nu, never the network.  g' is taken with create_graph=True, otherwise g gets no gradient.
"""
import torch


def d(f, z):
    """Total derivative of f with respect to z through the graph (zero if f does not depend on z)."""
    if not f.requires_grad:
        return torch.zeros_like(z)
    g = torch.autograd.grad(f, z, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(z) if g is None else g


one = lambda x, t, u: torch.ones_like(x)
zero = lambda x, t, u: torch.zeros_like(x)

# name: (xi, tau, phi, lam(x, t))
GENERATORS = {
    "d/dx":        (one,                    zero,                     zero,                        lambda x, t: 0.0),
    "d/dt":        (zero,                   one,                      zero,                        lambda x, t: 0.0),
    "galilean":    (lambda x, t, u: t,      zero,                     one,                         lambda x, t: 0.0),
    "scaling":     (lambda x, t, u: x,      lambda x, t, u: 2 * t,    lambda x, t, u: -u,          lambda x, t: -3.0),
    "projective":  (lambda x, t, u: x * t,  lambda x, t, u: t**2,     lambda x, t, u: x - t * u,   lambda x, t: -3.0 * t),
    "not_a_symmetry": (lambda x, t, u: x,   lambda x, t, u: t,        zero,                        lambda x, t: -1.0),
}
# true symmetries of each case that are used in the loss (d/dx, d/dt give zero loss and zero gradient: left out;
# projective is a symmetry of g0 only and has no simple closed form with a learned g: optional)
CASE_GENS = {"g0": ["galilean", "scaling"], "cubic": ["scaling"]}


def prolong(gen, model, x, t):
    """Jets of the network and the prolonged generator at (x, t).  x, t need requires_grad=True."""
    xi_f, tau_f, phi_f, lam_f = GENERATORS[gen]
    u = model(x, t)
    u_x, u_t = d(u, x), d(u, t)
    u_xx, u_xt = d(u_x, x), d(u_x, t)
    xi, tau, phi = xi_f(x, t, u), tau_f(x, t, u), phi_f(x, t, u)
    phi_x = d(phi, x) - u_x * d(xi, x) - u_t * d(tau, x)
    phi_t = d(phi, t) - u_x * d(xi, t) - u_t * d(tau, t)
    phi_xx = d(phi_x, x) - u_xx * d(xi, x) - u_xt * d(tau, x)
    lam = lam_f(x, t)
    return dict(u=u, u_x=u_x, u_t=u_t, u_xx=u_xx, phi=phi, phi_x=phi_x, phi_t=phi_t, phi_xx=phi_xx), lam


def sym_residual(gen, model, g_model, nu, x, t, detach=True):
    q, lam = prolong(gen, model, x, t)
    if detach:
        q = {k: v.detach() for k, v in q.items()}
        if torch.is_tensor(lam):
            lam = lam.detach()
    u_ = q["u"].detach().requires_grad_(True) if detach else q["u"]
    g = g_model(u_)
    g1 = torch.autograd.grad(g, u_, grad_outputs=torch.ones_like(g), create_graph=True)[0]
    pr = q["phi_t"] + q["phi"] * q["u_x"] + q["u"] * q["phi_x"] - nu * q["phi_xx"] - g1 * q["phi"]
    delta = q["u_t"] + q["u"] * q["u_x"] - nu * q["u_xx"] - g
    return pr - lam * delta


def sym_loss(model, g_model, nu, x, t, gens, detach=True):
    """mean over generators of mean(S^2).  gens: list of names in GENERATORS."""
    return sum(torch.mean(sym_residual(gn, model, g_model, nu, x, t, detach) ** 2) for gn in gens) / len(gens)
