"""Tests for pinn/sym_loss.py and its hook in pinn.pinn_oracle.train_pinn.

    python -m pytest tests/test_sym_loss.py -q
"""
import numpy as np
import sympy as sp
import torch

from pinn.sym_loss import sym_residual, sym_loss
from pinn.surrogate import Surrogate, jet
from pinn.learned_g import PolyG

torch.set_default_dtype(torch.float64)
NU = 0.1
SYM_GENS = {   # the same generators as pinn.sym_loss.GENERATORS, written in sympy for the independent check
    "d/dx": ("1", "0", "0", "0"), "d/dt": ("0", "1", "0", "0"), "galilean": ("t", "0", "1", "0"),
    "scaling": ("x", "2*t", "-u", "-3"), "projective": ("x*t", "t**2", "x - t*u", "-3*t"),
    "not_a_symmetry": ("x", "t", "0", "-1"),
}


def make_case(seed=0, n=200, c=None):
    torch.manual_seed(seed)
    model = Surrogate()
    g = PolyG()
    with torch.no_grad():
        g.c.copy_(torch.tensor(c, dtype=torch.float64) if c is not None else 0.5 * torch.randn(5))
    x = (2 * np.pi * torch.rand(n)).requires_grad_(True)
    t = (2.0 * torch.rand(n)).requires_grad_(True)
    return model, g, x, t


def u_of(model, x, t):
    return model(x, t).detach()


def test_translations_give_zero():
    model, g, x, t = make_case()
    for gen in ("d/dx", "d/dt"):
        S = sym_residual(gen, model, g, NU, x, t)
        assert S.abs().max().item() < 1e-12, gen


def test_galilean_is_minus_g_prime():
    model, g, x, t = make_case()
    S = sym_residual("galilean", model, g, NU, x, t)
    u = u_of(model, x, t)
    c = g.c.detach()
    expected = -sum((k + 1) * c[k] * u**k for k in range(5))
    assert torch.allclose(S, expected, rtol=1e-10, atol=1e-12)


def test_scaling_is_sum_k_minus_3():
    model, g, x, t = make_case()
    S = sym_residual("scaling", model, g, NU, x, t)
    u = u_of(model, x, t)
    c = g.c.detach()
    expected = sum((k + 1 - 3) * c[k] * u ** (k + 1) for k in range(5))
    assert torch.allclose(S, expected, rtol=1e-10, atol=1e-12)


def test_prolongation_matches_sympy_criterion():
    """Independent check of the autograd prolongation against discovery/criterion.py for every generator."""
    from discovery.criterion import criterion, x as X, t as Tt, u as U, ux, ut, uxx, uxt
    model, g, x, t = make_case(seed=1)
    c = g.c.detach().numpy()
    g_sym = sum(sp.Float(float(c[k])) * U ** (k + 1) for k in range(5))
    uu, u_x, u_t, u_xx, u_xt = (v.detach().numpy() for v in jet(model, x, t))
    xs, ts = x.detach().numpy(), t.detach().numpy()
    delta = u_t + uu * u_x - NU * u_xx - sum(c[k] * uu ** (k + 1) for k in range(5))
    loc = {"x": X, "t": Tt, "u": U}
    for gen, (xi, tau, phi, lam) in SYM_GENS.items():
        expr = criterion(sp.sympify(xi, locals=loc), sp.sympify(tau, locals=loc), sp.sympify(phi, locals=loc), g_sym, NU)
        f = sp.lambdify((X, Tt, U, ux, ut, uxx, uxt), expr, "numpy")
        pr = np.broadcast_to(np.asarray(f(xs, ts, uu, u_x, u_t, u_xx, u_xt), dtype=float), xs.shape)
        lam_v = np.broadcast_to(np.asarray(sp.lambdify((X, Tt), sp.sympify(lam, locals=loc), "numpy")(xs, ts), dtype=float), xs.shape)
        expected = pr - lam_v * delta
        S = sym_residual(gen, model, g, NU, x, t).detach().numpy()
        assert np.allclose(S, expected, rtol=1e-9, atol=1e-11), gen


def test_gradient_reaches_g_only_and_pure_cubic_is_free():
    model, g, x, t = make_case()
    loss = sym_loss(model, g, NU, x, t, ["scaling"])
    loss.backward()
    assert g.c.grad is not None and g.c.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.parameters())     # detach=True: the network gets nothing
    assert abs(g.c.grad[2].item()) < 1e-12                     # c3 is never penalised by the scaling
    model, g, x, t = make_case(c=[0, 0, -1.3, 0, 0])
    loss = sym_loss(model, g, NU, x, t, ["scaling"])
    loss.backward()
    assert loss.item() < 1e-25 and g.c.grad.abs().max().item() < 1e-12


def test_not_a_symmetry_is_nonzero():
    model, g, x, t = make_case(c=[0, 0, 0, 0, 0])
    S = sym_residual("not_a_symmetry", model, g, NU, x, t)
    assert S.abs().mean().item() > 1e-6


def test_mu_zero_reproduces_the_run_without_sym_loss():
    from pinn.pinn_oracle import train_pinn
    rng = np.random.default_rng(0)
    xo = torch.tensor(rng.uniform(0, 2 * np.pi, 40)); to = torch.tensor(rng.uniform(0, 2, 40))
    uo = torch.sin(xo) * torch.exp(-to)
    common = dict(lam=1.0, n_colloc=100, seed=0, adam_steps=5, verbose=False, learn_nu=True, nu0=0.05,
                  lbfgs_iter=3)
    m0, f0, h0, _ = train_pinn(xo, to, uo, None, g_model="poly", **common)
    m1, f1, h1, _ = train_pinn(xo, to, uo, None, g_model="poly", sym=["scaling"], mu=0.0, **common)
    m2, f2, h2, _ = train_pinn(xo, to, uo, None, g_model="poly", sym=["scaling"], mu=1.0, **common)
    assert h0 == h1 and f0 == f1
    assert torch.equal(m0.extras["g_model"].c, m1.extras["g_model"].c)
    assert h2 != h0 and len(m2.extras["sym_hist"]) > 0


def test_g_wd_zero_reproduces_and_wd_shrinks_g():
    """Weight-decay baseline: g_wd = 0 gives the identical run; a large g_wd pulls every coefficient towards 0."""
    from pinn.pinn_oracle import train_pinn
    rng = np.random.default_rng(0)
    xo = torch.tensor(rng.uniform(0, 2 * np.pi, 40)); to = torch.tensor(rng.uniform(0, 2, 40))
    uo = torch.sin(xo) * torch.exp(-to)
    common = dict(lam=1.0, n_colloc=100, seed=0, adam_steps=200, verbose=False, learn_nu=True, nu0=0.05,
                  lbfgs_iter=20)
    m0, f0, h0, _ = train_pinn(xo, to, uo, None, g_model="poly", **common)
    m1, f1, h1, _ = train_pinn(xo, to, uo, None, g_model="poly", g_wd=0.0, **common)
    m2, f2, h2, _ = train_pinn(xo, to, uo, None, g_model="poly", g_wd=10.0, **common)
    c0, c1, c2 = (m.extras["g_model"].c.detach() for m in (m0, m1, m2))
    assert h0 == h1 and f0 == f1 and torch.equal(c0, c1)
    assert m1.extras["g_wd"] == 0.0 and m2.extras["g_wd"] == 10.0
    assert (c2**2).sum() < 0.1 * (c0**2).sum()
