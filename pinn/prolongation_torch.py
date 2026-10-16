import torch
import torch.nn as nn
torch.manual_seed(0)
def d(f,z):
    """ Derivative of f with respect to z, keeping the graph for higher derivatives """
    if not f.requires_grad:
        return torch.zeros_like(z)
    g = torch.autograd.grad(f,z,grad_outputs=torch.ones_like(f), create_graph = True, allow_unused=True)[0]
    return torch.zeros_like(z) if g is None else g
def jet(net,x,t):
    u = net(torch.cat([x,t], dim = 1))
    u_x, u_t = d(u,x), d(u,t)
    u_xx, u_xt = d(u_x,x), d(u_x,t)
    return u,u_x,u_t,u_xx,u_xt
def prolong_torch(xi_fn,tau_fn,phi_fn,net,x,t):
    u,u_x,u_t,u_xx,u_xt = jet(net,x,t)
    xi, tau,phi = xi_fn(x,t,u), tau_fn(x,t,u), phi_fn(x,t,u)
    phi_x = d(phi,x) - u_x*d(xi,x) - u_t*d(tau,x)
    phi_t = d(phi,t) - u_x*d(xi,t) - u_t*d(tau,t)
    phi_xx = d(phi_x,x) - u_xx*d(xi,x) - u_xt*d(tau,x)
    return (u,u_x,u_t,u_xx), (xi,tau,phi,phi_x,phi_t,phi_xx)


def make_mlp(width=32,depth=3,activation=nn.Tanh):
    """MLP from (x,t) to u: `depth` hidden layers of `width` neurons."""

    layers = [nn.Linear(2,width),activation()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width,width), activation()]
    layers.append(nn.Linear(width,1))
    return nn.Sequential(*layers)
net = make_mlp()

x = torch.rand(100,1,requires_grad=True)
t = torch.rand(100,1,requires_grad=True)
u = net(torch.cat([x,t],dim=1))
u_x, u_t = d(u,x), d(u,t)
u_xx,u_xt = d(u_x,x), d(u_x,t)
xi_fn = lambda x,t,u:x
tau_fn = lambda x,t,u:2*t
phi_fn = lambda x,t,u:-u
one = lambda x,t,u:torch.ones_like(x)
zero = lambda x,t,u:torch.zeros_like(x)
u,u_x,u_t,u_xx,u_xt = jet(net,x,t)
xi,tau,phi = xi_fn(x,t,u), tau_fn(x,t,u), phi_fn(x,t,u)

nu = 0.1
generators = {
    # name:            (xi_fn,              tau_fn,              phi_fn,                  lambda)
    "x-translation":   (one,                zero,                zero,                    lambda t: 0),
    "galilean":        (lambda x,t,u: t,    zero,                one,                     lambda t: 0),
    "scaling":         (lambda x,t,u: x,    lambda x,t,u: 2*t,   lambda x,t,u: -u,        lambda t: -3),
    "projective":      (lambda x,t,u: x*t,  lambda x,t,u: t**2,  lambda x,t,u: x - t*u,   lambda t: -3*t),
    "NOT a symmetry":  (lambda x,t,u: x,    lambda x,t,u: t,     zero,                    lambda t: -1),
}
for name,(xi_fn,tau_fn,phi_fn,lam) in generators.items():
    (u, u_x, u_t, u_xx), (xi, tau, phi, phi_x, phi_t, phi_xx) = prolong_torch(xi_fn,tau_fn,phi_fn,net,x,t)
    Delta = u_t + u*u_x - nu*u_xx
    prv = phi_t + phi*u_x + u*phi_x - nu*phi_xx
    print(f"{name}: {torch.mean(prv - lam(t)*Delta).item():.2e}")
    