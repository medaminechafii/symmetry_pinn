import numpy as np,torch
from pathlib import Path
from discovery.fingerprint import spectral_derivative
from pinn.plots import plot_training, plot_snapshots, plot_error_maps
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
NU,T = 0.1, 2.0
seed = 0

class Surrogate(torch.nn.Module):
    def __init__(self, width=64, depth=3, K=1):
        super().__init__()
        self.K = K                                   # Fourier features: sin(jx), cos(jx), j = 1..K
        layers, n_in = [], 2*K + 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers += [torch.nn.Linear(n_in,1)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x, t):
        f = []
        for j in range(1, self.K + 1):
            f += [torch.sin(j*x), torch.cos(j*x)]
        feats = self.net(torch.stack(f + [t], dim=-1))
        return feats.squeeze(-1)
def variant_tag(nc=2000, cos=False, adam=4000, K=1):
    """File-name part for every non-default training option (default run -> empty, old names stay valid)."""
    return (f"_nc{nc}" if nc != 2000 else "") + ("_cos" if cos else "") + \
           (f"_ad{adam}" if adam != 4000 else "") + (f"_fK{K}" if K != 1 else "")
def jet( model,x, t):
    """Return u, u_x, u_t, u_xx evaluated at (x, t). x and t need requires_grad."""
    u = model(x,t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xt = torch.autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u,u_x,u_t,u_xx,u_xt
def train(x_obs, t_obs, u_obs, seed=0, adam_steps=4000, verbose= True):
    history = []
    torch.manual_seed(seed)
    model = Surrogate()
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(adam_steps):
        u_pred = model(x_obs,t_obs)
        loss = torch.mean((u_pred-u_obs)**2)
        history.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.3e}")
    opt = torch.optim.LBFGS(model.parameters(),max_iter=2000,line_search_fn="strong_wolfe",tolerance_grad = 1e-14,tolerance_change = 1e-12)
    def closure():
        opt.zero_grad()
        u_pred = model(x_obs,t_obs)
        loss = torch.mean((u_pred-u_obs)**2)
        history.append(loss.item())
        loss.backward()
        return loss
    opt.step(closure)
    final = closure().item()
    if verbose: print(f"Final Loss: {final:.3e}")
    return model, final,history

def dense_truth(name,k):
    """True jets on the full dense grid for experiment k."""
    d = np.load(DATA /f"dense_{name}.npz")
    x, t, U = d["x"], d["t"], d["U"][k]
    kk = np.fft.fftfreq(len(x), d=x[1]-x[0]) * 2 * np.pi
    Ux, Uxx = (spectral_derivative(U,kk,m) for m in (1,2))
    g = {"g0": np.zeros_like(U), "cubic": -U**3}[name]
    Ut = -U*Ux + NU*Uxx + g
    return x,t,{"u": U, "u_x": Ux, "u_t": Ut, "u_xx": Uxx} 
def surrogate_jets(model,x,t,chunk=4000):
    """Network jets on the same grid. Returns four (n_t, N) arrays."""
    X,T_ = np.meshgrid(x,t)
    xf, tf = X.ravel(), T_.ravel()
    out = {key: [] for key in ("u","u_x","u_t","u_xx")}
    for i in range(0,len(xf),chunk):
        xb = torch.tensor(xf[i:i+chunk],requires_grad=True)
        tb = torch.tensor(tf[i:i+chunk],requires_grad=True)
        for key, val in zip(out, jet(model,xb,tb)):
            out[key].append(val.detach().numpy())
    return {key: np.concatenate(val).reshape(X.shape) for key,val in out.items()}

def rel_error(pred, true):
    """Relative L2 error between two arrays."""
    return np.linalg.norm(true-pred)/np.linalg.norm(true)


if __name__== "__main__":
    import time
    name, n_obs,sig_pct, k = "g0", 1000,1,0
    s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig_pct}.npz")
    m = s["exp_id"]== k
    x_obs = torch.tensor(s["x_obs"][m],requires_grad=True)
    t_obs = torch.tensor(s["t_obs"][m],requires_grad=True)
    u_obs = torch.tensor(s["u_obs"][m])
    print(f"{name} N_obs={n_obs} sigma={sig_pct}%  -> {len(u_obs)} points for experiment {k}")
    # Train the surrogate model
    t0 = time.time()
    model,final,history = train(x_obs,t_obs,u_obs, seed=seed)
    torch.save(model.state_dict(), MODELS / f"{name}_N{n_obs}_s{sig_pct}_k{k}_seed{seed}.pt")

    print(f"Training completed in {time.time()-t0:.1f} seconds, final loss: {final:.3e}")

    # Evaluate the surrogate model on the dense grid and compare with the true solution
    x, t, truth = dense_truth(name,k)
    pred = surrogate_jets(model,x,t)
    for key in ("u","u_x","u_t","u_xx"):
        err = rel_error(pred[key], truth[key])
        print(f"Relative L2 error for {key}: {100*err:.3e}%")
    # ---overfitting check ---
    noise_var = (sig_pct/100*truth["u"].std())**2
    print(f"final loss: {final:.3e}, noise variance: {noise_var:.3e}, ratio: {final/noise_var:.2f}")
    plot_training(history, adam_steps=4000, noise_var=noise_var, path=RESULTS/f"training_{name}_N{n_obs}_sig{sig_pct}.png")
    plot_snapshots(x, t, truth, pred, ti=len(t)//2, path=RESULTS/f"snapshots_{name}_N{n_obs}_sig{sig_pct}.png")
    plot_error_maps(x, t, truth, pred, x_obs=x_obs.detach().numpy(), t_obs=t_obs.detach().numpy(), path=RESULTS/f"error_maps_{name}_N{n_obs}_sig{sig_pct}.png")
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for key in ("u", "u_x", "u_t", "u_xx"):
        rel = [np.linalg.norm(pred[key][i] - truth[key][i]) / np.linalg.norm(truth[key][i])
               for i in range(len(t))]
        ax.semilogy(t, rel, label=key)
    ax.set_xlabel("t"); ax.set_ylabel("relative L2 error at this time"); ax.legend(); ax.grid(alpha=.3)
    plt.show()