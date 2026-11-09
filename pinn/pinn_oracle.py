import numpy as np, torch
from pinn.surrogate import Surrogate, jet, MODELS, DATA, RESULTS, NU, T
from pinn.discover_from_surrogate import pde_residual
from discovery.noise_gap import CASES
from pinn.surrogate import dense_truth, surrogate_jets, rel_error

G_TORCH = {"g0": lambda u: torch.zeros_like(u), "cubic": lambda u: -u**3}


def train_pinn(x_obs, t_obs, u_obs, g_fun, lam=1.0, n_colloc=2000,
               adam_steps=4000, verbose=True):
    history = []
    model = Surrogate()
    rng = np.random.default_rng(0)
    xc = torch.tensor(rng.uniform(0.0,2*np.pi , n_colloc), requires_grad=True)
    tc = torch.tensor(rng.uniform(0.0, T,   n_colloc), requires_grad=True)

    def losses():
        data = torch.mean((model(x_obs, t_obs) - u_obs)**2)
        u, u_x, u_t, u_xx, _ = jet(model, xc, tc)
        res  = u_t + u*u_x - NU*u_xx - g_fun(u)          # P1: the PDE residual
        return data, torch.mean(res**2)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(adam_steps):
        d, r = losses()
        loss = d + lam * r
        history.append((loss.item(), d.item(), r.item()))
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 500 == 0:
            print(f"  {step:5d}  data {d.item():.3e}   pde {r.item():.3e}")

    opt = torch.optim.LBFGS(model.parameters(), max_iter=2000,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        d, r = losses()
        loss = d + lam*r
        loss.backward()
        history.append((loss.item(), d.item(), r.item()))
        return loss
    opt.step(closure)
    d, r = losses()
    final = (d + lam*r).item()

    print(f"  final  data {d.item():.3e}   pde {r.item():.3e}")
    return model, final, history

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from pinn.plots import plot_training_oracle, plot_snapshots, plot_error_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("lam", nargs="?", default="1e-4", choices=["1e-4", "1e-2", "1"])
    ap.add_argument("--steps", type=int, default=4000)   # != 4000 -> smoke test: nothing saved to models/
    args = ap.parse_args()
    tag, lam, steps = args.lam, float(args.lam), args.steps
    smoke = steps != 4000
    RESULTS_O = RESULTS / "oracle"; RESULTS_O.mkdir(exist_ok=True)
    # python -m pinn.pinn_oracle 1e-4   -> trains g0+cubic x N{50,200,1000} x sigma{1,3} x k{0,1,2} for this lam
    for name, (g_sym, _) in CASES.items():
        for n_obs in (50, 200, 1000):
            for sig in (1, 3):
                s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig}.npz")
                for k in range(3):
                    stem = f"oracle_{name}_N{n_obs}_s{sig}_lam{tag}_k{k}"
                    if not smoke and (MODELS / f"{stem}.pt").exists():
                        print(f"skip {stem} (exists)"); continue
                    sel = s["exp_id"] == k
                    xo, to, uo = s["x_obs"][sel], s["t_obs"][sel], s["u_obs"][sel]
                    model, final, history = train_pinn(torch.tensor(xo), torch.tensor(to), torch.tensor(uo),
                                                       G_TORCH[name], lam=lam, adam_steps=steps, verbose=False)
                    if not smoke:
                        torch.save(model.state_dict(), MODELS / f"{stem}.pt")
                    x, t, truth = dense_truth(name, k)
                    pred = surrogate_jets(model, x, t)
                    print(f"{name} N={n_obs} s={sig}% lam={tag} k={k}  |D|={pde_residual(pred, g_sym):.4f}  " + "  ".join(
                        f"{key}={100*rel_error(pred[key], truth[key]):.2f}%" for key in ("u", "u_x", "u_t", "u_xx")))
                    out = ("smoke_" if smoke else "") + stem
                    noise_var = (sig/100*truth["u"].std())**2
                    plot_training_oracle(history, steps, lam, noise_var, RESULTS_O/f"training_{out}.png")
                    plot_snapshots(x, t, truth, pred, path=RESULTS_O/f"snapshots_{out}.png")
                    plot_error_maps(x, t, truth, pred, x_obs=xo, t_obs=to, path=RESULTS_O/f"error_maps_{out}.png")
                    plt.close("all")
