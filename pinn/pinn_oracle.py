import json
import numpy as np, torch
from pinn.surrogate import Surrogate, jet, MODELS, DATA, RESULTS, NU, T, variant_tag
from pinn.discover_from_surrogate import pde_residual
from discovery.noise_gap import CASES
from pinn.surrogate import dense_truth, surrogate_jets, rel_error

G_TORCH = {"g0": lambda u: torch.zeros_like(u), "cubic": lambda u: -u**3}

def train_pinn(x_obs, t_obs, u_obs, g_fun, lam=1.0, n_colloc=2000,seed=0,
               adam_steps=4000, verbose=True, cos=False, K=1):
    history = []
    grad_his = []
    torch.manual_seed(seed)
    model = Surrogate(K=K)
    rng = np.random.default_rng(seed)
    xc = torch.tensor(rng.uniform(0.0,2*np.pi , n_colloc), requires_grad=True)
    tc = torch.tensor(rng.uniform(0.0, T,   n_colloc), requires_grad=True)

    def losses():
        data = torch.mean((model(x_obs, t_obs) - u_obs)**2)
        u, u_x, u_t, u_xx, _ = jet(model, xc, tc)
        res  = u_t + u*u_x - NU*u_xx - g_fun(u)          # P1: the PDE residual
        return data, torch.mean(res**2)
    def grad_norms(model,d,r):
        params = list(model.parameters())
        gd = torch.autograd.grad(d,params,retain_graph=True)
        gr = torch.autograd.grad(r,params,retain_graph=True)
        norm = lambda gs: torch.sqrt(sum((g**2).sum() for g in gs)).item()

        return norm(gd), norm(gr)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-5) if cos else None
    for step in range(adam_steps):
        d, r = losses()
        loss = d + lam * r
        history.append((loss.item(), d.item(), r.item()))
        if step%100 == 0:
            gd_norm,gr_norm = grad_norms(model,d,r)
            grad_his.append((step,gd_norm,gr_norm,gd_norm/(lam*gr_norm)))

        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if verbose and step % 500 == 0:
            print(f"  {step:5d}  data {d.item():.3e}   pde {r.item():.3e}")

    opt = torch.optim.LBFGS(model.parameters(), max_iter=2000,
                            line_search_fn="strong_wolfe",tolerance_grad=1e-14, tolerance_change=1e-12)
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
    return model, final, history, grad_his

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from pinn.plots import plot_training_oracle, plot_snapshots, plot_error_maps,plot_grad_norms
    # python -m pinn.pinn_oracle 1e-4 0.1 1 --name g0 --n 1000 --sig 1 --k 0 --seed 0
    ap = argparse.ArgumentParser()
    ap.add_argument("lams", nargs="*", help="lambda values, written as in the file names (default 1e-4)")
    ap.add_argument("--name", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--n", nargs="+", type=int, default=[50, 200, 1000],help="data points 50,200 and 1000 as default")
    ap.add_argument("--sig", nargs="+", type=int, default=[1, 3],help="sigma for the noise either 1 and 3 as default")
    ap.add_argument("--k", nargs="+", type=int, default=[0, 1, 2],help="the number of the experiment depending on k 0,1 and 2 are the only experiments available")
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=4000, help="Adam steps")
    ap.add_argument("--ncolloc", type=int, default=2000, help="collocation points")
    ap.add_argument("--cos", action="store_true", help="cosine learning-rate decay 1e-3 -> 1e-5 during Adam")
    ap.add_argument("--feat", type=int, default=1, help="Fourier features sin(jx),cos(jx), j=1..feat (1 = original)")
    ap.add_argument("--smoke", action="store_true", help="quick test: saves no models, runs/ or metrics")
    ap.add_argument("--plots", action="store_true",
                    help="also save snapshots, error maps and gradient-norm plots (default: training curve only)")
    args = ap.parse_args()
    steps = args.steps
    smoke = args.smoke
    var = variant_tag(args.ncolloc, args.cos, steps, args.feat)
    RESULTS_O = RESULTS / "oracle"; RESULTS_O.mkdir(exist_ok=True)
    RUNS = RESULTS_O / "runs"; RUNS.mkdir(exist_ok=True)      # one .npz per run: arrays + scalars
    METRICS = RESULTS_O / "metrics.jsonl"                     # one JSON line per run, append-only
    for tag in (args.lams or ["1e-4"]):
        lam = float(tag)
        for seed in args.seed:
            for name in args.name:
                g_sym = CASES[name][0]
                for n_obs in args.n:
                    for sig in args.sig:
                        s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig}.npz")
                        for k in args.k:
                            stem = f"oracle_{name}_N{n_obs}_s{sig}_lam{tag}_k{k}{var}_seed{seed}"
                            if not smoke and (MODELS / f"{stem}.pt").exists():
                                print(f"skip {stem} (exists)"); continue
                            sel = s["exp_id"] == k
                            xo, to, uo = s["x_obs"][sel], s["t_obs"][sel], s["u_obs"][sel]
                            model, final, history, grad_his = train_pinn(
                                torch.tensor(xo), torch.tensor(to), torch.tensor(uo),
                                G_TORCH[name], lam=lam, adam_steps=steps, seed=seed, verbose=False,
                                n_colloc=args.ncolloc, cos=args.cos, K=args.feat)
                            if not smoke:
                                torch.save(model.state_dict(), MODELS / f"{stem}.pt")
                            x, t, truth = dense_truth(name, k)
                            pred = surrogate_jets(model, x, t)
                            print(f"{name} N={n_obs} s={sig}% lam={tag} k={k} seed={seed}  |D|={pde_residual(pred, g_sym):.4f}  " + "  ".join(
                                f"{key}={100*rel_error(pred[key], truth[key]):.2f}%" for key in ("u", "u_x", "u_t", "u_xx")))
                            out = ("smoke_" if smoke else "") + stem
                            noise_var = (sig/100*truth["u"].std())**2
                            with torch.no_grad():
                                data_final = torch.mean((model(torch.tensor(xo), torch.tensor(to)) - torch.tensor(uo))**2).item()
                            n_lbfgs = len(history) - steps               # closure calls in the L-BFGS stage
                            m = {"name": name, "N": n_obs, "sig": sig, "k": k, "lam": lam, "lam_tag": tag,
                                 "seed": seed, "steps": steps, "ncolloc": args.ncolloc, "cos": args.cos, "feat": args.feat, "D": float(pde_residual(pred, g_sym)),
                                 "err_u": float(rel_error(pred["u"], truth["u"])),
                                 "err_u_x": float(rel_error(pred["u_x"], truth["u_x"])),
                                 "err_u_t": float(rel_error(pred["u_t"], truth["u_t"])),
                                 "err_u_xx": float(rel_error(pred["u_xx"], truth["u_xx"])),
                                 "data_final": data_final, "noise_var": float(noise_var),
                                 "data_over_noise": data_final/float(noise_var),
                                 "lbfgs_evals": n_lbfgs, "lbfgs_ran": bool(n_lbfgs > 5),
                                 "underfit": bool(data_final > 3*noise_var)}
                            if not m["lbfgs_ran"] or m["underfit"]:
                                print(f"   FLAG: lbfgs_ran={m['lbfgs_ran']} data/noise={m['data_over_noise']:.1f}")
                            if not smoke:
                                np.savez(RUNS / f"{stem}.npz", history=np.array(history),
                                         grad_his=np.array(grad_his), adam_steps=steps, **m)
                                with open(METRICS, "a") as f:
                                    f.write(json.dumps(m) + "\n")
                            plot_training_oracle(history, steps, lam, noise_var, RESULTS_O/f"training_{out}.png")
                            if args.plots:
                                plot_snapshots(x, t, truth, pred, path=RESULTS_O/f"snapshots_{out}.png")
                                plot_error_maps(x, t, truth, pred, x_obs=xo, t_obs=to, path=RESULTS_O/f"error_maps_{out}.png")
                                plot_grad_norms(grad_his, path=RESULTS_O/f"grad_norms_{out}.png")
                            plt.close("all")
