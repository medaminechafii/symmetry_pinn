import json
import numpy as np, torch
from pinn.surrogate import Surrogate, jet, MODELS, DATA, RESULTS, NU, T, variant_tag
from pinn.discover_from_surrogate import pde_residual
from discovery.noise_gap import CASES,ALL_CASES,EPS
from pinn.surrogate import dense_truth, surrogate_jets, rel_error
from pinn.learned_g import PolyG,MLPG
from pinn.sym_loss import sym_loss, CASE_GENS

G_TORCH = {"g0": lambda u: torch.zeros_like(u), "cubic": lambda u: -u**3}

G_TORCH.update({f"cubic_{tag}": lambda u, e=eps: -u**3 + e*u for tag, eps in EPS.items()})

def train_pinn(x_obs, t_obs, u_obs, g_fun, lam=1.0, n_colloc=2000,seed=0,
               adam_steps=4000, verbose=True, cos=False, K=1,learn_nu = False,nu0 = 0.05,g_model=None,
               sym=None, mu=0.0, sym_detach=True, lbfgs_iter=2000):
    """sym: list of generator names (pinn.sym_loss.GENERATORS); the symmetry loss mu * L_sym is only computed
    and added when sym is given AND mu > 0, so mu = 0 reproduces the run without it exactly."""
    if g_model is not None:
        torch.manual_seed(seed)
        if isinstance(g_model, str):
            g_model = {"poly": PolyG, "MLP": MLPG}[g_model]()
        g_fun = g_model
    history = []
    grad_his = []
    torch.manual_seed(seed)
    model = Surrogate(K=K)
    log_nu = torch.tensor(float(np.log(nu0)),requires_grad=True) if learn_nu else None
    def nu_now():
        return torch.exp(log_nu) if learn_nu else NU
    rng = np.random.default_rng(seed)
    xc = torch.tensor(rng.uniform(0.0,2*np.pi , n_colloc), requires_grad=True)
    tc = torch.tensor(rng.uniform(0.0, T,   n_colloc), requires_grad=True)

    def losses():
        data = torch.mean((model(x_obs, t_obs) - u_obs)**2)
        u, u_x, u_t, u_xx, _ = jet(model, xc, tc)
        res  = u_t + u*u_x - nu_now()*u_xx - g_fun(u)          # P1: the PDE residual
        return data, torch.mean(res**2)
    use_sym = bool(sym) and mu > 0
    sym_hist = []
    def sym_term():
        return sym_loss(model, g_fun, nu_now(), xc, tc, sym, detach=sym_detach)
    def grad_norms(model,d,r):
        params = list(model.parameters()) 
        gd = torch.autograd.grad(d,params,retain_graph=True)
        gr = torch.autograd.grad(r,params,retain_graph=True)
        norm = lambda gs: torch.sqrt(sum((g**2).sum() for g in gs)).item()

        return norm(gd), norm(gr)
    all_params = list(model.parameters()) + ([log_nu] if learn_nu else []) + (list(g_model.parameters()) if g_model is not None else [])
    opt = torch.optim.Adam(all_params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-5) if cos else None
    for step in range(adam_steps):
        d, r = losses()
        loss = d + lam * r
        if use_sym:
            s = sym_term()
            loss = loss + mu * s
            sym_hist.append(s.item())
        history.append((loss.item(), d.item(), r.item()))
        if step%100 == 0:
            gd_norm,gr_norm = grad_norms(model,d,r)
            grad_his.append((step,gd_norm,gr_norm,gd_norm/(lam*gr_norm)))

        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if verbose and step % 500 == 0:
            print(f"  {step:5d}  data {d.item():.3e}   pde {r.item():.3e}")

    opt = torch.optim.LBFGS(all_params, max_iter=lbfgs_iter,
                            line_search_fn="strong_wolfe",tolerance_grad=1e-14, tolerance_change=1e-12)
    def closure():
        opt.zero_grad()
        d, r = losses()
        loss = d + lam*r
        if use_sym:
            s = sym_term()
            loss = loss + mu * s
            sym_hist.append(s.item())
        loss.backward()
        history.append((loss.item(), d.item(), r.item()))
        return loss
    opt.step(closure)
    d, r = losses()
    final = (d + lam*r).item()
    s_final = sym_term().item() if use_sym else None
    if use_sym:
        final += mu * s_final

    print(f"  final  data {d.item():.3e}   pde {r.item():.3e}" + (f"   sym {s_final:.3e}" if use_sym else ""))
    model.nu_hat = float(nu_now().detach()) if learn_nu else NU
    model.extras = {"g_model":g_model, "sym": sym if use_sym else None, "mu": mu if use_sym else 0.0,
                    "sym_hist": sym_hist, "sym_final": s_final}
    return model, final, history, grad_his

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from pinn.plots import plot_training_oracle, plot_snapshots, plot_error_maps,plot_grad_norms
    # python -m pinn.pinn_oracle 1e-4 0.1 1 --name g0 --n 1000 --sig 1 --k 0 --seed 0
    ap = argparse.ArgumentParser()
    ap.add_argument("lams", nargs="*", help="lambda values, written as in the file names (default 1e-4)")
    ap.add_argument("--name", nargs="+", choices=list(ALL_CASES), default=list(CASES))
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
    ap.add_argument("--loss-g" , choices=list(G_TORCH),default=None,help="g loss = -u**3 with default same as delta)")
    ap.add_argument("--nu",type=int, default=0.1)
    ap.add_argument("--learn-nu", type=float, default=None,
                help="___ (hint: learn nu, starting at this value; default = nu fixed at the truth)")
    ap.add_argument("--g-model",choices=["poly","MLP"],default=None)
    ap.add_argument("--sym", nargs="*", choices=["auto", "galilean", "scaling", "projective"], default=None,
                    help="generators for the symmetry loss; 'auto' = the true ones of the case (pinn.sym_loss.CASE_GENS)")
    ap.add_argument("--mu", type=float, default=0.0, help="weight of the symmetry loss (0 = off)")
    args = ap.parse_args()
    steps = args.steps
    smoke = args.smoke
    learn_nu = args.learn_nu
    var = variant_tag(args.ncolloc, args.cos, steps, args.feat)
    RESULTS_O = RESULTS / "oracle"; RESULTS_O.mkdir(exist_ok=True)
    RUNS = RESULTS_O / "runs"; RUNS.mkdir(exist_ok=True)      # one .npz per run: arrays + scalars
    METRICS = RESULTS_O / "metrics.jsonl" 
                        # one JSON line per run, append-only
    for tag in (args.lams or ["1e-4"]):
        lam = float(tag)
        for seed in args.seed:
            for name in args.name:
                g_sym = ALL_CASES[name][0]
                for n_obs in args.n:
                    for sig in args.sig:
                        s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig}.npz")
                        for k in args.k:
                            mis = f"_loss{args.loss_g}" if args.loss_g else ""
                            nutag = f"_nu{args.learn_nu}" if args.learn_nu is not None else ""
                            g_tag = f"_{args.g_model}" if args.g_model is not None else ""
                            gens = None
                            if args.sym and args.mu > 0:
                                gens = CASE_GENS[name] if args.sym == ["auto"] else list(args.sym)
                            sym_tag = f"_sym{'+'.join(gens)}_mu{args.mu:g}" if gens else ""
                            stem = f"oracle_{name}_N{n_obs}_s{sig}_lam{tag}_k{k}{var}{mis}{nutag}{g_tag}{sym_tag}_seed{seed}"
                            if not smoke and (MODELS / f"{stem}.pt").exists():
                                print(f"skip {stem} (exists)"); continue
                            sel = s["exp_id"] == k
                            xo, to, uo = s["x_obs"][sel], s["t_obs"][sel], s["u_obs"][sel]
                            g_loss = G_TORCH[args.loss_g or name]
                            g_model = args.g_model
                            model, final, history, grad_his = train_pinn(
                                torch.tensor(xo), torch.tensor(to), torch.tensor(uo),
                                g_loss, lam=lam, adam_steps=steps, seed=seed, verbose=False,
                                n_colloc=args.ncolloc, cos=args.cos, K=args.feat,learn_nu = learn_nu is not None,
                                nu0 = learn_nu if learn_nu is not None else 0.05,g_model=g_model,
                                sym=gens, mu=args.mu)
                            g = model.extras["g_model"]
                            if not smoke:
                                torch.save(model.state_dict(), MODELS / f"{stem}.pt")
                                if g is not None:
                                    torch.save(g.state_dict(),MODELS / f"{stem}_g.pt")
                            x, t, truth = dense_truth(name, k)
                            pred = surrogate_jets(model, x, t)
                            print(f"{name} N={n_obs} s={sig}% lam={tag} k={k} seed={seed}  |D|={pde_residual(pred, g_sym):.4f}  " + "  ".join(
                                f"{key}={100*rel_error(pred[key], truth[key]):.2f}%" for key in ("u", "u_x", "u_t", "u_xx")))
                            out = ("smoke_" if smoke else "") + stem
                            noise_var = (sig/100*truth["u"].std())**2
                            with torch.no_grad():
                                data_final = torch.mean((model(torch.tensor(xo), torch.tensor(to)) - torch.tensor(uo))**2).item()
                            n_lbfgs = len(history) - steps               # closure calls in the L-BFGS stage
                            m = {
                                "name": name, "N": n_obs, "sig": sig, "k": k, "lam": lam, "lam_tag": tag,
                                 "seed": seed, "steps": steps, "ncolloc": args.ncolloc, "cos": args.cos, "feat": args.feat, "D": float(pde_residual(pred, g_sym)),
                                 "err_u": float(rel_error(pred["u"], truth["u"])),
                                 "err_u_x": float(rel_error(pred["u_x"], truth["u_x"])),
                                 "err_u_t": float(rel_error(pred["u_t"], truth["u_t"])),
                                 "err_u_xx": float(rel_error(pred["u_xx"], truth["u_xx"])),
                                 "data_final": data_final, "noise_var": float(noise_var),
                                 "data_over_noise": data_final/float(noise_var) if noise_var > 0 else None,
                                 "lbfgs_evals": n_lbfgs, "lbfgs_ran": bool(n_lbfgs > 5),
                                 "underfit": bool(noise_var > 0 and data_final > 3*noise_var),
                                 "loss_g": (args.loss_g or name) if args.g_model is None else None,
                                 "learn_nu": args.learn_nu is not None,
                                 "nu0": args.learn_nu,
                                 "nu_hat": model.nu_hat, 
                                 "nu_err": abs(model.nu_hat - NU) / NU,
                                 "g_model": args.g_model,
                                 "c": g.c.tolist() if isinstance(g, PolyG) else None,
                                 "sym": model.extras["sym"],
                                 "mu": model.extras["mu"],
                                 "sym_final": model.extras["sym_final"]
                                 }
                            if not m["lbfgs_ran"] or m["underfit"]:
                                ratio = m["data_over_noise"]
                                ratio_txt = f"{ratio:.1f}" if ratio is not None else "n/a"
                                print(f"   FLAG: lbfgs_ran={m['lbfgs_ran']} data/noise={ratio_txt}")
                            if not smoke:
                                np.savez(RUNS / f"{stem}.npz", history=np.array(history),
                                         grad_his=np.array(grad_his), adam_steps=steps, **m)
                                with open(METRICS, "a") as f:
                                    f.write(json.dumps(m) + "\n")
                            plot_training_oracle(history, steps, lam, noise_var if noise_var > 0 else None, RESULTS_O/f"training_{out}.png")
                            if args.plots:
                                plot_snapshots(x, t, truth, pred, path=RESULTS_O/f"snapshots_{out}.png")
                                plot_error_maps(x, t, truth, pred, x_obs=xo, t_obs=to, path=RESULTS_O/f"error_maps_{out}.png")
                                plot_grad_norms(grad_his, path=RESULTS_O/f"grad_norms_{out}.png")
                            plt.close("all")
