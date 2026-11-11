import numpy as np
import sympy as sp
import torch
import matplotlib.pyplot as plt
from discovery.noise_gap import column_raw, evaluate7, gap_at, JET7_KEY, CASES
from discovery.fingerprint import  normalise, count_by_gap, KNOWN, projection_residual
from pinn.surrogate import Surrogate, jet, MODELS, NU, T, train, DATA, variant_tag

N_COLLOC = 3000
def find_by_lambda(name, n_obs, sig_pct, k, prefix, tag, suffix):
    """Oracle files are named with the lambda text exactly as typed ('1', '1.0', '0.1', '1e-4').
    If the exact name is missing, accept any file whose lambda has the same numeric value."""
    want = float(tag[len("_lam"):])
    head, tail = f"{prefix}{name}_N{n_obs}_s{sig_pct}_lam", f"_k{k}{suffix}.pt"
    for c in sorted(MODELS.glob(f"{head}*{tail}")):
        try:
            if float(c.name[len(head):-len(tail)]) == want:
                return c
        except ValueError:
            pass
    return None
def get_model(name, n_obs, sig_pct, k, prefix="", tag="", suffix="", seed=0, retrain=False, feat=1):
        """Load this experiment's surrogate, training and saving it if it isn't on disk."""
        path = MODELS / f"{prefix}{name}_N{n_obs}_s{sig_pct}{tag}_k{k}{suffix}.pt"
        if not path.exists() and prefix and tag.startswith("_lam"):
            path = find_by_lambda(name, n_obs, sig_pct, k, prefix, tag, suffix) or path
        if path.exists() and not retrain:
            model = Surrogate(K=feat)
            model.load_state_dict(torch.load(path))
            return model
        elif prefix or tag:
            raise FileNotFoundError(f"{path.name} missing: oracle models are trained by pinn.pinn_oracle, not here")
        else:
            s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig_pct}.npz")
            m = s["exp_id"] == k
            x_obs = torch.tensor(s["x_obs"][m], requires_grad=True)
            t_obs = torch.tensor(s["t_obs"][m], requires_grad=True)
            u_obs = torch.tensor(s["u_obs"][m])
            print(f"{name} N_obs={n_obs} sigma={sig_pct}%  -> {len(u_obs)} points for experiment {k}")
            model, final, _ = train(x_obs, t_obs, u_obs, seed=seed, verbose=False)
            MODELS.mkdir(exist_ok=True)
            torch.save(model.state_dict(), path)
            return model

def load_models(name, n_obs, sig_pct, K=3, prefix="", tag="", suffix="", seed=0, feat=1):
    return [get_model(name, n_obs, sig_pct, k, prefix, tag, suffix, seed, feat=feat) for k in range(K)]
def surrogate_jets(models,n_points,rng,t_min=0.0,t_max=T):
    """Random collocation points across all experiments -> the 7-variable jet dict."""
    per, parts = n_points//len(models), []
    for model in models:
        xs = rng.uniform(0.0, 2*np.pi, size=per)
        ts = rng.uniform(t_min, t_max, size=per)
        xb = torch.tensor(xs, requires_grad=True)
        tb = torch.tensor(ts, requires_grad=True)
        u, ux, ut, uxx, uxt = jet(model, xb, tb)
        parts.append({"x": xs, "t": ts, "u": u.detach().numpy(),"u_x": ux.detach().numpy(), "u_t": ut.detach().numpy(), "u_xx": uxx.detach().numpy(), "u_xt": uxt.detach().numpy()})
    return {key: np.concatenate([p[key] for p in parts]) for key in JET7_KEY}
def pde_residual(jd, g_sym):
    """How badly these jets violate the equation -- the bridge to Part B."""
    g_fun = sp.lambdify(sp.symbols("u"), g_sym, "numpy")
    gv = np.broadcast_to(np.asarray(g_fun(jd["u"]),dtype=float),jd["u"].shape)
    delta = jd["u_t"] + jd["u"]*jd["u_x"] - NU*jd["u_xx"] - gv
    return np.linalg.norm(delta)/np.linalg.norm(jd["u_t"])

if __name__ == "__main__":
    # data-only M1 (legacy files, no seed in name):  python -m pinn.discover_from_surrogate
    # oracle, one lam:       python -m pinn.discover_from_surrogate oracle 1e-4
    # oracle, a few lams, one setting, 2 seeds:
    #   python -m pinn.discover_from_surrogate oracle 1e-4 0.1 1 --name g0 --n 1000 --sig 1 --seed 0 1
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["oracle"])
    ap.add_argument("lams", nargs="*", help="lambda values as in the file names, e.g. 1e-4 0.1 1")
    ap.add_argument("--name", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--n", nargs="+", type=int, default=[50, 200, 1000])
    ap.add_argument("--sig", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--steps", type=int, default=4000, help="Adam steps the models were trained with")
    ap.add_argument("--ncolloc", type=int, default=2000)
    ap.add_argument("--cos", action="store_true")
    ap.add_argument("--feat", type=int, default=1)
    ap.add_argument("--seed", nargs="+", type=int, default=None)   # oracle: default 0; data-only: none = legacy names
    args = ap.parse_args()
    oracle = args.mode == "oracle"
    lams = (args.lams or ["1e-4"]) if oracle else [None]
    seeds = args.seed if args.seed is not None else ([0] if oracle else [None])
    for lam in lams:
        for seed in seeds:
            prefix, tag = ("oracle_", f"_lam{lam}") if oracle else ("", "")
            suffix = (variant_tag(args.ncolloc, args.cos, args.steps, args.feat) if oracle else "") + (f"_seed{seed}" if seed is not None else "")
            if oracle or seed is not None:
                print(f"--- {'oracle lam=' + lam if oracle else 'data-only'}  seed={seed}")
            for name in args.name:
                g, k_exp = CASES[name]
                cols = column_raw(g)
                for n_obs in args.n:
                    for sig in args.sig:
                        rng = np.random.default_rng(0)
                        models = load_models(name, n_obs, sig, prefix=prefix, tag=tag,
                                             suffix=suffix, seed=seed or 0, feat=args.feat)
                        jd = surrogate_jets(models, N_COLLOC, rng)

                        A = np.column_stack([evaluate7(c, jd) for c in cols])
                        A_n, scale, is_zero = normalise(A)
                        _, s, Vt = np.linalg.svd(A_n, full_matrices=False)

                        res = {gn: projection_residual(w, Vt, k_exp, scale) for gn, w in KNOWN.items()}
                        print(f"{name} N={n_obs:5d} s={sig}%  count={count_by_gap(s)[0]}  "
                              f"gap={gap_at(s, k_exp):5.2f}  |D|={pde_residual(jd, g):.3f}  "
                              + "  ".join(f"{gn}={res[gn]:.1e}" for gn in KNOWN))
