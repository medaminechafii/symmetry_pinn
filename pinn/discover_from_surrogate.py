import numpy as np
import sympy as sp
import torch
import matplotlib.pyplot as plt
from discovery.noise_gap import column_raw, evaluate7, gap_at, JET7_KEY, CASES
from discovery.fingerprint import  normalise, count_by_gap, KNOWN, projection_residual
from pinn.surrogate import Surrogate, jet, MODELS, NU, T, train,DATA

N_COLLOC = 3000
def get_model(name, n_obs, sig_pct, k,retrain=False):
        """Load this experiment's surrogate, training and saving it if it isn't on disk."""
        path = MODELS / f"{name}_N{n_obs}_s{sig_pct}_k{k}.pt"
        if path.exists() and not retrain:
            model = Surrogate()
            model.load_state_dict(torch.load(path))
            return model
        else:
            s = np.load(DATA / f"sparse_{name}_N{n_obs}_s{sig_pct}.npz")
            m = s["exp_id"] == k
            x_obs = torch.tensor(s["x_obs"][m], requires_grad=True)
            t_obs = torch.tensor(s["t_obs"][m], requires_grad=True)
            u_obs = torch.tensor(s["u_obs"][m])
            print(f"{name} N_obs={n_obs} sigma={sig_pct}%  -> {len(u_obs)} points for experiment {k}")
            model, final, _ = train(x_obs, t_obs, u_obs,verbose=False)
            MODELS.mkdir(exist_ok=True)
            torch.save(model.state_dict(), path)
            return model

def load_models(name, n_obs, sig_pct, K=3):
    return [get_model(name, n_obs, sig_pct, k) for k in range(K)]
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
    rng = np.random.default_rng(0)
    for name, (g, k_exp) in CASES.items():
        cols = column_raw(g)
        for n_obs in (50, 200, 1000):
            for sig in (1,3):
                rng = np.random.default_rng(0)
                models = load_models(name,n_obs,sig)
                jd = surrogate_jets(models, N_COLLOC, rng)

                A = np.column_stack([evaluate7(c, jd) for c in cols])
                A_n, scale, is_zero = normalise(A)
                _, s, Vt = np.linalg.svd(A_n, full_matrices=False)

                res = {gn: projection_residual(w, Vt, k_exp, scale) for gn, w in KNOWN.items()}
                print(f"{name} N={n_obs:5d} s={sig}%  count={count_by_gap(s)[0]}  "
                      f"gap={gap_at(s, k_exp):5.2f}  |D|={pde_residual(jd, g):.3f}  "
                      + "  ".join(f"{gn}={res[gn]:.1e}" for gn in KNOWN))



        
    

