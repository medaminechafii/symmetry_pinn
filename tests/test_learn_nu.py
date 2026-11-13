import numpy as np, torch
from pinn.pinn_oracle import train_pinn, G_TORCH
from pinn.surrogate import DATA

def load(name="g0", n=1000, sig=1, k=0):
    s = np.load(DATA / f"sparse_{name}_N{n}_s{sig}.npz"); m = s["exp_id"] == k
    return (torch.tensor(s[key][m]) for key in ("x_obs", "t_obs", "u_obs"))

def run(nu0, steps):
    xo, to, uo = load()
    model, *_ = train_pinn(xo, to, uo, G_TORCH["g0"], adam_steps=steps, learn_nu=True, nu0=nu0, verbose=False)
    return model.nu_hat

if __name__ == "__main__":
    nu = run(nu0=0.1, steps=200)
    print("test 2, start at truth:", nu, "PASS" if abs(nu - 0.1) / 0.1 < 0.5 else "FAIL")
    nu = run(nu0=0.03, steps=1500)
    print("test 3, start at 0.03:", nu, "PASS" if nu > 0.03 else "FAIL")