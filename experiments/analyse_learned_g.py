import json, numpy as np
from pinn.surrogate import dense_truth

rows = [json.loads(l) for l in open("results/oracle/metrics.jsonl")]
rows = [r for r in rows if r.get("g_model") == "poly"]      # the 18 learned-g runs
print(len(rows), "rows")                                      # expect 18

ks = np.arange(1, 6)
out = []
for r in rows:
    name, k = r["name"], r["k"]
    c = np.array(r["c"])
    _, _, truth = dense_truth(name, k)
    u = truth["u"].ravel()                  # the u values the solution visits

    g_hat  = (c[None, :] * u[:, None] ** ks[None, :]).sum(1)   # sum_k c_k u^k at every u
    g_true = np.zeros_like(u) if name=="g0" else -u**3   # 0 for g0, -u**3 for cubic
    scale  = np.sqrt(np.mean((u**3)**2))  # RMS of u**3
    err_g  = np.sqrt(np.mean((g_hat - g_true)**2)) / scale

    c3     = c[2]   # c[2]  (the u^3 coefficient); compare to -1 for cubic
    c_nc   = c.copy(); c_nc[2] = 0
    noncub = np.linalg.norm(c_nc)   # norm of c with index 2 set to 0
    homog  = np.sum((ks - 3)[None, :] * c[None, :] * u[:, None] ** ks[None, :], axis=1)
    R      = np.linalg.norm(homog) / (3 * np.linalg.norm(g_hat)) if name == "cubic" else np.linalg.norm(homog)   # norm of sum_k (k-3) c_k u^k, divided by 3*norm(g_hat) for cubic; absolute for g0

    out.append((name, k, r["seed"], err_g, c3, noncub, R, r["nu_err"], r["D"]))
    print(name, k, r["seed"], f"err_g={err_g:.3f} c3={c3:.3f} noncub={noncub:.3f} R={R:.3f} nu_err={r['nu_err']:.3f} D={r['D']:.4f}")

# for name in ("g0","cubic"): print median and min..max of err_g, c3, nu_err, D over the 9 runs
for name in ("g0", "cubic"):
    sub = np.array([o[3:] for o in out if o[0] == name])      # columns: err_g, c3, noncub, R, nu_err, D
    for i, lab in enumerate(["err_g", "c3", "noncub", "R", "nu_err", "D"]):
        print(f"{name:6s} {lab:7s} median {np.median(sub[:, i]):.4f}   min {sub[:, i].min():.4f}   max {sub[:, i].max():.4f}")