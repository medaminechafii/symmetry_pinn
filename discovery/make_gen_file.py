"""Write the generator file for the general symmetry loss (pinn.sym_general) from a trained BASELINE run.

The column scales are computed ONCE here, from the baseline model (network jets at its collocation points, its g_hat
and nu_hat), and then stay fixed during the training with the symmetry loss (design section 2.4).

For the delta pilot the generators are known ones (fingerprint.KNOWN); later the loop will write SVD candidates here.

    python -m discovery.make_gen_file --name cubic --N 1000 --sig 1 --k 0 --holdout 1.5 --gens scaling
    -> results/oracle/gens/<baseline stem>__scaling.npz   (V: unit vectors, scale, W: original coefficients)
"""
import argparse
import numpy as np, torch
from pinn.runs import load_rows, stem
from pinn.surrogate import Surrogate, MODELS, RESULTS, T
from pinn.learned_g import PolyG, MLPG
from pinn.sym_general import network_jets, columns, rms_scale, unit
from discovery.fingerprint import KNOWN

GENS_DIR = RESULTS / "oracle" / "gens"


def load_baseline(r):
    model = Surrogate(K=r.get("feat", 1)).double()
    model.load_state_dict(torch.load(MODELS / f"{stem(r)}.pt"))
    g = {"poly": PolyG, "MLP": MLPG}[r["g_model"]]()
    g.load_state_dict(torch.load(MODELS / f"{stem(r)}_g.pt"))
    return model, g.double()


def colloc(r):
    """The collocation points of the run (same rng as pinn_oracle.train_pinn)."""
    rng = np.random.default_rng(r["seed"])
    n = r.get("ncolloc", 2000)
    x = torch.tensor(rng.uniform(0.0, 2 * np.pi, n), requires_grad=True)
    t = torch.tensor(rng.uniform(0.0, T, n), requires_grad=True)
    return x, t


def baseline_columns(r):
    model, g = load_baseline(r)
    x, t = colloc(r)
    jets = network_jets(model, x, t)
    return columns(jets, g, torch.tensor(r["nu_hat"], dtype=torch.float64)).detach().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--sig", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["poly", "MLP"], default="poly")
    ap.add_argument("--holdout", type=float, default=None)
    ap.add_argument("--gens", nargs="+", required=True, choices=list(KNOWN))
    a = ap.parse_args()
    rows = [r for r in load_rows(a.arm, names=[a.name], holdout=a.holdout)
            if (r["N"], r["sig"], r["k"], r["seed"]) == (a.N, a.sig, a.k, a.seed)]
    assert len(rows) >= 1, "baseline run not found in metrics.jsonl"
    r = rows[-1]
    A = baseline_columns(r)
    scale = rms_scale(A)
    W = np.array([KNOWN[gn] for gn in a.gens])
    V = unit(W, scale)
    GENS_DIR.mkdir(parents=True, exist_ok=True)
    out = GENS_DIR / f"{stem(r)}__{'+'.join(a.gens).replace('/', '')}.npz"
    np.savez(out, V=V, scale=scale, W=W, gens=np.array(a.gens), baseline=stem(r))
    L0 = np.mean((A / scale @ V.T) ** 2)
    print(f"wrote {out}\n  L_sym of the baseline itself with these generators: {L0:.3e}")
