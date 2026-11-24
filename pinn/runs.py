"""Selecting learned-g runs from results/oracle/metrics.jsonl and rebuilding their model file names.

Stage 1 (no symmetry loss) and stage 2 (symmetry loss, mu > 0) runs share most labels (arm, case, N, sigma, k, seed),
so every analysis must select one value of mu.  Rows written before the symmetry loss existed have no "mu": they count
as mu = 0.
"""
import json
import pinn.surrogate as ps


def mu_of(r):
    return float(r.get("mu") or 0.0)


def load_rows(arm=None, mu=0.0, names=None):
    """Learned-g rows (learn_nu, g_model == arm, or poly/MLP if arm is None) with symmetry weight mu."""
    rows = [json.loads(l) for l in open(ps.RESULTS / "oracle" / "metrics.jsonl")]
    keep = []
    for r in rows:
        if not r.get("learn_nu"):
            continue
        if (r.get("g_model") != arm) if arm else (r.get("g_model") not in ("poly", "MLP")):
            continue
        if abs(mu_of(r) - mu) > 1e-12:
            continue
        if names and r["name"] not in names:
            continue
        keep.append(r)
    return keep


def sym_tag(r):
    return f"_sym{'+'.join(r['sym'])}_mu{r['mu']:g}" if r.get("sym") and mu_of(r) > 0 else ""


def stem(r):
    """Model file stem of a learned-g run, as written by pinn.pinn_oracle."""
    var = ps.variant_tag(r.get("ncolloc", 2000), r.get("cos", False), r.get("steps", 4000), r.get("feat", 1))
    return (f"oracle_{r['name']}_N{r['N']}_s{r['sig']}_lam{r['lam_tag']}_k{r['k']}{var}"
            f"_nu{r['nu0']}_{r['g_model']}{sym_tag(r)}_seed{r['seed']}")
