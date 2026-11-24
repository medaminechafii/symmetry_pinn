"""Data-free calibration of the substituted gap against the size of the symmetry breaking.

With the EXACT near-symmetric equation  g = -u^3 + eps*u  (true nu), the substituted columns do not
depend on any network error. d/dx and d/dt stay exact symmetries (k = 2); scaling is broken by eps
(u g' - 3 g = -2 eps u), so the third-smallest singular value grows with eps.
The jets are only the sampling points: here the jets of the dense CUBIC solution are used for every eps
(the eps solutions themselves are not needed, and are not on every machine).

    python -m discovery.eps_calibration
"""
import numpy as np

from discovery.criterion import columns
from discovery.fingerprint import evaluate, N_POINTS
from discovery.noise_gap import CASES, NEAR_CASES, EPS, gap_at
from discovery.stage0_floor import load_truth, sample_indices, pool, svd_stats

if __name__ == "__main__":
    g_cubic, _ = CASES["cubic"]
    x, t, J = load_truth("cubic", g_cubic)                     # jets of the cubic solution, all 3 experiments
    it, ix = sample_indices(np.random.default_rng(0), len(t), len(x))
    jd = pool(J, x, t, it, ix)
    print(f"sampling points: {len(jd['u'])}, u range [{jd['u'].min():.2f}, {jd['u'].max():.2f}], "
          f"rms u = {np.sqrt(np.mean(jd['u']**2)):.3f}")

    cases = [("cubic (eps=0)", 0.0, g_cubic)] + [(n, EPS[n.split('_', 1)[1]], g) for n, (g, _) in NEAR_CASES.items()]
    print(f"\n{'case':16s} {'eps':>6s} {'s[-2]':>9s} {'s[-3]':>9s} {'s[-4]':>9s} {'gap k=2':>8s} {'gap k=3':>8s} {'s[-3]/eps':>10s}")
    for name, eps, g in cases:
        cols = columns(g)
        s, _, _, _ = svd_stats(cols, jd, 2, evaluate)
        ratio = s[-3] / eps if eps > 0 else float("nan")
        print(f"{name:16s} {eps:6.2f} {s[-2]:9.1e} {s[-3]:9.1e} {s[-4]:9.1e} {gap_at(s, 2):8.2f} {gap_at(s, 3):8.2f} {ratio:10.2e}")
