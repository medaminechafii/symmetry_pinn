import numpy as np
from discovery.noise_gap import ALL_CASES, EPS, column_raw, evaluate7, clean_jet
from discovery.fingerprint import normalise, DATA

d = np.load(DATA / "dense_cubic.npz")
for name in ["cubic"] + [f"cubic_{tag}" for tag in EPS]:
    g, k_exp = ALL_CASES[name]
    cols = column_raw(g)                       # sympy is slow: once per g
    jet = clean_jet(d, g, np.random.default_rng(0))
    A = np.column_stack([evaluate7(c,jet) for c in cols])   # hint: same call as in noise_gap's __main__
    A_n, scale, is_zero = normalise(A)       # hint: normalise returns three things, see discover_from_surrogate
    s = np.linalg.svd(A_n, compute_uv=False)
    print(name, np.array2string(s[-5:], precision=2))     # the smallest five singular values