"""Week 3 Part C, step 2: the symbolic symmetry criterion, one column per basis field.

Equation:  Delta = u_t + u u_x - nu u_xx - g(u) = 0
Fill every `...` marked TODO (blanks C1 to C5), then run from the repo root:

    python -m discovery.criterion

All self-checks at the bottom must print ✓ before you move on to fingerprint.py.
"""
import sympy as sp

# --- jet coordinates (independent symbols) ---------------------------------------
x, t, u = sp.symbols("x t u")
ux, ut, uxx, uxt, uxxx = sp.symbols("u_x u_t u_xx u_xt u_xxx")
JET = (x, t, u, ux, uxx, uxxx)          # what the columns depend on after restricting to solutions

NU = sp.Rational(1, 10)                  # must match the nu of your data (0.1). Rational keeps zeros exact.

# --- the 30 basis fields ----------------------------------------------------------
MONOMIALS  = [sp.Integer(1), x, t, u, x**2, t**2, u**2, x*t, x*u, t*u]
MONO_NAMES = ["1", "x", "t", "u", "x^2", "t^2", "u^2", "xt", "xu", "tu"]


def basis_fields():
    """Return (fields, names): 30 triples (xi, tau, phi) and their names.

    Ordering: columns 0-9 put a monomial in xi, 10-19 in tau, 20-29 in phi.
    Example: column 13 is the field with tau = u (and xi = phi = 0), named "tau:u"."""
    fields, names = [], []
    for comp in range(3):                         # 0: xi, 1: tau, 2: phi
        for m, mname in zip(MONOMIALS, MONO_NAMES):
            triple = [sp.Integer(0)] * 3
            # C1: put the monomial m in the right component of the triple
            triple[comp] = m                                    # TODO (one line)
            fields.append(tuple(triple))
            names.append(f"{['xi', 'tau', 'phi'][comp]}:{mname}")
    return fields, names


# --- total derivatives ------------------------------------------------------------
def Dx(f):
    """Total x-derivative, enough for functions of (x, t, u, u_x, u_t).

    C2: one partial derivative per jet coordinate f can depend on, each multiplied by
        "the next derivative": u -> u_x, u_x -> u_xx, u_t -> u_xt.
        (Same pattern as Dx in toy lesson 2, with one more term for u_t.)"""
    return sp.diff(f,x) + ux*sp.diff(f,u) + uxt*sp.diff(f,ut) + uxx*sp.diff(f,ux)                                    # TODO


def Dt(f):
    """Total t-derivative, only applied to xi, tau, phi (functions of x, t, u only).

    C3: two terms. Which jet coordinate does d/dt turn u into?"""
    return sp.diff(f,t) + ut*sp.diff(f,u) + uxt*sp.diff(f,ux)                                  # TODO


# --- prolongation and criterion ---------------------------------------------------
def criterion(xi, tau, phi, g):
    """pr v(Delta) for Delta = u_t + u u_x - nu u_xx - g(u), before restricting to solutions.

    C4: the formulas from your Week 2 Part C derivation (also in the status file, section 7):
        phi^x  = D_x phi   - u_x  D_x xi - u_t  D_x tau
        phi^t  = D_t phi   - u_x  D_t xi - u_t  D_t tau
        phi^xx = D_x phi^x - u_xx D_x xi - u_xt D_x tau
        pr v(Delta) = phi^t + u phi^x + u_x phi - nu phi^xx - g'(u) phi
    Hint: g'(u) is sp.diff(g, u)."""
    phi_x  = Dx(phi) - ux*Dx(xi) - ut*Dx(tau)     # TODO
    phi_t  = Dt(phi) - ux*Dt(xi) - ut*Dt(tau)     # TODO
    phi_xx = Dx(phi_x) - uxx*Dx(xi) - uxt*Dx(tau)   # TODO
    return phi_t + u*phi_x + ux*phi - NU*phi_xx - sp.diff(g,u)*phi                                 # TODO


def on_solutions(expr, g):
    """Replace u_t and u_xt using the equation, so only (x, t, u, u_x, u_xx, u_xxx) remain.

    C5: rhs   = the right-hand side of u_t = ...   (same as your rhs_burgers, with nu = NU)
        rhs_x = the total x-derivative of rhs. rhs depends on x, u, u_x, u_xx, so you need
                the terms for u -> u_x, u_x -> u_xx, u_xx -> u_xxx (Dx above stops at u_t,
                so write this one out).
    Then substitute with expr.subs({uxt: rhs_x, ut: rhs}) and sp.expand the result
    (expanding makes true-symmetry columns simplify to exactly 0)."""
    rhs   = -u*ux + NU*uxx + g                                   # TODO
    rhs_x = -ux**2 - u*uxx + NU*uxxx + sp.diff(g, u)*ux          # TODO
    return expr.subs({uxt: rhs_x, ut: rhs}).expand()                                    # TODO


def columns(g):
    """The 30 symbolic columns (restricted to solutions) for a given g(u)."""
    fields, _ = basis_fields()
    return [on_solutions(criterion(*f, g), g) for f in fields]


# --- self-checks (do not change) -------------------------------------------------
def _check(name, ok):
    print(("✓ " if ok else "✗ ") + name)


if __name__ == "__main__":
    g0, g3 = sp.Integer(0), -u**3
    zero = lambda e: sp.simplify(e) == 0

    fields, names = basis_fields()
    _check("30 basis fields; column 13 is tau = u", len(fields) == 30 and fields[13] == (0, u, 0)
           and names[13] == "tau:u")
    _check("d/dt (tau = 1) gives exactly 0",
           zero(on_solutions(criterion(0, 1, 0, g0), g0)))
    _check("Galilean (xi = t, phi = 1), before substitution, equals -g'(u) for g = -u^3",
           zero(criterion(t, 0, 1, g3) - (-sp.diff(g3, u))))
    _check("Galilean is a symmetry for g = 0",
           zero(on_solutions(criterion(t, 0, 1, g0), g0)))
    _check("scaling (xi = x, tau = 2t, phi = -u) is a symmetry for g = 0",
           zero(on_solutions(criterion(x, 2*t, -u, g0), g0)))
    _check("scaling is a symmetry for g = -u^3",
           zero(on_solutions(criterion(x, 2*t, -u, g3), g3)))
    _check("projective (xi = xt, tau = t^2, phi = x - tu) is a symmetry for g = 0",
           zero(on_solutions(criterion(x*t, t**2, x - t*u, g0), g0)))
    _check("projective is NOT a symmetry for g = -u^3",
           not zero(on_solutions(criterion(x*t, t**2, x - t*u, g3), g3)))

    cols = columns(g0)
    n_zero = sum(1 for c in cols if c == 0)
    _check(f"g = 0: exactly 2 identically-zero columns (d/dx, d/dt); found {n_zero}", n_zero == 2)
    _check("after on_solutions, no u_t or u_xt is left in any column",
           all(not (c.has(ut) or c.has(uxt)) for c in cols))
