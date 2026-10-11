"""Symbolic checks: prolongation formula, Burgers determining equations, finite transformations."""
import sympy as sp, json
x, t, u, ux, ut, uxx, uxt, utt = sp.symbols('x t u u_x u_t u_xx u_xt u_tt')
uxxx, uxxt, uxtt, uttt = sp.symbols('u_xxx u_xxt u_xtt u_ttt')
nu = sp.symbols('nu', positive=True)

def Dx(f):
    return (sp.diff(f, x) + ux*sp.diff(f, u) + uxx*sp.diff(f, ux) + uxt*sp.diff(f, ut)
            + uxxx*sp.diff(f, uxx) + uxxt*sp.diff(f, uxt) + uxtt*sp.diff(f, utt))
def Dt(f):
    return (sp.diff(f, t) + ut*sp.diff(f, u) + uxt*sp.diff(f, ux) + utt*sp.diff(f, ut)
            + uxxt*sp.diff(f, uxx) + uxtt*sp.diff(f, uxt) + uttt*sp.diff(f, utt))

def prolong(xi, tau, phi):
    px = Dx(phi) - ux*Dx(xi) - ut*Dx(tau)
    pt = Dt(phi) - ux*Dt(xi) - ut*Dt(tau)
    pxx = Dx(px) - uxx*Dx(xi) - uxt*Dx(tau)
    return sp.expand(px), sp.expand(pt), sp.expand(pxx)

out = {}
# 1. check the expanded textbook formula for phi^xx with general xi,tau,phi(x,t,u)
XI, TAU, PHI = [sp.Function(n)(x, t, u) for n in ('xi', 'tau', 'phi')]
px, pt, pxx = prolong(XI, TAU, PHI)
d = lambda f, *v: sp.diff(f, *v)
pxx_book = (d(PHI,x,x) + (2*d(PHI,x,u) - d(XI,x,x))*ux - d(TAU,x,x)*ut + (d(PHI,u,u) - 2*d(XI,x,u))*ux**2
            - 2*d(TAU,x,u)*ux*ut - d(XI,u,u)*ux**3 - d(TAU,u,u)*ux**2*ut + (d(PHI,u) - 2*d(XI,x))*uxx
            - 2*d(TAU,x)*uxt - 3*d(XI,u)*ux*uxx - d(TAU,u)*ut*uxx - 2*d(TAU,u)*ux*uxt)
out["phi_xx_formula_matches"] = sp.simplify(pxx - pxx_book) == 0

# 2. Burgers determining equations with a general quadratic ansatz
Delta = ut + u*ux - nu*uxx
mons = [sp.Integer(1), x, t, u, x**2, t**2, u**2, x*t, x*u, t*u]
cs = sp.symbols('c0:30')
xi = sum(c*m for c, m in zip(cs[0:10], mons)); tau = sum(c*m for c, m in zip(cs[10:20], mons))
phi = sum(c*m for c, m in zip(cs[20:30], mons))
px, pt, pxx = prolong(xi, tau, phi)
crit = pt + phi*ux + u*px - nu*pxx                 # pr v (Delta)
crit = sp.expand(crit.subs(ut, -u*ux + nu*uxx))  # on the surface Delta = 0
eqs = sp.Poly(crit, x, t, u, ux, uxx, uxt).coeffs()
sol = sp.solve(eqs, cs, dict=True)[0]
free = sorted(set().union(*[sp.sympify(v).free_symbols for v in sol.values()]) - {nu} | (set(cs) - set(sol)), key=str)
gens = []
for f in free:
    sub = {c: (1 if c == f else 0) for c in free}
    g = [sp.factor(sp.expand(e.subs(sol).subs(sub))) for e in (xi, tau, phi)]
    gens.append([str(v) for v in g])
out["burgers_dim"] = len(free); out["burgers_generators_(xi,tau,phi)"] = gens

# 3. prolongations of the 5 Burgers generators
G = {"dx": (1, 0, 0), "dt": (0, 1, 0), "galilean": (t, 0, 1),
     "scaling": (x, 2*t, -u), "projective": (x*t, t**2, x - t*u)}
out["prolongations"] = {k: [str(sp.simplify(e)) for e in prolong(*map(sp.sympify, v))] for k, v in G.items()}
out["criterion_check"] = {k: str(sp.simplify(sp.expand((lambda p: p[1] + v[2]*ux + u*p[0] - nu*p[2])(prolong(*map(sp.sympify, v))).subs(ut, -u*ux + nu*uxx))))
                          for k, v in G.items()}

# 4. finite transformations mapping an exact solution to new exact solutions
a, k, eps, X, T = sp.symbols('a k epsilon X T', real=True)
th = 2 + a*sp.cos(k*x)*sp.exp(-nu*k**2*t)            # heat solution
u0 = -2*nu*sp.diff(th, x)/th                           # Cole-Hopf: Burgers solution
res = lambda w: sp.simplify(sp.diff(w, T) + w*sp.diff(w, X) - nu*sp.diff(w, X, 2))
U = lambda xx, tt: u0.subs({x: xx, t: tt})
new = {
 "dx":        U(X - eps, T),
 "dt":        U(X, T - eps),
 "galilean":  U(X - eps*T, T) + eps,
 "scaling":   sp.exp(-eps)*U(sp.exp(-eps)*X, sp.exp(-2*eps)*T),
 "projective": (U(X/(1 + eps*T), T/(1 + eps*T)) + eps*X)/(1 + eps*T),
}
out["transformed_solution_residuals"] = {kk: str(res(w)) for kk, w in new.items()}
print(json.dumps(out, indent=1))
json.dump(out, open("symbolic_results.json", "w"), indent=1)
