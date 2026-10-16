import sympy as sp
x, t, u = sp.symbols('x t u')
u_x, u_t, u_xx, u_xt, u_tt = sp.symbols('u_x u_t u_xx u_xt u_tt')
u_xxx, u_xxt, u_xtt = sp.symbols('u_xx u_xxt u_xtt')
def Dx(f):
    return (sp.diff(f,x) + u_x*sp.diff(f,u) + u_xx*sp.diff(f,u_x) + u_xt*sp.diff(f,u_t) + u_xxx*sp.diff(f,u_xx) + u_xxt*sp.diff(f,u_xt))
def Dt(f):
    return (sp.diff(f,t) + u_t*sp.diff(f,u) + u_xt*sp.diff(f,u_x) + u_tt*sp.diff(f,u_t) + u_xtt*sp.diff(f,u_xt) + u_xxt*sp.diff(f,u_xx))

def prolong (xi, tau, phi):
    phi_x = Dx(phi) - u_x*Dx(xi) - u_t*Dx(tau)
    phi_t = Dt(phi) - u_x*Dt(xi) - u_t*Dt(tau)
    phi_xx = Dx(phi_x) - u_xx*Dx(xi) - u_xt*Dx(tau)
    return sp.expand(phi_x), sp.expand(phi_t), sp.expand(phi_xx)

def burgers_criterion(xi,tau,phi):
    nu = sp.Symbol('nu')
    phi_x ,phi_t , phi_xx = prolong(xi, tau, phi)
    crit = phi_t + u_x*phi + u*phi_x - nu*phi_xx
    return sp.simplify(crit.subs(u_t, -u*u_x + nu*u_xx))
print("burgers criterions:")
print(burgers_criterion(1,0,0)) # translation -> 0
print(burgers_criterion(t,0,1)) # Galilean -> 0
print(burgers_criterion(x,2*t,-u)) # scaling -> 0
print(burgers_criterion(x*t,t**2,x-t*u))
print(burgers_criterion(x,t,0))
def heat_criterion(xi, tau, phi):
    phi_x ,phi_t, phi_xx = prolong(xi, tau, phi)
    crit = phi_t - phi_xx
    return sp.simplify(crit.subs(u_t,u_xx))
print("heat criterions:")
print(heat_criterion(1,0,0)) # space translation -> 0
print(heat_criterion(0,1,0)) # time translation -> 0
print(heat_criterion(0,0,u)) # scaling of u -> 0
print(heat_criterion(x,2*t,0)) # scaling of x and t -> 0
print(heat_criterion(2*t,0,-x*u)) # Galilean -> 0
print(heat_criterion(4*t*x,4*t**2,-(x**2+2*t)*u)) # projective -> 0
