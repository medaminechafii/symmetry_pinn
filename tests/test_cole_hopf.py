import numpy as np
import matplotlib.pyplot as plt
a = [0.5,0.2,0.1]
x = np.linspace(0, 2*np.pi, 256, endpoint=False)
nu = 0.1
def theta(x,t):
    total = np.ones_like(x)
    for n,cn in enumerate(a,start=1):
        total += cn*np.exp(-nu*n**2*t)*np.cos(n*x)
    return total
def theta_x(x,t):
    total = np.zeros_like(x)
    for n,cn in enumerate(a,start=1):
        total += -cn*n*np.exp(-nu*n**2*t)*np.sin(n*x)
    return total
def u_exact(x,t):
    return -2*nu*theta_x(x,t)/theta(x,t)
# ---spectral derivatives using FFT ---

nu, N, T = 0.1, 256, 2.0

k = np.fft.fftfreq(N, 2*np.pi/N)*2*np.pi

# 1 --- FFT and RK4 ---

#derivation using fft 
def dx(u):
    return np.fft.ifft(1j*k*np.fft.fft(u)).real
def d_xx(u):
    return np.fft.ifft(-k**2*np.fft.fft(u)).real

#right hand side of the equation
def rhs_burgers(u,g):
    return -u*dx(u) + nu*d_xx(u) + g(u)

t0 ,h = 1.0, 1e-4
u = u_exact(x,t0)
u_t = (u_exact(x,t0+h) - u_exact(x,t0-h))/(2*h)
u_x = dx(u)
u_xx = d_xx(u)
res_exact = u_t + u*u_x - nu*u_xx
def rk4_step(u,dt,g):
    k1 = rhs_burgers(u,g)
    k2 = rhs_burgers(u + 0.5*dt*k1,g)
    k3 = rhs_burgers(u + 0.5*dt*k2,g)
    k4 = rhs_burgers(u + dt*k3,g)
    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)
#constraint on the time step for stability of the RK4 method
lambda_rk = -nu*(N/2)**2
dt_max = 2.785/np.abs(lambda_rk)
steps = int(np.ceil(T/(dt_max*0.5)))
dt = T/steps
t = np.linspace(0,T,steps+1)
#dt<(2*np.pi/N)/np.max(np.abs(u)) # CFL condition

# 2 --- time stepping loop ---
save_every = 10
def solve_burgers_saved(u0,steps,g):
    dt = T/steps
    u = u0.copy()
    U = [u.copy()]
    for n in range(1,steps+1):
        u = rk4_step(u,dt,g)
        if n % save_every == 0:
            U.append(u.copy())
    return np.array(U)
rng = np.random.default_rng(0)
def random_ic(rng):
    A = rng.uniform(-0.5,0.5,size=3)
    phi = rng.uniform(0,2*np.pi,size=3)
    u0 = np.zeros_like(x)
    for j in range(3):
        u0 += A[j]*np.sin((j+1)*x+phi[j])
    return u0
g_cases = {"g0":lambda u: 0*u, "cubic": lambda u: -u**3}
 
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the script's folder, then one level up
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True) 

for name,g in g_cases.items():
    U_all = []
    for i in range(3):
        rng = np.random.default_rng(i)
        u0 = random_ic(rng)
        U = solve_burgers_saved(u0,steps,g)
        U_all.append(U)
        t_saved = t[::save_every]
    np.savez(DATA / f"dense_{name}.npz", x=x, t=t_saved, U=U_all, nu=nu, seeds=np.arange(3))
    



