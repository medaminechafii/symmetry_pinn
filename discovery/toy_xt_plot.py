import numpy as np

nu, N, T = 0.1, 256, 2.0

x = np.linspace(0, 2*np.pi, N, endpoint=False)
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
#runge kutta 4th order time stepping
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
u = np.sin(x)
U = [u.copy()]
#dt<(2*np.pi/N)/np.max(np.abs(u)) # CFL condition

# 2 --- time stepping loop ---
save_every = 10
for n in range(1,steps):
    u = rk4_step(u,dt,lambda u: 0*u)
    if n % save_every == 0:
        U.append(u.copy())
U = np.array(U)
t_saved = t[::save_every]
print(steps, dt, dt_max, U.shape, t_saved.shape, t_saved[-1])
err = np.max(np.abs(dx(np.sin(3*x)) - 3*np.cos(3*x)))
print("derivative error:", err)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 3 --- Left: heat map of u(x, t) ---
ax = axes[0]
pc = ax.pcolormesh(x, t_saved, U, shading="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)          # fix colour range so 0 is white
fig.colorbar(pc, ax=ax, label="u")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("Burgers: u(x, t)")

# --- Right: snapshots at a few times ---
ax = axes[1]
idx = np.linspace(0, len(t_saved) - 1, 5).astype(int)   # 5 evenly spaced rows
for i in idx:
    ax.plot(x, U[i], label=f"t = {t_saved[i]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("Snapshots")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("burgers_g0.png", dpi=150)   # saves the image next to your script
plt.show()