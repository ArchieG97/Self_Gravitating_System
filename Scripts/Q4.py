import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = "C:\\CATAM_SelfGrav\\Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Parameters
N = 5000
rT = 100.0
epsilon_max = 0.5
a, G, M = 1.0, 1.0, 1.0
vmax = np.sqrt(2 * epsilon_max)

# Potential
def phi(r):
    return -0.5 * (((r + a)**2 - r**2) / (r + a)**2)

def binding_energy(r, v):
    return -(0.5 * v**2 + phi(r))

def f_eps(eps):
    out = np.zeros_like(eps)
    mask = (eps > 0) & (eps < epsilon_max)
    e = eps[mask]
    x = np.sqrt(2 * e / (1 - 2 * e))
    term1 = (3 - 4 * e) * x
    term2 = 3 * np.arcsinh(x)
    out[mask] = (3 / (2 * np.pi**3)) * (term2 - term1)
    return out

def g_eps(eps):
    out = np.zeros_like(eps)
    mask = (eps > 0) & (eps < epsilon_max)
    e = eps[mask]
    A = np.sqrt(1 - 2*e)**3 * (3 - 14*e - 8*e**2) / (12 * e**2)
    B = -np.pi
    C = ((1 - 6*e + 16*e**2) / ((2*e)**2.5)) * np.arccos(-np.sqrt(1 - 2*e))
    out[mask] = 8 * np.pi**2 * (A + B + C)
    return out

#Phase-space rejection sampling
accept_rate = 0.022
batch = int(N / accept_rate * 1.2)

r_grid = np.linspace(1e-6, rT, 300)
v_grid = np.linspace(0, vmax, 300)
R, V = np.meshgrid(r_grid, v_grid, indexing='ij')
W = R**2 * V**2 * f_eps(binding_energy(R, V))
w_max = np.nanmax(W)
print(f"Estimated w_max: {w_max:.3e}")

r_acc, v_acc = np.empty(0), np.empty(0)
accepted, attempts = 0, 0
while len(r_acc) < N:
    r_block = rT * np.random.rand(batch)
    v_block = vmax * np.random.rand(batch)
    eps_block = binding_energy(r_block, v_block)
    f_block = f_eps(eps_block)
    weights = r_block**2 * v_block**2 * f_block
    rand = w_max * np.random.rand(batch)
    mask = weights > rand
    r_acc = np.concatenate([r_acc, r_block[mask]])
    v_acc = np.concatenate([v_acc, v_block[mask]])
    accepted += mask.sum()
    attempts += batch

r_acc, v_acc = r_acc[:N], v_acc[:N]
print(f"Samples drawn: {len(r_acc)} (target {N})")
print(f"Overall acceptance fraction: {accepted/attempts:.3%}")

#3D
phi_r = 2 * np.pi * np.random.rand(N)
cos_theta_r = 2 * np.random.rand(N) - 1
sin_theta_r = np.sqrt(1 - cos_theta_r**2)

phi_v = 2 * np.pi * np.random.rand(N)
cos_theta_v = 2 * np.random.rand(N) - 1
sin_theta_v = np.sqrt(1 - cos_theta_v**2)

x = r_acc * sin_theta_r * np.cos(phi_r)
y = r_acc * sin_theta_r * np.sin(phi_r)
z = r_acc * cos_theta_r

vx = v_acc * sin_theta_v * np.cos(phi_v)
vy = v_acc * sin_theta_v * np.sin(phi_v)
vz = v_acc * cos_theta_v

positions = np.vstack([x, y, z]).T
velocities = np.vstack([vx, vy, vz]).T

radii = np.sqrt(x**2 + y**2 + z**2)
mass_inside_RT = np.sum(radii <= rT) / N
print(f"Fraction of mass inside r_T={rT}: {mass_inside_RT:.5f} (should be close to M(r_T))")

M_analytic = (rT / (rT + 1))**3
print(f"Analytic M(r_T) = {M_analytic:.5f}")

# Compare M(r_T) for different values
rT_values = [10, 30, 50, 100, 300, 1000]
print("\nAnalytic M(r_T) for various r_T:")
for rt in rT_values:
    m_rt = (rt / (rt + 1))**3
    print(f"r_T = {rt:4d} --> M(r_T) = {m_rt:.6f}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:1000], y[:1000], z[:1000], s=1)
ax.set_title("Sample of Particle Positions")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
outpath = os.path.join(output_dir, "Q4_positions_3d_sample.pdf")
plt.savefig(outpath)
print(f"Saved 3D position plot to {outpath}")
plt.show()
