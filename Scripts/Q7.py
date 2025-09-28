import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
G = M = a = 1
N = 5000
r_T = 100
n_bins = 30
min_count = 10

output_dir = "Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Rejection Sampling 
epsilon_max = 0.5
vmax = np.sqrt(2 * epsilon_max)

# Define potential and binding energy
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

r_grid = np.linspace(1e-6, r_T, 300)
v_grid = np.linspace(0, vmax, 300)
R, V = np.meshgrid(r_grid, v_grid, indexing='ij')
W = R**2 * V**2 * f_eps(binding_energy(R, V))
w_max = np.nanmax(W)

r_acc, v_acc = np.empty(0), np.empty(0)
batch = int(N / 0.022)
np.random.seed(42)
while len(r_acc) < N:
    r_block = r_T * np.random.rand(batch)
    v_block = vmax * np.random.rand(batch)
    eps_block = binding_energy(r_block, v_block)
    f_block = f_eps(eps_block)
    weights = r_block**2 * v_block**2 * f_block
    rand = w_max * np.random.rand(batch)
    mask = weights > rand
    r_acc = np.concatenate([r_acc, r_block[mask]])
    v_acc = np.concatenate([v_acc, v_block[mask]])

r_acc = r_acc[:N]
v_acc = v_acc[:N]

# Convert to Cartesian Coordinates
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

# Compute Anisotropy Profile
r_bins = np.logspace(np.log10(0.05), np.log10(r_T), n_bins + 1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
beta_profile = []
beta_masked = []

print("Computing anisotropy in radial bins...")

for i in range(n_bins):
    in_bin = (radii >= r_bins[i]) & (radii < r_bins[i + 1])
    count = np.sum(in_bin)

    if count == 0:
        beta_profile.append(np.nan)
        beta_masked.append(np.nan)
        print(f"Bin {i}: r = [{r_bins[i]:.2f}, {r_bins[i+1]:.2f}] -> 0 particles")
        continue

    r_vecs = positions[in_bin]
    v_vecs = velocities[in_bin]
    r_hat = r_vecs / np.linalg.norm(r_vecs, axis=1, keepdims=True)
    v_r = np.sum(v_vecs * r_hat, axis=1)
    v_r2_mean = np.mean(v_r**2)
    v2_mean = np.mean(np.sum(v_vecs**2, axis=1))
    v_t2_mean = v2_mean - v_r2_mean

    beta = 1 - v_t2_mean / (2 * v_r2_mean)
    beta_profile.append(beta)
    beta_masked.append(beta if count >= min_count else np.nan)
    print(f"Bin {i}: count = {count}, beta = {beta:.4f}" + (" (masked)" if count < min_count else ""))

beta_profile = np.array(beta_profile)
beta_masked = np.array(beta_masked)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r_centers, beta_masked, 'o-', color='darkorange', label=r"$\beta(r)$ (trusted)")
plt.plot(r_centers, beta_profile, 'o', color='gray', alpha=0.3, label=r"$\beta(r)$ (all)")
plt.axhline(0, color='gray', linestyle='--')
plt.ylim(-1, 1)
plt.xlabel("Radius $r$ (dimensionless)")
plt.ylabel(r"$\beta(r)$")
plt.title(r"Velocity Anisotropy $\beta(r)$")
plt.grid(True, ls='--')
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, "Q7_Anisotropy_Profile_Combined.pdf")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Saved anisotropy profile plot to: {plot_path}")
