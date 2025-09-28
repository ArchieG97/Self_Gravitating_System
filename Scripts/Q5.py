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

# Model functions
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

# Convert to 3D Cartesian
phi_r = 2 * np.pi * np.random.rand(N)
cos_theta_r = 2 * np.random.rand(N) - 1
sin_theta_r = np.sqrt(1 - cos_theta_r**2)

x = r_acc * sin_theta_r * np.cos(phi_r)
y = r_acc * sin_theta_r * np.sin(phi_r)
z = r_acc * cos_theta_r

radii = np.sqrt(x**2 + y**2 + z**2)

# Density profile for different bin counts 
bin_counts = [5, 20, 100, 500]  
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.flatten()

r_plot = np.logspace(np.log10(0.1), np.log10(rT), 300)
rho_analytic = 3 / (4 * np.pi * (r_plot + a)**4)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for idx, (nbins, color) in enumerate(zip(bin_counts, colors)):
    bins = np.linspace(0, rT, nbins + 1)
    hist, edges = np.histogram(radii, bins=bins)
    vol_shell = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_emp = hist * (1.0 / N) / vol_shell
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    mask = rho_emp > 0
    axs[idx].plot(r_mid[mask], rho_emp[mask], color=color, label=f"{nbins} bins")
    axs[idx].plot(r_plot, rho_analytic, 'k--', linewidth=2, label='Analytic')
    axs[idx].set_xscale('log')
    axs[idx].set_yscale('log')
    axs[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[idx].legend()

for ax in axs:
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho(r)$")

fig.suptitle("Density Profile: Effect of Bin Size", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
outpath = os.path.join(output_dir, "Q5_density_bin_subplots.pdf")
plt.savefig(outpath)
print(f"Saved subplot comparison plot to {outpath}")
plt.show()

#Histogram of particles per bin for highest resolution
bins = np.linspace(0, rT, 501)
hist, _ = np.histogram(radii, bins=bins)
r_mid = 0.5 * (bins[:-1] + bins[1:])

plt.figure(figsize=(8, 5))
plt.plot(r_mid, hist, drawstyle='steps-mid')
plt.xlabel(r"$r$")
plt.ylabel("Particles per bin")
plt.title("Particle Count per Bin (500 bins)")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

outpath_hist = os.path.join(output_dir, "Q5_particle_counts_histogram.pdf")
plt.savefig(outpath_hist)
print(f"Saved particle count histogram to {outpath_hist}")
plt.show()
