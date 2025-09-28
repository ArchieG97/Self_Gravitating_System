import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# Output directory (Windows path)
output_dir = "C:\\CATAM_SelfGrav\\Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Parameters
N = 5000          
rT = 100.0        
epsilon_max = 0.5  
a, G, M = 1.0, 1.0, 1.0 
vmax = np.sqrt(2 * epsilon_max)

def phi(r):
    return -0.5 * (((r + a)**2 - r**2) / (r + a)**2)

# Binding energy
def binding_energy(r, v):
    return -(0.5 * v**2 + phi(r))

# Analytic distribution
def f_eps(eps):
    out = np.zeros_like(eps)
    mask = (eps > 0) & (eps < epsilon_max)
    e = eps[mask]
    x = np.sqrt(2 * e / (1 - 2 * e))
    term1 = (3 - 4 * e) * x
    term2 = 3 * np.arcsinh(x)
    out[mask] = (3 / (2 * np.pi**3)) * (term2 - term1)
    return out

# Density of states
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

# Estimate w_max
r_grid = np.linspace(1e-6, rT, 300)
v_grid = np.linspace(0, vmax, 300)
R, V = np.meshgrid(r_grid, v_grid, indexing='ij')
W = R**2 * V**2 * f_eps(binding_energy(R, V))
w_max = np.nanmax(W)
print(f"Estimated w_max: {w_max:.3e}")

# Draw samples
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

# Compute differential energy distribution
eps = binding_energy(r_acc, v_acc)
eps_min = eps.min()
nbins = 20
edges = np.linspace(eps_min, epsilon_max, nbins+1)
hist_e, _ = np.histogram(eps, bins=edges)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
bin_widths = edges[1:] - edges[:-1]
dMde_emp = hist_e * (1.0/N) / bin_widths
dMde_analytic = f_eps(bin_centers) * g_eps(bin_centers)
dMde_analytic /= np.sum(dMde_analytic * bin_widths)

# Plot
plt.figure()
plt.loglog(bin_centers, dMde_analytic, label='Analytic', linewidth=2)
plt.step(bin_centers, dMde_emp, where='mid', label='Sampled', linewidth=2)

ax = plt.gca()
ax.set_xscale('log')

# Desired ticks
ticks = [0.01, 0.05] + [0.1, 0.2, 0.3, 0.5]
ax.set_xticks(ticks)

ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.2f}"))

# Grid and labels
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$dM/d\epsilon$')
plt.title('Differential Energy Distribution (γ=0)')
plt.legend()
plt.tight_layout()

# Save and show
outpath = os.path.join(output_dir, "Q3_energy_distribution_gamma0.pdf")
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved plot to {outpath}")
plt.show()
