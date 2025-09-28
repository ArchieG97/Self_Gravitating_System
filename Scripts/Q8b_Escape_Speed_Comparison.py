import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
G = 1
M = 1
a = 1
r_trunc = 100
n_realizations = 10

# Ensure output directory exists
os.makedirs("Figures_SG", exist_ok=True)

# Analytic escape speed squared
def v_esc_squared(r):
    return 2 * G * M * ((r + a)**2 - r**2) / (a * (r + a)**2)

# Generate particle positions and speeds
def generate_positions_velocities(N):
    u = np.random.uniform(0, 1 - 1e-10, N)
    r = a * ((1 - u) ** (-1/3) - 1)

    theta = np.arccos(1 - 2 * np.random.rand(N))
    phi = 2 * np.pi * np.random.rand(N)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    vx = np.random.normal(0, 1, N)
    vy = np.random.normal(0, 1, N)
    vz = np.random.normal(0, 1, N)

    return r, vx**2 + vy**2 + vz**2

# Binning
r_bins = np.logspace(np.log10(1), np.log10(r_trunc), 25)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# Settings
Ns = [10, 100, 1000, 10000, 100000, 1000000]
colors = ['orange', 'green', 'blue', 'purple', 'brown', 'red']
labels = [
    r'Sampled $N = 10$',
    r'Sampled $N = 10^2$',
    r'Sampled $N = 10^3$',
    r'Sampled $N = 10^4$',
    r'Sampled $N = 10^5$',
    r'Sampled $N = 10^6$'
]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(r_centers, v_esc_squared(r_centers), 'k-', linewidth=2, label=r'$v^2_{\mathrm{esc}}(r) = 2|\Phi(r)|$')

for idx, N in enumerate(Ns):
    v2_all = []

    for _ in range(n_realizations):
        r, v2 = generate_positions_velocities(N)
        bin_means = []
        for i in range(len(r_bins) - 1):
            in_bin = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.any(in_bin):
                bin_means.append(np.mean(v2[in_bin]))
            else:
                bin_means.append(np.nan)
        v2_all.append(bin_means)

    v2_all = np.array(v2_all)
    v2_mean = np.nanmean(v2_all, axis=0)
    v2_std = np.nanstd(v2_all, axis=0)

    # Error bars with transparency
    plt.errorbar(
        r_centers, v2_mean, yerr=v2_std,
        fmt='o-', capsize=3,
        color=colors[idx],
        ecolor=colors[idx], alpha=0.8,  # line opacity
        elinewidth=1, capthick=1,
        errorevery=1,
        label=labels[idx],
        zorder=10 - idx  # Ensure overlap stacking
    )

# Final touches
plt.xscale('log')
plt.xlabel(r'$r$')
plt.ylabel(r'$v^2$')
plt.title(r'$\langle v^2 \rangle$ vs $v^2_{\mathrm{esc}}$ from Sampled Particles for Different $N$')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Figures_SG/Q8b_Escape_Speed_vs_Mean_Speed_All_N_Improved.pdf")
plt.show()
