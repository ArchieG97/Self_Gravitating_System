import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
G = 1
M = 1
a = 1
r_trunc = 100

# Ensure output directory exists
os.makedirs("Figures_SG", exist_ok=True)

# Analytic escape speed squared
def v_esc_squared(r):
    return 2 * G * M * ((r + a)**2 - r**2) / (a * (r + a)**2)

# Generate Plummer-distributed positions and velocities
def generate_positions_velocities(N):
    u = np.random.uniform(0, 1 - 1e-10, N)
    r = a * ((1 - u) ** (-1/3) - 1)

    theta = np.arccos(1 - 2 * np.random.rand(N))
    phi = 2 * np.pi * np.random.rand(N)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Maxwellian velocity distribution
    vx = np.random.normal(0, 1, N)
    vy = np.random.normal(0, 1, N)
    vz = np.random.normal(0, 1, N)

    return np.column_stack((x, y, z)), np.column_stack((vx, vy, vz)), r

# Radial bins
r_bins = np.logspace(np.log10(1), np.log10(r_trunc), 25)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# Parameters for the experiment
Ns = [10, 100, 1000, 10000, 100000, 1000000]
n_realizations = 10
colors = ['orange', 'green', 'blue', 'purple', 'brown', 'red']
labels = [
    r'Sampled $N = 10$',
    r'Sampled $N = 10^2$',
    r'Sampled $N = 10^3$',
    r'Sampled $N = 10^4$',
    r'Sampled $N = 10^5$',
    r'Sampled $N = 10^6$']

# Store results
v2_means = []
v2_stds = []

for N in Ns:
    v2_samples = []

    for _ in range(n_realizations):
        positions, velocities, radii = generate_positions_velocities(N)
        v_mag_sq = np.sum(velocities**2, axis=1)

        bin_means = []
        for i in range(len(r_bins)-1):
            in_bin = (radii >= r_bins[i]) & (radii < r_bins[i+1])
            if np.any(in_bin):
                bin_means.append(np.mean(v_mag_sq[in_bin]))
            else:
                bin_means.append(np.nan)
        v2_samples.append(bin_means)

    v2_samples = np.array(v2_samples)
    v2_means.append(np.nanmean(v2_samples, axis=0))
    v2_stds.append(np.nanstd(v2_samples, axis=0))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(r_centers, v_esc_squared(r_centers), 'k-', linewidth=2, label=r'$v^2_{\mathrm{esc}}(r) = 2|\Phi(r)|$')

for i in range(len(Ns)):
    plt.errorbar(r_centers, v2_means[i], yerr=v2_stds[i], fmt='o-', capsize=3,
                 color=colors[i], label=labels[i])

plt.xscale('log')
plt.xlabel('$r$')
plt.ylabel('$v^2$')
plt.title('Mean Squared Speed vs Escape Speed Squared for Different $N$')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Figures_SG/Q8_Escape_Speed_Comparison_Ns.pdf")
plt.show()
