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

# Analytic Plummer-like potential
def analytic_phi(r):
    return -G * M / 2 * ((r + a)**2 - r**2) / (a * (r + a)**2)

# Generate Plummer-distributed positions
def generate_positions(N):
    u = np.random.uniform(0, 1 - 1e-10, N)
    r = a * ((1 - u) ** (-1/3) - 1)

    theta = np.arccos(1 - 2 * np.random.rand(N))
    phi = 2 * np.pi * np.random.rand(N)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z)), r

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
phi_means = []
phi_stds = []

for N in Ns:
    phi_samples = []

    for _ in range(n_realizations):
        positions, radii = generate_positions(N)
        mask = radii < r_bins[-1]
        radii = radii[mask]

        mass_per_particle = M / N
        hist, _ = np.histogram(radii, bins=r_bins)
        cumulative_mass = np.cumsum(hist) * mass_per_particle
        phi_sampled = -G * cumulative_mass / r_centers
        phi_samples.append(phi_sampled)

    phi_samples = np.array(phi_samples)
    phi_means.append(np.mean(phi_samples, axis=0))
    phi_stds.append(np.std(phi_samples, axis=0))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(r_centers, analytic_phi(r_centers), 'k-', linewidth=2, label='Analytic')

for i in range(len(Ns)):
    plt.errorbar(r_centers, phi_means[i], yerr=phi_stds[i], fmt='o-', capsize=3,
                 color=colors[i], label=labels[i])

plt.xlabel('$r$')
plt.ylabel('$\\Phi(r)$')
plt.title('Gravitational Potential for Different $N$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Figures_SG/Q8_Potential_Comparison_Ns_ErrorBars.pdf")
plt.show()
