import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
G = M = a = 1
N = 5000
r_T = 100
n_bins = 30

# Output directory
output_dir = "Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Generate Sample Positions and Velocities
np.random.seed(42)

r = np.random.power(3, N) * r_T
theta = np.arccos(1 - 2 * np.random.rand(N))
phi = 2 * np.pi * np.random.rand(N)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
positions = np.vstack((x, y, z)).T

v_mag = np.random.uniform(0, np.sqrt(1), N)
theta_v = np.arccos(1 - 2 * np.random.rand(N))
phi_v = 2 * np.pi * np.random.rand(N)
vx = v_mag * np.sin(theta_v) * np.cos(phi_v)
vy = v_mag * np.sin(theta_v) * np.sin(phi_v)
vz = v_mag * np.cos(theta_v)
velocities = np.vstack((vx, vy, vz)).T

radii = np.linalg.norm(positions, axis=1)
speeds_squared = np.sum(velocities**2, axis=1)

r_bins = np.logspace(np.log10(0.5), np.log10(r_T), n_bins + 1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

# Initialize Output Lists
L_magnitudes = []
L_normalized = []
sigma2_numeric = []
sigma2_analytic = []
counts = []

# Loop Over Radial Bins
for i in range(n_bins):
    in_bin = (radii >= r_bins[i]) & (radii < r_bins[i + 1])
    r_vecs = positions[in_bin]
    v_vecs = velocities[in_bin]
    v2s = speeds_squared[in_bin]
    r_mid = r_centers[i]
    counts.append(np.sum(in_bin))

    # Angular momentum
    if len(r_vecs) == 0:
        L_vec = np.array([0.0, 0.0, 0.0])
        L_norm = 0.0
        sigma2 = np.nan
    else:
        L_individual = np.cross(r_vecs, v_vecs)
        L_vec = np.sum(L_individual, axis=0)
        L_total = np.linalg.norm(L_vec)
        L_scalar_sum = np.sum(np.linalg.norm(L_individual, axis=1))
        L_norm = L_total / L_scalar_sum if L_scalar_sum > 0 else 0.0
        sigma2 = np.mean(v2s)

    L_magnitudes.append(np.linalg.norm(L_vec))
    L_normalized.append(L_norm)
    sigma2_numeric.append(sigma2)
    sigma2_analytic.append(G * M * (a + 6*r_mid) / (10 * (r_mid + a)**2))

# Convert to Arrays
L_magnitudes = np.array(L_magnitudes)
L_normalized = np.array(L_normalized)
sigma2_numeric = np.array(sigma2_numeric)
sigma2_analytic = np.array(sigma2_analytic)
counts = np.array(counts)

# Plot: Angular Momentum
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

ax1.plot(r_centers, L_magnitudes, 'o--', color='navy', markersize=5, label=r'$|\mathbf{L}(r)|$')
ax1.set_ylabel(r"$|\mathbf{L}(r)|$", fontsize=12)
ax1.set_title("Angular Momentum Magnitude in Radial Shells", fontsize=14)
ax1.grid(True, ls='--')
ax1.legend()

ax2.plot(r_centers, L_normalized, 'o--', color='darkred', markersize=5, label="Coherence Ratio")
ax2.set_xlabel("$r$", fontsize=12)
ax2.set_ylabel("Coherence Ratio", fontsize=12)
ax2.set_title("Angular Momentum Directional Coherence", fontsize=14)
ax2.set_xscale("log")
ax2.grid(True, ls='--')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Q6_Combined_Angular_Momentum_Plots.pdf"), dpi=300)
plt.show()

#Plot: Velocity Dispersion
plt.figure(figsize=(8, 5))
plt.plot(r_centers, sigma2_numeric, 'o-', label=r"Numerical $\sigma^2(r)$", color='black')
plt.plot(r_centers, sigma2_analytic, '--', label=r"Analytic $\sigma^2(r)$", color='blue')
plt.xlabel("$r$")
plt.ylabel(r"$\sigma^2(r)$")
plt.title("Velocity Dispersion Comparison")
plt.grid(True, ls='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Q6_Velocity_Dispersion_Comparison.pdf"), dpi=300)
plt.show()

# Print Outputs for Verification
print("\nRadial Bin Centers:")
print(r_centers)
print("\nUn-normalised Angular Momentum Magnitudes:")
print(L_magnitudes)
print("\nCoherence Ratios:")
print(L_normalized)
print("\nParticle Counts per Bin:")
print(counts)
print("\nNumeric Dispersion:")
print(sigma2_numeric)
print("\nAnalytic Dispersion:")
print(sigma2_analytic)