import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
G = M = a = 1
N = 5000
r_T = 100
n_bins = 30
particle_mass = M / N

#Output directory
output_dir = "Figures_SG"
os.makedirs(output_dir, exist_ok=True)

#Monte Carlo sample of positions
np.random.seed(42)
r = np.random.power(3, N) * r_T
theta = np.arccos(1 - 2 * np.random.rand(N))
phi = 2 * np.pi * np.random.rand(N)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
positions = np.vstack((x, y, z)).T
radii = np.linalg.norm(positions, axis=1)

#Define spherical radial bins
r_bins = np.logspace(np.log10(0.5), np.log10(r_T), n_bins + 1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

#Compute enclosed mass
M_enclosed = []
for r_val in r_centers:
    M_enc = particle_mass * np.sum(radii <= r_val)
    M_enclosed.append(M_enc)
M_enclosed = np.array(M_enclosed)

#Numerical potential using shell theorem: Phi(r) ≈ -G * M(<r) / r
phi_numerical = -G * M_enclosed / r_centers

#Analytic potential
phi_analytic = -G * M / (2 * a) * ((r_centers + a)**2 - r_centers**2) / (r_centers + a)**2

#Plot
plt.figure(figsize=(8, 5))
plt.plot(r_centers, phi_numerical, 'o', label="Sampled (Enclosed Mass)", color='darkorange')
plt.plot(r_centers, phi_analytic, '-', label="Analytic", color='black')
plt.xlabel("$r$")
plt.ylabel("$\\Phi(r)$")
plt.title("Gravitational Potential Profile (Enclosed Mass Approximation)")
plt.legend()
plt.grid(True, ls='--')
plt.tight_layout()

#Save plot
plot_path = os.path.join(output_dir, "Q8_Potential_Profile_Corrected.pdf")
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
