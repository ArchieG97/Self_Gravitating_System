import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
G = 1
M = 1
a = 1
r_trunc = 100
n_realisations = 5

# Output directory
os.makedirs("Figures_SG", exist_ok=True)

# Analytic potential and escape speed squared
def analytic_phi(r):
    return -G * M / (r + a) + G * M * a / (2 * (r + a)**2)

def v_esc_squared(r):
    return 2 * np.abs(analytic_phi(r))

# Particle generator
def generate_sample(N):
    u = np.random.uniform(0, 1 - 1e-10, N)
    r = a * ((1 - u) ** (-1/3) - 1)

    theta = np.arccos(1 - 2 * np.random.rand(N))
    phi = 2 * np.pi * np.random.rand(N)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    radii = np.sqrt(x**2 + y**2 + z**2)

    vx = np.random.normal(0, 1, N)
    vy = np.random.normal(0, 1, N)
    vz = np.random.normal(0, 1, N)
    v2 = vx**2 + vy**2 + vz**2

    return radii, v2

# Binning
r_bins = np.logspace(np.log10(1), np.log10(r_trunc), 30)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# Ns to test
Ns = [10, 100, 1000, 10000, 100000, 1000000]
phi_devs = []
v2_devs = []

for N in Ns:
    phi_deviation_real = []
    v2_deviation_real = []

    for _ in range(n_realisations):
        radii, v2 = generate_sample(N)

        M_r = []
        phi_r = []
        v2_mean = []

        for i in range(len(r_bins) - 1):
            in_bin = (radii >= r_bins[i]) & (radii < r_bins[i+1])
            if np.any(in_bin):
                r_mid = r_centers[i]
                mass_enclosed = len(radii[in_bin]) / N * M
                phi_est = -G * mass_enclosed / r_mid
                phi_r.append(phi_est)

                v2_mean.append(np.mean(v2[in_bin]))
            else:
                phi_r.append(np.nan)
                v2_mean.append(np.nan)

        # Compare to analytic (normalised)
        valid_phi = ~np.isnan(phi_r)
        phi_analytic = analytic_phi(r_centers[valid_phi])
        phi_err = np.abs(np.array(phi_r)[valid_phi] - phi_analytic) / np.abs(phi_analytic)
        phi_deviation_real.append(np.mean(phi_err))

        valid_v2 = ~np.isnan(v2_mean)
        v2_analytic = 0.5 * v_esc_squared(r_centers[valid_v2])
        v2_err = np.abs(np.array(v2_mean)[valid_v2] - v2_analytic) / np.abs(v2_analytic)
        v2_deviation_real.append(np.mean(v2_err))

    phi_devs.append(np.mean(phi_deviation_real))
    v2_devs.append(np.mean(v2_deviation_real))

# Plotting
plt.figure(figsize=(8, 6))
plt.yscale("log")
plt.xscale("log")
plt.plot(Ns, phi_devs, 'o-', label=r'$\delta_\Phi$', color='blue')
plt.plot(Ns, v2_devs, 's--', label=r'$\delta_{\langle v^2 \rangle}$', color='darkorange')

# Slope lines
phi_fit = np.polyfit(np.log10(Ns), np.log10(phi_devs), 1)
v2_fit  = np.polyfit(np.log10(Ns), np.log10(v2_devs), 1)
plt.text(Ns[1], phi_devs[1]*1.2, f"Slope: {phi_fit[0]:.2f}", color='blue')
plt.text(Ns[1], v2_devs[1]*0.8, f"Slope: {v2_fit[0]:.2f}", color='darkorange')

plt.xlabel(r'$N$')
plt.ylabel('Mean Fractional Deviation')
plt.title('Convergence of Sampled Profiles)')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Figures_SG/Q8c_Convergence_Deviation_vs_N_Corrected.pdf", dpi=300)
plt.show()

