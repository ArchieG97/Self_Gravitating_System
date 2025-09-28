# Question 2 – Generate and save f(epsilon) * g(epsilon) plot

import numpy as np
import matplotlib.pyplot as plt
import os

# Output Directory
output_dir = r"C:\\CATAM_SelfGrav\\Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Analytic f(epsilon)
def f_analytic(epsilon):
    term1 = (3 - 4 * epsilon) * np.sqrt(2 * epsilon) / (1 - 2 * epsilon)
    term2 = 3 * np.arcsinh(np.sqrt(2 * epsilon / (1 - 2 * epsilon)))
    return (3 / (2 * np.pi**3)) * (term1 - term2)

# Analytic g(epsilon)
def g_analytic(epsilon):
    sqrt_term = np.sqrt(1 - 2 * epsilon)
    eps2 = epsilon**2
    eps52 = (2 * epsilon)**(2.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        arccos_term = np.arccos(-sqrt_term)

    term1 = sqrt_term * (3 - 14 * epsilon - 8 * eps2) / (12 * eps2)
    term2 = -np.pi
    term3 = ((1 - 6 * epsilon + 16 * eps2) / eps52) * arccos_term
    return 8 * np.pi**2 * (term1 + term2 + term3)

# Evaluate
eps_vals = np.linspace(1e-4, 0.499, 500)
f_vals = f_analytic(eps_vals)
g_vals = g_analytic(eps_vals)
dM_deps = f_vals * g_vals

# Plot
plt.figure(figsize=(8, 5))
plt.plot(eps_vals, dM_deps, label=r"$f(\epsilon)g(\epsilon)$", color='purple')
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\mathrm{d}M/\mathrm{d}\epsilon$")
plt.title(r"Differential Energy Distribution $\mathrm{d}M/\mathrm{d}\epsilon$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q2_Energy_Distribution_fg.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print("Finished!")
