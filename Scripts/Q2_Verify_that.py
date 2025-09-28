import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# Output Directory
output_dir = r"C:\\CATAM_SelfGrav\\Figures_SG"
os.makedirs(output_dir, exist_ok=True)

# Constants
const_numeric = 3 / (4 * np.sqrt(2) * np.pi**3)
const_analytic = 3 / (2 * np.pi**3)

# Numerical integrand function
def integrand(Psi, eps):
    y = np.sqrt(1 - 2 * Psi)
    numerator = (1 - y)**2 * (2 * y + 4 * y**2)
    denominator = y**4 * np.sqrt(eps - Psi)
    return numerator / denominator

# Numerical computation of f(ϵ)
def f_numeric(eps):
    if eps <= 0 or eps >= 0.5:
        return 0
    val, _ = quad(integrand, 0, eps, args=(eps,), limit=100)
    return const_numeric * val

# Analytic expression for f(ϵ)
def f_analytic(eps):
    if eps <= 0 or eps >= 0.5:
        return 0
    num = (3 - 4*eps) * np.sqrt(2*eps)
    denom = 1 - 2*eps
    arcsinh_term = np.arcsinh(np.sqrt(2*eps / denom))
    return const_analytic * (num / denom - 3 * arcsinh_term)

# Sample ϵ values
eps_values = np.linspace(0.01, 0.49, 100)
f_numeric_vals = np.array([f_numeric(eps) for eps in eps_values])
f_analytic_vals = np.array([f_analytic(eps) for eps in eps_values])

# Relative error
rel_diff = 100 * np.abs(f_numeric_vals - f_analytic_vals) / f_analytic_vals

# Plotting the relative percentage difference
plt.figure(figsize=(10, 4))
plt.semilogy(eps_values, rel_diff, color='purple', lw=2)
plt.xlabel('ϵ', fontsize=14)
plt.ylabel('Relative Error [%]', fontsize=14)
plt.title('Log-Scale Relative Difference: Numerical vs. Analytic f(ϵ)', fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q2_Verify_that.pdf'), dpi=300, bbox_inches='tight')
plt.show()
