import numpy as np
import matplotlib.pyplot as plt
import os
# Output Directory
output_dir = r"C:\\CATAM_SelfGrav\\Figures_SG"
os.makedirs(output_dir, exist_ok=True)

def phi(r, G=1, M=1, a=1):
    return -G * M / (2 * a) * (((r + a)**2 - r**2) / ((r + a)**2))

def main():
    # Define range of r values (avoid r = -a singularity)
    r = np.linspace(0.01, 50, 1000)
    y = phi(r)

    plt.figure()
    plt.plot(r, y)
    plt.xlabel('r')
    plt.ylabel('$\psi(r)$')
    plt.title('$\psi(r)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1_phi_of_r.pdf'), dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
