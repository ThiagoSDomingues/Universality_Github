import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import i0, k1
from scipy.optimize import curve_fit  # Alternative to iminuit

# Pion mass (GeV/c^2)
m_pi = 0.1396

# Integrand (r from 0 to 1)
def blastwave_integrand(r, pt, m, T, beta_s, n):
    beta = beta_s * r**n
    rho = np.arctanh(beta)
    mt = np.sqrt(m**2 + pt**2)
    xip = pt * np.sinh(rho) / T
    xim = mt * np.cosh(rho) / T
    return 2 * np.pi * pt * r * mt * i0(xip) * k1(xim)

# Vectorized BW function
def blastwave(pt, T, beta_s, n):
    def scalar(pt_i):
        integ, _ = quad(blastwave_integrand, 0, 1, args=(pt_i, m_pi, T, beta_s, n))
        return integ
    return np.array([scalar(p) for p in pt])

# Sample pion data (replace with real: e.g., ALICE PbPb 0-5%, HEPData Table1.csv)
pt_data = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
dndpt_data = np.array([1000, 800, 500, 200, 50, 10])  # Example yields
dndpt_err = 0.1 * dndpt_data

# Fit (with norm A)
def model(pt, A, T, beta_s, n):
    return A * blastwave(pt, T, beta_s, n)

popt, _ = curve_fit(model, pt_data, dndpt_data, sigma=dndpt_err,
                    p0=[1, 0.12, 0.7, 1.0], bounds=([0,0.05,0,0], [10,0.2,1,3]))

# Plot
pt_fit = np.linspace(0.2, 3, 100)
dndpt_fit = model(pt_fit, *popt)
plt.errorbar(pt_data, dndpt_data, dndpt_err, fmt='o', label='Data')
plt.plot(pt_fit, dndpt_fit, 'r-', label=f'Fit: T={popt[1]:.3f} GeV, β_s={popt[2]:.2f}')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('p_T (GeV/c)'); plt.ylabel('d²N/(2π p_T dp_T dy)')
plt.legend(); plt.show()

print(f"Parameters: A={popt[0]:.2f}, T={popt[1]:.3f} GeV, β_s={popt[2]:.2f}, n={popt[3]:.2f}")