import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import i0, k1
from scipy.optimize import curve_fit

# Pion mass (GeV/c^2)
m_pi = 0.1396

# Blast-wave integrand and function (from before)
def blastwave_integrand(r, pt, m, T, beta_s, n):
    beta = beta_s * r**n
    rho = np.arctanh(beta)
    mt = np.sqrt(m**2 + pt**2)
    xip = pt * np.sinh(rho) / T
    xim = mt * np.cosh(rho) / T
    return 2 * np.pi * pt * r * mt * i0(xip) * k1(xim)

def blastwave(pt, T, beta_s, n):
    def scalar(pt_i):
        integ, _ = quad(blastwave_integrand, 0, 1, args=(pt_i, m_pi, T, beta_s, n))
        return integ
    return np.array([scalar(p) for p in pt])

# Sample ALICE-like data: replace with HEPData (e.g., np.loadtxt('alice_pions.csv'))
pt_data = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
dndpt_data = np.array([1000, 800, 500, 200, 50, 10])  # d²N/(2π p_T dp_T dy)
dndpt_err = 0.1 * dndpt_data

# Compute <pT> via trapezoid (N=1 normalized)
mean_pt_num = np.trapz(pt_data * dndpt_data, pt_data)
mean_pt_den = np.trapz(dndpt_data, pt_data)
mean_pt = mean_pt_num / mean_pt_den
print(f'Mean p_T: {mean_pt:.3f} GeV/c')

xt_data = pt_data / mean_pt
u_xt_data = mean_pt * dndpt_data  # U(x_T) for normalized spectrum
u_xt_err = mean_pt * dndpt_err

# BW model for U(x_T): rescale pT to xt, fit to U
def model_uxt(xt_data, T, beta_s, n):
    pt_model = xt_data * mean_pt
    bw = blastwave(pt_model, T, beta_s, n)
    norm = np.trapz(bw, pt_model)  # Normalize model integral to 1
    return bw / norm * mean_pt  # -> U(x_T)

# Fit to U(x_T)
popt_u, _ = curve_fit(model_uxt, xt_data, u_xt_data, sigma=u_xt_err,
                      p0=[0.12, 0.7, 1.0], bounds=([0.05,0,0], [0.2,1,3]))

# Plot spectra and U(x_T)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original spectrum
pt_fit = np.linspace(0.2, 3, 100)
def model(pt, T, beta_s, n):
    return blastwave(pt, T, beta_s, n)
dndpt_fit = model(pt_fit, *popt_u)
ax1.errorbar(pt_data, dndpt_data, dndpt_err, fmt='o', label='Data')
ax1.semilogy(pt_fit, dndpt_fit, 'r-', label=f'BW fit to U(x_T): T={popt_u[0]:.3f}')
ax1.set_xlabel('p_T (GeV/c)'); ax1.set_ylabel('Yield'); ax1.legend()

# U(x_T)
xt_fit = np.linspace(0.2, 3.5, 100)
u_xt_fit = model_uxt(xt_fit, *popt_u)
ax2.errorbar(xt_data, u_xt_data, u_xt_err, fmt='o', label='Data')
ax2.plot(xt_fit, u_xt_fit, 'r-', label='BW')
ax2.set_xlabel('x_T'); ax2.set_ylabel('U(x_T)'); ax2.legend()
ax2.set_xlim(0, 3.5); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'BW params from U(x_T): T={popt_u[0]:.3f} GeV, β_s={popt_u[1]:.2f}, n={popt_u[2]:.2f}')