"""
Script to plot the Universal spectra $U(x_T)$ experimental data for XeXe at 5.44 TeV.
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Path to your data file
out_path = 'ALICE_XeXe5440.dat'

# Define centrality mapping based on index
centrality_classes = [
    '0-5', '5-10', '10-20', '20-30', '30-40', 
    '40-50', '50-60', '60-70', '70-80', '80-90'
]

# --- Load and Plot ---
# Read the dataset
df2 = pd.read_csv(out_path, sep=' ')

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Group by centrality and plot each with errors
for cent, g in df2.groupby('centrality'):
    ax.errorbar(
        g['xT'], 
        g['U_xT'], 
        yerr=g['err'], 
        fmt='o', 
        ms=4, 
        label=f'{centrality_classes[cent]}%'
    )

# Formatting
ax.set_xscale('log')
ax.set_yscale('log') # Usually needed for spectral data, remove if not
ax.set_xlabel(r'$x_T$')
ax.set_ylabel(r'$U(x_T)$')
ax.legend(title='Centrality')
ax.set_title('ALICE XeXe 5.44 TeV Scaled Spectra')
ax.grid(True, which='both', ls='--', lw=0.5)

plt.tight_layout()
plt.show()
