### Author: OptimusThi
"""
Python Script to generate the paper figures
"""
import matplotlib.pyplot as plt
import scienceplots



plt.style.use([
    "science",
    "no-latex",   # or "latex" if you use TeX
    "journal"
])

# Set figure size in inches, not pixels
# One-column PRC figure
fig_width_cm = 8.6
fig_height_cm = 6.0

fig = plt.figure(
    figsize=(fig_width_cm/2.54, fig_height_cm/2.54)
)

# convert mm -> pt
MM_TO_PT = 72 / 25.4  # ≈ 2.835
MIN_FONT_PT = 2.0 * MM_TO_PT  # ≈ 5.7 pt
SUB_FONT_PT = 1.5 * MM_TO_PT  # ≈ 4.3 pt

# Apply global adjustments
plt.rcParams.update({
    "font.size": MIN_FONT_PT,
    "axes.labelsize": MIN_FONT_PT,
    "axes.titlesize": MIN_FONT_PT,
    "xtick.labelsize": SUB_FONT_PT,
    "ytick.labelsize": SUB_FONT_PT,
    "legend.fontsize": SUB_FONT_PT,
})

# Python sanity check to ensure that all figures are in the PRC requirements!
for text in fig.findobj(match=plt.Text):
    if text.get_fontsize() < MIN_FONT_PT:
        print("Warning:", text.get_text())

# Save any figure as: 
#plt.savefig("figure1.pdf", bbox_inches="tight")
