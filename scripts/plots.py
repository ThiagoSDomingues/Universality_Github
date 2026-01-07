### Author: OptimusThi
"""
Python Script to generate the paper figures
"""
import matplotlib.pyplot as plt
import scienceplots

# Define PRC physical constants
# =========================
# PRC / APS FIGURE SETTINGS
# =========================

MM_TO_PT = 72.0 / 25.4  # exact conversion

# APS minimums
MIN_TEXT_MM = 2.0
MIN_SUB_MM  = 1.5

MIN_TEXT_PT = MIN_TEXT_MM * MM_TO_PT     # ≈ 5.7 pt
MIN_SUB_PT  = MIN_SUB_MM  * MM_TO_PT     # ≈ 4.3 pt

# PRC column widths
PRC_ONE_COL_CM = 8.6
PRC_TWO_COL_CM = 17.8

FIG_WIDTH_CM  = PRC_TWO_COL_CM
FIG_HEIGHT_CM = 12.0  # safe for PRC

fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(FIG_WIDTH_CM/2.54, FIG_HEIGHT_CM/2.54),
    sharex='col',
    gridspec_kw={'height_ratios': [3, 1]}
)

FONT_MAIN = MIN_TEXT_PT
FONT_SUB  = MIN_SUB_PT

# Automatic PRC compliance checks: before to save
def check_prc_font_compliance(fig):
    """
    Check that all text objects satisfy APS minimum font sizes.
    """
    violations = []

    for text in fig.findobj(match=plt.Text):
        size = text.get_fontsize()
        if size < MIN_SUB_PT:
            violations.append((text.get_text(), size))

    if violations:
        print("❌ PRC FONT VIOLATIONS FOUND:")
        for txt, size in violations:
            print(f"  '{txt}' -> {size:.2f} pt")
        raise RuntimeError("Figure does NOT satisfy PRC font requirements.")
    else:
        print("✅ PRC font-size compliance check passed.")

plt.savefig(
    "posterior_Grad_paper_model_to_data.pdf",
    format="pdf",
    bbox_inches="tight"
)
