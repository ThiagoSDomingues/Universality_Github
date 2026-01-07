#!/usr/bin/env python3
# ==========================================
# PRC / APS COMPLIANT FIGURE SCRIPT
# Figure 1: Prior vs Posterior + Ratios
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PRC / APS PHYSICAL CONSTANTS (DO NOT CHANGE)
# ============================================================

MM_TO_PT = 72.0 / 25.4

MIN_TEXT_MM = 2.0      # APS minimum
MIN_SUB_MM  = 1.5

MIN_TEXT_PT = MIN_TEXT_MM * MM_TO_PT   # ≈ 5.7 pt
MIN_SUB_PT  = MIN_SUB_MM  * MM_TO_PT   # ≈ 4.3 pt

PRC_ONE_COL_CM = 8.6
PRC_TWO_COL_CM = 17.8

# ============================================================
# 2. GLOBAL MATPLOTLIB STYLE (PRC SAFE)
# ============================================================

plt.rcParams.update({
    # Line widths
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,

    # Tick sizes
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,

    # Tick direction
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Fonts (absolute, PRC-safe)
    "font.family": "serif",
    "font.size": MIN_TEXT_PT,
    "axes.labelsize": MIN_TEXT_PT,
    "xtick.labelsize": MIN_SUB_PT,
    "ytick.labelsize": MIN_SUB_PT,
    "legend.fontsize": MIN_SUB_PT,
})

# ============================================================
# 3. PLACEHOLDER FUNCTIONS (REPLACE WITH REAL ONES)
# ============================================================

def load_experimental():
    """
    RETURNS:
        xt_exp      : (7, 41)
        u_xt_exp    : (7, 41)
        err_u_xt    : (7, 41)
    """
    x = np.logspace(-2, 0, 41)
    xt = np.tile(x, (7, 1))
    u  = np.exp(-x)[None, :] * np.linspace(1.2, 0.6, 7)[:, None]
    err = 0.1 * u
    return xt, u, err


def predict_observables(theta, Emulators, inverse_tf_matrix, scaler):
    """
    Placeholder emulator.
    Replace with your trained GP surrogate.
    """
    y = np.exp(-np.linspace(0.01, 1.0, 7 * 41))
    return y.reshape(-1, 1), None


def make_save_dir(*args):
    return "."


# ============================================================
# 4. PRC FONT COMPLIANCE CHECK
# ============================================================

def check_prc_font_compliance(fig):
    violations = []

    for text in fig.findobj(match=plt.Text):
        if text.get_fontsize() < MIN_SUB_PT:
            violations.append((text.get_text(), text.get_fontsize()))

    if violations:
        print("\n❌ PRC FONT VIOLATIONS:")
        for txt, size in violations:
            print(f"   '{txt}' → {size:.2f} pt")
        raise RuntimeError("Figure does NOT satisfy PRC font requirements.")
    else:
        print("✅ PRC font compliance check passed.")


# ============================================================
# 5. AXIS CONFIGURATION
# ============================================================

def configure_axis(ax):
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


# ============================================================
# 6. MAIN FIGURE FUNCTION (FIGURE 1)
# ============================================================

def plot_qm_proc(posterior_file, Emulators, inverse_tf_matrix, scaler):

    # -------------------------
    # Load posterior samples
    # -------------------------
    try:
        samples_df = pd.read_csv(posterior_file)
        posterior_samples = samples_df.values
    except FileNotFoundError:
        posterior_samples = np.random.rand(500, 17)

    n_pr_samples = 1000
    design_min = np.zeros(17)
    design_max = np.ones(17)

    # -------------------------
    # Experimental data
    # -------------------------
    xt_exp, u_xt_exp, err_u_xt = load_experimental()

    # -------------------------
    # Generate predictions
    # -------------------------
    pr_predictions = []
    post_predictions = []

    for p in np.random.uniform(design_min, design_max, (n_pr_samples, 17)):
        y, _ = predict_observables(p, Emulators, inverse_tf_matrix, scaler)
        pr_predictions.append(y.flatten())

    for p in posterior_samples:
        y, _ = predict_observables(p, Emulators, inverse_tf_matrix, scaler)
        post_predictions.append(y.flatten())

    pr_predictions = np.array(pr_predictions).reshape(n_pr_samples, 7, 41)
    post_predictions = np.array(post_predictions).reshape(len(posterior_samples), 7, 41)

    centrality_indices = [0, 6]

    # -------------------------
    # PRC FIGURE SIZE (CRITICAL)
    # -------------------------
    FIG_WIDTH_CM = PRC_TWO_COL_CM
    FIG_HEIGHT_CM = 12.0

    fig, axs = plt.subplots(
        2, 2,
        figsize=(FIG_WIDTH_CM / 2.54, FIG_HEIGHT_CM / 2.54),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1]}
    )

    axs_top = axs[0]
    axs_bot = axs[1]

    for i, cent in enumerate(centrality_indices):

        ax = axs_top[i]
        axr = axs_bot[i]

        pr_env = np.percentile(pr_predictions[:, cent, :], [5, 50, 95], axis=0)
        post_env = np.percentile(post_predictions[:, cent, :], [5, 50, 95], axis=0)

        # ---- Spectra
        ax.fill_between(
            xt_exp[cent], pr_env[0], pr_env[2],
            hatch="///", facecolor="gray", edgecolor="black",
            alpha=0.3, label="Prior 90% C.I."
        )

        ax.fill_between(
            xt_exp[cent], post_env[0], post_env[2],
            color="tab:blue", alpha=0.4,
            label="Posterior 90% C.I."
        )

        ax.errorbar(
            xt_exp[cent], u_xt_exp[cent],
            yerr=err_u_xt[cent],
            fmt="o", color="black", capsize=3,
            label="ALICE Pb–Pb 2.76 TeV"
        )

        ax.set_xscale("log")
        configure_axis(ax)

        if i == 0:
            ax.set_ylabel(r"$U(x_T)$", fontsize=MIN_TEXT_PT)

        ax.legend(frameon=False)

        # ---- Ratio
        ratio_low = post_env[0] / u_xt_exp[cent]
        ratio_high = post_env[2] / u_xt_exp[cent]

        axr.fill_between(
            xt_exp[cent], ratio_low, ratio_high,
            color="tab:blue", alpha=0.4
        )

        axr.errorbar(
            xt_exp[cent], np.ones_like(xt_exp[cent]),
            yerr=err_u_xt[cent] / u_xt_exp[cent],
            fmt="o", color="black", capsize=3
        )

        axr.axhline(1.0, color="k", ls="--", lw=1.2)
        axr.set_ylim(0.5, 1.4)
        axr.set_xscale("log")
        axr.set_xlabel(r"$x_T$", fontsize=MIN_TEXT_PT)

        if i == 0:
            axr.set_ylabel("Model/Data", fontsize=MIN_SUB_PT)

        configure_axis(axr)

    plt.tight_layout()
    check_prc_font_compliance(fig)

    plt.savefig(
        "figure1_prc_compliant.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()
    return fig


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":

    Emulators = None
    inverse_tf_matrix = None
    scaler = None

    plot_qm_proc(
        posterior_file="posterior_samples.csv",
        Emulators=Emulators,
        inverse_tf_matrix=inverse_tf_matrix,
        scaler=scaler
    )
