#!/usr/bin/env python3
"""
Universal Spectra Analysis from JETSCAPE MAP Parameters
========================================================

This script computes pT-differential spectra and universal scaled spectra
from JETSCAPE (Pb-Pb collisions at 2.76 TeV) using MAP parameters with
Chapman-Enskog corrections.

The universal scaling U(x_T) follows: U(x_T) = <pT>/N * dN/dpT
where x_T = pT/<pT> is the scaled transverse momentum.

Author: OptimusThi
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from calculations_file_format_single_event import (
    return_result_dtype, 
    Qn_species, 
    Qn_diff_pT_cuts
)

# ============================================================================
# PHYSICAL CONSTANTS AND CONFIGURATION
# ============================================================================

# Particle masses (GeV/c²): π, K, p, Σ, Ξ
MASS_LIST = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])

# pT bin edges (GeV/c)
PT_CUTS = np.array([
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
    0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25,
    1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
    1.95, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.4,
    3.6, 3.8, 4.0, 10.0
])

# pT bin centers
PT_LIST = (PT_CUTS[1:] + PT_CUTS[:-1]) / 2

# Particle configuration
PARTICLES = {
    'pi': {'id': 0, 'label': r'$\pi^{+} + \pi^{-}$'},
    'ka': {'id': 1, 'label': r'$K^{+} + K^{-}$'},
    'pr': {'id': 2, 'label': r'$p + \hat{p}$'},
    'Sigma': {'id': 3, 'label': r'$\Sigma$'},
    'Xi': {'id': 4, 'label': r'$\Xi$'}
}

PARTICLE_ORDER = ['pi', 'ka', 'pr', 'Sigma', 'Xi']

# Centrality bins (%)
CENTRALITY_BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]

# Delta-f correction index (1 = Chapman-Enskog)
DELTA_F_INDEX = 1

# Plot styling
COLORS = ['red', 'black', 'blue', 'green', 'orange', 'cyan', 'gray']
MARKERS = ['o', '^', 's', 'v', 'd', 'H', '*']
CENTRALITY_LABELS = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%']

# ============================================================================
# PLOTTING STYLE CONFIGURATION
# ============================================================================

def configure_matplotlib_style():
    """Set global matplotlib parameters for publication-quality plots."""
    params = {
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'xtick.major.pad': 4,
        'ytick.major.pad': 4,
        'xtick.minor.pad': 4,
        'ytick.minor.pad': 4,
        'legend.handletextpad': 0.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'text.usetex': False,
        'font.family': 'serif'
    }
    plt.rcParams.update(params)

 ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def get_Q_0(alldata, delta_f_index, pid):
    """
    Extract Q_0 values (pT spectrum) for a specific particle type.

    Parameters
    ----------
    alldata : np.ndarray
        Array containing all event data from JETSCAPE.
    delta_f_index : int
        Index for delta-f corrections (0=no correction, 1=Chapman-Enskog).
    pid : int
        Particle ID (0: π, 1: K, 2: p, 3: Σ, 4: Ξ).

    Returns
    -------
    np.ndarray
        Q_0 values for each event, shape (n_events, n_pt_bins).
    """
    return np.array([event[4][delta_f_index][pid][0] for event in alldata])

def get_nsamples(alldata, delta_f_index):
    """
    Extract number of samples per hydro event.

    Parameters
    ----------
    alldata : np.ndarray
        Array containing all event data.
    delta_f_index : int
        Index for delta-f corrections.

    Returns
    -------
    np.ndarray
        Number of SMASH events in each hydro event. Zeros replaced with inf
        to avoid division errors (resulting in zero per-sample averages).
    """
    nsamples = np.array([event[3][delta_f_index][0] for event in alldata])
    nsamples[nsamples == 0] = np.inf
    return nsamples

# ============================================================================
# CENTRALITY SELECTION FUNCTIONS
# ============================================================================

def sort_by_charged_multiplicity(Qn_norm, n_charged_species=3):
    """
    Sort events by charged particle multiplicity (descending order).

    Parameters
    ----------
    Qn_norm : np.ndarray
        Normalized spectra, shape (n_events, n_species, n_pt_bins) or
        (n_events, n_species, n_harmonics, n_pt_bins).
    n_charged_species : int, optional
        Number of charged particle species to sum (default: 3 for π, K, p).

    Returns
    -------
    np.ndarray
        Event indices sorted by descending multiplicity.
    """
    # Handle different array dimensions
    if Qn_norm.ndim == 4:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :n_charged_species, 0, :], 
                                     axis=(1, 2))
    elif Qn_norm.ndim == 3:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :n_charged_species, :], 
                                     axis=(1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {Qn_norm.ndim}D")
    
    return np.argsort(charged_mult)[::-1]


def select_centrality_groups(eventlist, centrality_bins):
    """
    Group events into centrality percentile bins.

    Parameters
    ----------
    eventlist : np.ndarray
        Event indices sorted by multiplicity (descending).
    centrality_bins : list of tuples
        List of (low%, high%) centrality boundaries.

    Returns
    -------
    list of np.ndarray
        Event indices for each centrality bin.
    """
    n_events = len(eventlist)
    groups = []
    
    for low_pct, high_pct in centrality_bins:
        idx_low = int(np.floor(low_pct / 100.0 * n_events))
        idx_high = int(np.floor(high_pct / 100.0 * n_events))
        groups.append(eventlist[idx_low:idx_high])
    
    return groups
    
# ============================================================================
# UNIVERSAL SPECTRA CALCULATION
# ============================================================================

def compute_universal_spectra(dNdpt_cent, pt_bins):
    """
    Compute universal scaling function U(x_T) for a centrality class.

    The universal scaling follows:
        U(x_T) = <pT>/N * dN/dpT
    where:
        x_T = pT/<pT> is the scaled transverse momentum
        <pT> is the mean transverse momentum
        N is the total multiplicity

    Parameters
    ----------
    dNdpt_cent : np.ndarray
        Differential spectra per event, shape (n_events_in_cent, n_pt_bins).
    pt_bins : np.ndarray
        pT bin centers (GeV/c).

    Returns
    -------
    mean_pT : float
        Mean transverse momentum <pT> (GeV/c).
    N : float
        Total multiplicity (integrated yield).
    U : np.ndarray
        Universal spectrum U(x_T), same shape as pt_bins.
    """
    # Average spectrum over events in this centrality
    mean_spectrum = np.mean(dNdpt_cent, axis=0)
    
    # Total multiplicity (integrate dN/dpT)
    N = np.trapz(mean_spectrum, pt_bins)
    
    # Mean transverse momentum
    mean_pT = np.trapz(pt_bins * mean_spectrum, pt_bins) / N
    
    # Universal function
    U = mean_pT * mean_spectrum / N
    
    return mean_pT, N, U    

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def configure_axis(ax, fontsize=23):
    """Configure axis styling for publication quality."""
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelsize=fontsize)


def plot_spectra_comparison(dNdpt_all, pt_bins, particle_order, particle_info,
                            centrality_bins, centrality_labels, colors, markers,
                            fontsize=23):
    """
    Create comparison plots of differential and universal spectra.

    Parameters
    ----------
    dNdpt_all : dict
        Dictionary with particle names as keys, spectra as values.
    pt_bins : np.ndarray
        pT bin centers.
    particle_order : list
        Order of particles for plotting.
    particle_info : dict
        Particle metadata (id, label).
    centrality_bins : list of tuples
        Centrality bin boundaries.
    centrality_labels : list of str
        Labels for centrality bins.
    colors : list
        Colors for each centrality.
    markers : list
        Markers for each centrality.
    fontsize : int, optional
        Base font size for labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray
        Array of axes objects.
    """
    n_particles = len(particle_order)
    fig, axes = plt.subplots(n_particles, 2, figsize=(12, 18), 
                             sharey=False, sharex='col')
    
    for row, particle_name in enumerate(particle_order):
        dNdpt = dNdpt_all[particle_name]
        particle_label = particle_info[particle_name]['label']
        
        # Get axes for this particle
        ax_diff = axes[row, 0]  # dN/dpT
        ax_univ = axes[row, 1]  # U(x_T)
        
        configure_axis(ax_diff, fontsize)
        configure_axis(ax_univ, fontsize)
        
        # Plot each centrality
        for cent_idx, (event_indices, cent_label) in enumerate(
            zip(centrality_bins, centrality_labels)):
            
            if len(event_indices) == 0:
                print(f"Warning: No events in centrality {cent_label}, "
                      f"particle {particle_name}")
                continue
            
            dNdpt_cent = dNdpt[event_indices]
            mean_pT, N, U = compute_universal_spectra(dNdpt_cent, pt_bins)
            
            # Plot differential spectrum
            mean_spectrum = np.mean(dNdpt_cent, axis=0)
            ax_diff.plot(pt_bins, mean_spectrum,
                        color=colors[cent_idx], linestyle='--', 
                        marker=markers[cent_idx], markersize=2,
                        markerfacecolor='none', markeredgewidth=2)
            
            # Plot universal spectrum
            x_T = pt_bins / mean_pT
            ax_univ.plot(x_T, U,
                        color=colors[cent_idx], linestyle='--',
                        marker=markers[cent_idx], markersize=2,
                        markerfacecolor='none', markeredgewidth=2,
                        label=cent_label)
        
        # Configure axes limits and labels
        ax_diff.set_xlim(-0.1, 3.0)
        ax_univ.set_xlim(-0.1, 5.0)
        ax_univ.set_ylim(-0.06, 1.04)
        
        # Add legends (only for top row)
        if row == 0:
            ax_univ.legend(loc='best', frameon=False, fontsize=14,
                          title='Pb-Pb @ 2.76 TeV (MAP CE)',
                          title_fontsize=20, handletextpad=0.3, numpoints=1)
            ax_diff.legend(loc='best', frameon=False, fontsize=14,
                          title='JETSCAPE', title_fontsize=20,
                          handletextpad=0.3, numpoints=1, labels=[])
        
        # X-axis labels (only for bottom row)
        if row == n_particles - 1:
            ax_diff.set_xlabel(r'$p_T$ (GeV/c)', fontsize=fontsize + 5)
            ax_univ.set_xlabel(r'$x_T = p_T / \langle p_T \rangle$', 
                              fontsize=fontsize + 5)
        
        # Y-axis labels
        ax_diff.set_ylabel(r'$dN/dp_T$ (GeV/c)$^{-1}$', fontsize=fontsize + 5)
        ax_univ.yaxis.set_label_position("right")
        ax_univ.yaxis.tick_right()
        ax_univ.yaxis.set_ticks_position('both')
        ax_univ.set_ylabel(r'$U(x_T)$', rotation=-90, 
                          fontsize=fontsize + 5, labelpad=30)
        
        # Reference lines
        ax_diff.axhline(y=0, color='black', linestyle='--', 
                       linewidth=0.7, alpha=0.3)
        ax_univ.axhline(y=0, color='black', linestyle='--', 
                       linewidth=0.7, alpha=0.3)
        
        # Particle label
        ax_diff.text(0.98, 0.95, particle_label, fontsize=fontsize + 6,
                    transform=ax_diff.transAxes, va='top', ha='right')
    
    plt.subplots_adjust(wspace=0.02, hspace=0)
    return fig, axes

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Configure plotting style
    configure_matplotlib_style()
    
    # Load data
    print("Loading JETSCAPE MAP data...")
    result_dtype = return_result_dtype('ALICE')
    data = np.fromfile('results_recalc_MAP_CE_Pb_2760_fine_pT.dat', 
                       dtype=result_dtype)
    print(f"Loaded {len(data)} events")
    
    # Get normalization samples
    samples = get_nsamples(data, DELTA_F_INDEX)
    
    # Extract and normalize spectra for all particles
    print("Processing particle spectra...")
    Qn_norm_all = np.zeros((len(data), len(PARTICLES), len(PT_LIST)))
    
    for particle_name, info in PARTICLES.items():
        pid = info['id']
        Q_0 = get_Q_0(data, DELTA_F_INDEX, pid)
        Qn_norm_all[:, pid, :] = Q_0 / samples[:, np.newaxis] / np.diff(PT_CUTS)
    
    # Sort by charged multiplicity
    print("Sorting events by centrality...")
    eventlist = sort_by_charged_multiplicity(Qn_norm_all)
    cent_groups = select_centrality_groups(eventlist, CENTRALITY_BINS)
    
    # Prepare data for plotting
    dNdpt_dict = {}
    for particle_name, info in PARTICLES.items():
        pid = info['id']
        dNdpt_dict[particle_name] = Qn_norm_all[:, pid, :]
    
    # Create plots
    print("Creating plots...")
    fig, axes = plot_spectra_comparison(
        dNdpt_dict, PT_LIST, PARTICLE_ORDER, PARTICLES,
        cent_groups, CENTRALITY_LABELS, COLORS, MARKERS
    )
    
    print("Done! Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
