#!/usr/bin/env python3
"""
Comprehensive plotting suite for transverse momentum spectra dN/dpT
and universal scaled spectra U(xT) across design points.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# --- Configuration ---
SPECIES_SPECTRA = {"pi": 0, "kaon": 1, "proton": 2, "Sigma": 3, "Xi": 4}
COLORS_SPECIES = {"pi": "red", "kaon": "black", "proton": "blue", "Sigma": "green", "Xi": "orange"}
MARKERS_SPECIES = {"pi": "o", "kaon": "^", "proton": "s", "Sigma": "v", "Xi": "d"}
MASSES = {"pi": 0.13957, "kaon": 0.49368, "proton": 0.93827, "Sigma": 1.18937, "Xi": 1.32132}

centbins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]])
cent_colors = ['red', 'black', 'blue', 'green', 'orange', 'cyan', 'gray', 'pink', 'gold', 'chocolate']
cent_markers = ['o', '^', 's', 'v', 'd', 'H', '*', 'h', 'p', 'P']
cent_labels = [f"{int(lo)}-{int(hi)}%" for lo, hi in centbins]

ALPHA_LINE = 0.25
ALPHA_MEAN = 0.9
LINEWIDTH = 1.5
MARKERSIZE = 3
DPI = 150

# Plot style configuration (matching original)
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.minor.width"] = 1.0
plt.rcParams["ytick.minor.width"] = 1.0
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

def load_spectra_results(filepath):
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

# ========== Plot Category 1: Original style multi-panel plot ==========
def plot_spectra_original_style(results, save_plots=True, output_dir="spectra_plots"):
    """
    Reproduce the original plot style: 5 rows (species) × 2 columns (dN/dpT and U(xT))
    Overlays all design points in light color, bold ensemble mean.
    """
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        print("No results to plot")
        return
    
    available_centralities = list(results[0]['centrality_data'].keys())
    fontsize = 23
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), sharey=False, sharex='col')
    
    for row, (species_name, pid) in enumerate(SPECIES_SPECTRA.items()):
        axD = axes[row, 0]  # dN/dpT
        axU = axes[row, 1]  # U(xT)
        
        # Configure axes
        axD.yaxis.set_ticks_position('both')
        axD.xaxis.set_ticks_position('both')
        axD.tick_params(labelsize=fontsize)
        axU.yaxis.set_ticks_position('both')
        axU.xaxis.set_ticks_position('both')
        axU.tick_params(labelsize=fontsize)
        
        # Loop over centralities
        for cent_idx in available_centralities:
            color = cent_colors[cent_idx]
            marker = cent_markers[cent_idx]
            label = cent_labels[cent_idx]
            
            # Collect data from all design points
            all_dNdpt = []
            all_U_xT = []
            all_x_T = []
            all_pt = []
            
            for result in results:
                if cent_idx in result['centrality_data']:
                    cent_data = result['centrality_data'][cent_idx]
                    if species_name in cent_data:
                        species_data = cent_data[species_name]
                        all_dNdpt.append(species_data['dNdpt'])
                        all_U_xT.append(species_data['U_xT'])
                        all_x_T.append(species_data['x_T'])
                        all_pt.append(species_data['pt'])
            
            if not all_dNdpt:
                continue
            
            # Plot individual design points with low alpha
            for i, (pt, dNdpt, x_T, U) in enumerate(zip(all_pt, all_dNdpt, all_x_T, all_U_xT)):
                label_dp = label if i == 0 else None
                axD.plot(pt, dNdpt, color=color, linestyle='--', marker=marker,
                        markersize=MARKERSIZE-1, markerfacecolor='none',
                        markeredgewidth=1.5, alpha=ALPHA_LINE, linewidth=LINEWIDTH-0.5)
                axU.plot(x_T, U, color=color, linestyle='--', marker=marker,
                        markersize=MARKERSIZE-1, markerfacecolor='none',
                        markeredgewidth=1.5, alpha=ALPHA_LINE, linewidth=LINEWIDTH-0.5,
                        label=label_dp)
            
            # Compute and plot ensemble mean
            if len(all_dNdpt) > 1:
                all_dNdpt = np.array(all_dNdpt)
                all_U_xT = np.array(all_U_xT)
                mean_dNdpt = np.mean(all_dNdpt, axis=0)
                mean_U_xT = np.mean(all_U_xT, axis=0)
                pt_ref = all_pt[0]
                x_T_ref = all_x_T[0]
                
                axD.plot(pt_ref, mean_dNdpt, color=color, linestyle='-',
                        marker=marker, markersize=MARKERSIZE+2, linewidth=LINEWIDTH+1,
                        markerfacecolor='none', markeredgewidth=2)
                axU.plot(x_T_ref, mean_U_xT, color=color, linestyle='-',
                        marker=marker, markersize=MARKERSIZE+2, linewidth=LINEWIDTH+1,
                        markerfacecolor='none', markeredgewidth=2)
        
        # Formatting
        axD.set_xlim(-0.1, 3)
        axU.set_xlim(-0.1, 5)
        axU.set_ylim(-0.06, 1.04)
        
        if row == 0:
            axU.legend(loc='best', frameon=False, fontsize=14,
                      title_fontsize=20, title=f'{coll_system}',
                      handletextpad=0.3, numpoints=1)
            axD.set_title('JETSCAPE - All Design Points', fontsize=18)
            axU.set_title(f'Universal Scaling ({len(results)} DPs)', fontsize=18)
        
        if row == len(SPECIES_SPECTRA) - 1:
            axD.set_xlabel(r'$p_T$ (GeV)', fontsize=fontsize+5)
            axU.set_xlabel(r'$x_T$', fontsize=fontsize+5)
        
        axD.set_ylabel(r'$dN/dp_T$', fontsize=fontsize+5)
        axU.yaxis.set_label_position("right")
        axU.yaxis.tick_right()
        axU.yaxis.set_ticks_position('both')
        axU.set_ylabel(r'$U(x_T)$', rotation=-90, fontsize=fontsize+5, labelpad=30)
        
        axD.axhline(y=0, color='black', linestyle='--', linewidth=0.7, alpha=0.3)
        axU.axhline(y=0, color='black', linestyle='--', linewidth=0.7, alpha=0.3)
        
        axD.text(0.98, 0.95, species_name, fontsize=fontsize+6,
                transform=axD.transAxes, va='top', ha='right')
    
    plt.subplots_adjust(wspace=0.02, hspace=0)
    
    if save_plots:
        filename = f"{output_dir}/spectra_all_species_allDP.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

# ========== Plot Category 2: Centrality-separated spectra ==========
def plot_spectra_by_centrality(results, save_plots=True, output_dir="spectra_plots"):
    """
    For each centrality, plot all species dN/dpT and U(xT) with design point overlay.
    """
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        return
    
    available_centralities = list(results[0]['centrality_data'].keys())
    
    for cent_idx in available_centralities:
        cent_label = cent_labels[cent_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
        axD, axU = axes
        
        for species_name in SPECIES_SPECTRA.keys():
            color = COLORS_SPECIES[species_name]
            marker = MARKERS_SPECIES[species_name]
            
            # Collect data
            all_dNdpt = []
            all_U_xT = []
            all_x_T = []
            all_pt = []
            
            for result in results:
                if cent_idx in result['centrality_data']:
                    cent_data = result['centrality_data'][cent_idx]
                    if species_name in cent_data:
                        species_data = cent_data[species_name]
                        all_dNdpt.append(species_data['dNdpt'])
                        all_U_xT.append(species_data['U_xT'])
                        all_x_T.append(species_data['x_T'])
                        all_pt.append(species_data['pt'])
            
            if not all_dNdpt:
                continue
            
            # Plot individual design points
            for pt, dNdpt, x_T, U in zip(all_pt, all_dNdpt, all_x_T, all_U_xT):
                axD.plot(pt, dNdpt, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH-0.5)
                axU.plot(x_T, U, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH-0.5)
            
            # Plot ensemble mean
            if len(all_dNdpt) > 1:
                all_dNdpt = np.array(all_dNdpt)
                all_U_xT = np.array(all_U_xT)
                mean_dNdpt = np.mean(all_dNdpt, axis=0)
                mean_U_xT = np.mean(all_U_xT, axis=0)
                std_dNdpt = np.std(all_dNdpt, axis=0)
                std_U_xT = np.std(all_U_xT, axis=0)
                
                axD.plot(all_pt[0], mean_dNdpt, color=color, linewidth=3,
                        marker=marker, markersize=6, label=f"{species_name} (m={MASSES[species_name]:.2f})",
                        markeredgecolor='black', markeredgewidth=0.5)
                axD.fill_between(all_pt[0], mean_dNdpt - std_dNdpt, mean_dNdpt + std_dNdpt,
                                color=color, alpha=0.2)
                
                axU.plot(all_x_T[0], mean_U_xT, color=color, linewidth=3,
                        marker=marker, markersize=6, label=species_name,
                        markeredgecolor='black', markeredgewidth=0.5)
                axU.fill_between(all_x_T[0], mean_U_xT - std_U_xT, mean_U_xT + std_U_xT,
                                color=color, alpha=0.2)
        
        # Formatting
        axD.set_xlim(0, 3.0)
        axD.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
        axD.set_ylabel(r'$dN/dp_T$', fontsize=14)
        axD.set_title(f'Transverse Momentum Spectra - {cent_label}', fontsize=13)
        axD.legend(fontsize=10, loc='upper right')
        axD.grid(True, alpha=0.3)
        
        axU.set_xlim(0, 5.0)
        axU.set_ylim(-0.05, 1.05)
        axU.set_xlabel(r'$x_T = p_T/\langle p_T \rangle$', fontsize=14)
        axU.set_ylabel(r'$U(x_T) = \langle p_T \rangle / N \cdot dN/dp_T$', fontsize=14)
        axU.set_title(f'Universal Scaling - {cent_label}', fontsize=13)
        axU.legend(fontsize=10, loc='upper right')
        axU.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/spectra_centrality_{cent_idx:02d}_{cent_label.replace('%','pct')}.png"
            plt.savefig(filename, dpi=DPI, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()

# ========== Plot Category 3: Species-separated across centralities ==========
def plot_spectra_by_species(results, species='pi', save_plots=True, output_dir="spectra_plots"):
    """
    For each species, show how dN/dpT and U(xT) evolve across centralities.
    """
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        return
    
    available_centralities = list(results[0]['centrality_data'].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
    axD, axU = axes
    
    for cent_idx in available_centralities:
        color = cent_colors[cent_idx]
        marker = cent_markers[cent_idx]
        label = cent_labels[cent_idx]
        
        # Collect data
        all_dNdpt = []
        all_U_xT = []
        all_x_T = []
        all_pt = []
        
        for result in results:
            if cent_idx in result['centrality_data']:
                cent_data = result['centrality_data'][cent_idx]
                if species in cent_data:
                    species_data = cent_data[species]
                    all_dNdpt.append(species_data['dNdpt'])
                    all_U_xT.append(species_data['U_xT'])
                    all_x_T.append(species_data['x_T'])
                    all_pt.append(species_data['pt'])
        
        if not all_dNdpt:
            continue
        
        # Plot individual design points
        for i, (pt, dNdpt, x_T, U) in enumerate(zip(all_pt, all_dNdpt, all_x_T, all_U_xT)):
            label_dp = label if i == 0 else None
            axD.plot(pt, dNdpt, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH,
                    marker=marker, markersize=MARKERSIZE, label=label_dp)
            axU.plot(x_T, U, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH,
                    marker=marker, markersize=MARKERSIZE, label=label_dp)
        
        # Plot ensemble mean
        if len(all_dNdpt) > 1:
            all_dNdpt = np.array(all_dNdpt)
            all_U_xT = np.array(all_U_xT)
            mean_dNdpt = np.mean(all_dNdpt, axis=0)
            mean_U_xT = np.mean(all_U_xT, axis=0)
            
            axD.plot(all_pt[0], mean_dNdpt, color=color, linewidth=3,
                    marker=marker, markersize=6, markeredgecolor='black', markeredgewidth=0.7)
            axU.plot(all_x_T[0], mean_U_xT, color=color, linewidth=3,
                    marker=marker, markersize=6, markeredgecolor='black', markeredgewidth=0.7)
    
    # Formatting
    axD.set_xlim(0, 3.0)
    axD.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
    axD.set_ylabel(r'$dN/dp_T$', fontsize=14)
    axD.set_title(f'{species} Transverse Momentum Spectra', fontsize=13)
    axD.legend(fontsize=10, loc='upper right')
    axD.grid(True, alpha=0.3)
    
    axU.set_xlim(0, 5.0)
    axU.set_ylim(-0.05, 1.05)
    axU.set_xlabel(r'$x_T = p_T/\langle p_T \rangle$', fontsize=14)
    axU.set_ylabel(r'$U(x_T) = \langle p_T \rangle / N \cdot dN/dp_T$', fontsize=14)
    axU.set_title(f'{species} Universal Scaling', fontsize=13)
    axU.legend(fontsize=10, loc='upper right')
    axU.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{output_dir}/spectra_species_{species}_all_centralities.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

# ========== Main Function ==========
def main():
    """
    Main function to load data and execute all plotting functions.
    """
    # Default input file path
    #input_file = 'spectra_design_points_results.pkl'
    input_file = f'spectra_design_points_results_{coll_system}_{delta_f_name[df]}.pkl'
    
    # Load results
    try:
        results = load_spectra_results(input_file)
        print(f"Loaded results from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Output directory for plots
    output_dir = "spectra_plots"
    
    # Plot Category 1: Original style multi-panel plot
    print("Generating original style multi-panel plot...")
    plot_spectra_original_style(results, save_plots=False, output_dir=output_dir)
    
    # Plot Category 2: Centrality-separated spectra
    print("Generating centrality-separated spectra plots...")
    plot_spectra_by_centrality(results, save_plots=True, output_dir=output_dir)
    
    # Plot Category 3: Species-separated across centralities
    print("Generating species-separated spectra plots...")
    for species in SPECIES_SPECTRA.keys():
        print(f"Plotting for species: {species}")
        plot_spectra_by_species(results, species=species, save_plots=True, output_dir=output_dir)

if __name__ == "__main__":
    main()
