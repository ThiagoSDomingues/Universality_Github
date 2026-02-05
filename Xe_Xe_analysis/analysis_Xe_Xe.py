#!/usr/bin/env python3
###  Xe-Xe analysis: Load and calculate the universal spectra for Xe-Xe 5.44 TeV
# System: Xe-Xe-5440

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Data Loading Functions
# -------------------------------

#def load_moment_data(base_dir, idf):
#    num_design_points = 1000
#    all_data = []
    
    # these are the problematic design points for Xe Xe 5440 w/ 1000 design points
#    nan_design_pts_set_Xe = set([354, 494, 601, 682, 699, 719, 736, 758, 768, 770, 834, 902, 908, 949])
#    unfinished_design_pts_set_Xe = set([328, 354, 562, 672, 682, 736, 818, 897, 902])
#    delete_design_pts_set_Xe = nan_design_pts_set_Xe.union(unfinished_design_pts_set_Xe)

#    for dp in range(num_design_points):
#        if dp in delete_design_pts_set_Xe:
#            continue
#        file_path = os.path.join(base_dir, str(dp), f'universal_alicecut_{idf}.dat')
#        try:
#            data = np.loadtxt(file_path, comments='#')
#            all_data.append(data)
#        except Exception as e:
#            print(f"Error reading file for design point {dp}: {e}")
#    return np.array(all_data)

# Loading design points

def load_design():
    
    # these are the problematic design points for Xe Xe 5440 w/ 1000 design points
    nan_design_pts_set_Xe = set([354, 494, 601, 682, 699, 719, 736, 758, 768, 770, 834, 902, 908, 949])
    unfinished_design_pts_set_Xe = set([328, 354, 562, 672, 682, 736, 818, 897, 902])
    delete_design_pts_set_Xe = nan_design_pts_set_Xe.union(unfinished_design_pts_set_Xe)
    
    main_dir = 'design_pts_Xe_Xe_5440_production'   
    design_file = f'{main_dir}/design_points_main_XeXe-5440.dat'
    range_file = f'{main_dir}/design_ranges_main_XeXe-5440.dat'
    design = pd.read_csv(design_file, index_col=0)
    design = design.drop(labels=list(delete_design_pts_set_Xe), errors='ignore')
    labels = design.keys()
    design_range = pd.read_csv(range_file) # prior
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, labels, design_max, design_min

# Loading Xe-Xe 5.44 TeV experimental data
def load_experimental():
    
    centrality_classes = ['0-5','5-10','10-20','20-30','30-40','40-50',
                        '50-60','60-70','70-80','80-90']
    exp_file_path = 'ALICE_XeXe5440.dat'
    data = pd.read_csv(exp_file_path, sep=' ')
    num_centrality_bins = len(centrality_classes)
#    for i in range(num_centrality_bins):
#        xt_values[i, :] = data[i, 0::3]
#        u_xt_values[i, :] = data[i, 1::3]
#        u_xt_error[i, :] = data[i, 2::3]
#    return xt_values, u_xt_values, u_xt_error

# -------------------------------
# Design points against experimental data (Separated by Centrality)
# -------------------------------
def plot_design_vs_experimental(save_folder, exp=False):
    """
    Plot the design points for U(x_T) (simulation outputs)
    for each centrality by plotting each design point's curve versus x_T,
    and overlay the experimental data with error bars (optional).
    
    Parameters:
      save_folder : str
          Folder to save the resulting figure.
      exp : bool
            If True, overlay experimental data on the plot.
    
    Returns:
      fig : matplotlib.figure.Figure object.
    """
    # Load simulation design points and flatten it
    Y_sim = load_moment_data('production_1000pts_Xe_Xe_5440', 'XeXe-5440')
    Y_flat = prepare_simulation_data(Y_sim) # shape (n_design, n_xT * n_centrality)

    # Load experimental data (each of shape (n_xT, n_centrality))
    xt_values, u_xt_values, u_xt_error = load_experimental()
    
    n_design_points = Y_flat.shape[0]
    n_centrality_bins = xt_values.shape[0]
    n_xT_bins = xt_values.shape[1]

    # Prepare the figure
    fig, axes = plt.subplots(n_centrality_bins, 1, figsize=(10, 6 * n_centrality_bins), sharex=True)
    fig.suptitle('Design Points vs Experimental Data for Xe-Xe 5.44 TeV', fontsize=16)
    if n_centrality_bins == 1:
        axes = [axes]  # Ensure axes is iterable
    # loop through each centrality class
    for i in range(n_centrality_bins):
        ax = axes[i]
        ax.set_title(f'Centrality Class: {centrality_classes[i]}')
        
        # Extract the simulation for this centrality (for each design point)
        data_cent = Y_flat[:, i * n_xT_bins:(i + 1) * n_xT_bins] # shape (n_design_points, n_xT_bins)

        # Plot each design point
        for j in range(n_design_points):
            ax.plot(xt_values[i, :], data_cent[j, :], color='yellow', lw=0.2, alpha=0.5)
        
        # Overlay experimental data if requested
        if exp:
            ax.errorbar(xt_values[i, :], u_xt_values[i, :], yerr=u_xt_error[i, :],
                        fmt='o', color='black', label='Experimental Data', capsize=3, markersize=4, label='ALICE Xe-Xe 5.44 TeV')
        
        # Set labels and legend
        ax.set_title(f'Centrality Class: {centrality_classes[i]}')
        ax.set_xlabel(r"$x_T$")
        ax.set_ylabel(r"$U(x_T)$")
        ax.set_xscale('log')
        ax.legend(fontsize='small')
        ax.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    fig.savefig(os.path.join(save_folder, 'design_vs_experimental_XeXe5440.pdf'), dpi=300)    
    return fig
    
def main():

    xT, UxT, err = load_experimental()
    save_folder = 'Xe_Xe_5.44TeV_analysis'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)               
    fig = plot_design_vs_experimental(save_folder, exp=True)
    fig.savefig(os.path.join(save_folder, 'design_vs_experimental_XeXe5440.pdf'), dpi=300)
    plt.show()
    print("Analysis complete. Figures saved in:", save_folder)         
    
if __name__ == "__main__":
    main()    


