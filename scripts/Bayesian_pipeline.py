"""
Script to create a entire Bayesian pipeline to make all the analysis of the Universal spectra.
"""

import os
import pickle
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lapack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import ptemcee
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
np.float = float  # Fix deprecated alias for ptemcee

# -------------------------------
# Plot Decorator & Saving Helpers
# -------------------------------
def plot(func):
    """Decorator to display and return the figure."""
    def wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        plt.show()
        return fig
    return wrapper

def make_save_dir(main_folder, visc_corr_label, subfolder):
    path = os.path.join(main_folder, f"{visc_corr_label}", subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig, filename, folder, dpi=600):
    full_path = os.path.join(folder, filename)
    fig.savefig(full_path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# -------------------------------
# Global Definitions for Labels & Colors
# -------------------------------
exp_centrality_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"]
exp_markers = ['o', 's', '^', 'D', 'v', '<', '>']
idf_label_short = {0: 'Grad', 1: 'CE', 2: 'PTM', 3: 'PTB'}
color_map = {0: 'blue', 1: 'red', 2: 'magenta', 3: 'green'}
#Model parameter names in Latex compatble form
model_param_dsgn = ['$N$[$2.76$TeV]',
 '$p$',
 '$\\sigma_k$',
 '$w$ [fm]',
 '$d_{\\mathrm{min}}$ [fm]',
 '$\\tau_R$ [fm/$c$]',
 '$\\alpha$',
 '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
 '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
 '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
 '$(\\eta/s)_{\\mathrm{kink}}$',
 '$(\\zeta/s)_{\\max}$',
 '$T_{\\zeta,c}$ [GeV]',
 '$w_{\\zeta}$ [GeV]',
 '$\\lambda_{\\zeta}$',
 '$b_{\\pi}$',
 '$T_{\\mathrm{sw}}$ [GeV]']

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_moment_data(base_dir, idf):
    num_design_points = 500
    all_data = []
    nan_sets_by_deltaf = {
        0: {334, 341, 377, 429, 447, 483},
        1: {285, 334, 341, 447, 483, 495},
        2: {209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495},
        3: {60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495}
    }
    nan_design_pts_set = nan_sets_by_deltaf.get(idf, set())
    unfinished_events_design_pts_set = {289, 324, 326, 459, 462, 242, 406, 440, 123}
    strange_features_design_pts_set = {289, 324, 440, 459, 462}
    delete_design_pts_set = nan_design_pts_set.union(unfinished_events_design_pts_set).union(strange_features_design_pts_set)
    for dp in range(num_design_points):
        if dp in delete_design_pts_set:
            continue
        file_path = os.path.join(base_dir, str(dp), f'universal_alicecut_{idf}.dat')
        try:
            data = np.loadtxt(file_path, comments='#')
            all_data.append(data)
        except Exception as e:
            print(f"Error reading file for design point {dp}: {e}")
    return np.array(all_data)

def load_design(idf):
    nan_sets_by_deltaf = {
        0: {334, 341, 377, 429, 447, 483},
        1: {285, 334, 341, 447, 483, 495},
        2: {209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495},
        3: {60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495}
    }
    nan_design_pts_set = nan_sets_by_deltaf.get(idf, set())
    unfinished_events_design_pts_set = {289, 324, 326, 459, 462, 242, 406, 440, 123}
    strange_features_design_pts_set = {289, 324, 440, 459, 462}
    delete_design_pts_set = nan_design_pts_set.union(unfinished_events_design_pts_set).union(strange_features_design_pts_set)
    design = pd.read_csv('design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat', index_col=0)
    design = design.iloc[:325, :]
    design = design.drop(labels=list(delete_design_pts_set), errors='ignore')
    return design

def load_full_design(system, idf):
    
    if system == 'Pb-Pb-2760':
        # load the design
        nan_sets_by_deltaf = {
            0: {334, 341, 377, 429, 447, 483},
            1: {285, 334, 341, 447, 483, 495},
            2: {209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495},
            3: {60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495}
        }
        nan_design_pts_set = nan_sets_by_deltaf.get(idf, set())
        unfinished_events_design_pts_set = {289, 324, 326, 459, 462, 242, 406, 440, 123}
        strange_features_design_pts_set = {289, 324, 440, 459, 462}
        delete_design_pts_set = nan_design_pts_set.union(unfinished_events_design_pts_set).union(strange_features_design_pts_set)
    
        design_file = 'design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat'
        range_file = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat'
        design = pd.read_csv(design_file, index_col=0)
        design = design.drop(labels=list(delete_design_pts_set), errors='ignore')
        labels = design.keys()
        design_range = pd.read_csv(range_file) # prior
        design_max = design_range['max'].values
        design_min = design_range['min'].values

        #    elif system == 'Au-Au-200':
        
#        design_file = 'design_pts_Pb_Pb_2760_production/'
#        range_file = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat'   
#    else: 
#        design_file = 'design_pts_Pb_Pb_2760_production/'
#        range_file = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat'    
    
#    return design, labels, design_max, design_min        
    return design, labels, design_max, design_min

def load_validation():
    validation = pd.read_csv('design_pts_Pb_Pb_2760_production/design_points_validation_PbPb-2760.dat', index_col=0)
    return validation

def load_experimental():
    exp_file_path = "Bayesian_data/ALICE_PbPb2p76.dat"
    num_centrality_bins = len(exp_centrality_labels)
    num_xt_bins = 41
    data = np.loadtxt(exp_file_path, comments='#')
    xt_values = np.zeros((num_centrality_bins, num_xt_bins))
    u_xt_values = np.zeros((num_centrality_bins, num_xt_bins))
    u_xt_error = np.zeros((num_centrality_bins, num_xt_bins))
    for i in range(num_centrality_bins):
        xt_values[i, :] = data[i, 0::3]
        u_xt_values[i, :] = data[i, 1::3]
        u_xt_error[i, :] = data[i, 2::3]
    return xt_values, u_xt_values, u_xt_error

def prepare_simulation_data(Y):
    Y_reduced = Y[:, :7, :]
    n_design, n_cent, n_xt = Y_reduced.shape
    Y_flat = Y_reduced.reshape(n_design, n_cent * n_xt)
    return Y_flat

# -------------------------------
# Prior for each viscous correction (Separated by Centrality)
# -------------------------------

@plot
def plot_prior_scaled_spectra_by_centrality(idf, save_folder, exp=False):
    """
    Plot the prior distribution of the observable U(x_T) (simulation outputs)
    for each centrality by plotting each design point's curve versus x_T,
    and overlay the experimental data with error bars.
    
    Parameters:
      idf : int
          Viscous correction index.
      save_folder : str
          Folder to save the resulting figure.
    
    Returns:
      fig : matplotlib.figure.Figure object.
    """
    # Load simulation data and flatten it
    Y_sim = load_moment_data('Bayesian_data', idf)
    Y_flat = prepare_simulation_data(Y_sim)  # shape (n_design, 287)
    
    # Load experimental data (each of shape (7, 41))
    xt_exp, u_xt_exp, err_u_xt = load_experimental()
    
    n_design = Y_flat.shape[0]
    n_cent = 7
    n_xt = 41

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
    axs = axs.flatten()
    
    # Loop over each centrality bin
    for cent in range(n_cent):
        ax = axs[cent]
        # Extract the simulation predictions for this centrality (for each design point)
        data_cent = Y_flat[:, cent*n_xt:(cent+1)*n_xt]  # shape (n_design, 41)
        # Plot each design point separately
        for i in range(n_design):
            ax.plot(xt_exp[cent, :], data_cent[i, :], color=color_map[idf], lw=0.2, alpha=0.5)
        # Optionally, you could also compute and plot an envelope:
#        lower = np.percentile(data_cent, 5, axis=0)
#        median = np.percentile(data_cent, 50, axis=0)
#        upper = np.percentile(data_cent, 95, axis=0)
#        ax.fill_between(xt_exp[cent, :], lower, upper, color='gray', alpha=0.3, label="5-95% Envelope")
#        ax.plot(xt_exp[cent, :], median, color='black', lw=2, label="Median")
        # Overlay experimental data (with error bars)
        if exp:
            ax.errorbar(xt_exp[cent, :], u_xt_exp[cent, :], yerr=err_u_xt[cent, :],
                        fmt=exp_markers[cent], color='black', capsize=3, label="Experimental")
        ax.set_title(f"Centrality {exp_centrality_labels[cent]}")
        ax.set_xlabel(r"$x_T$")
        ax.set_ylabel(r"$U(x_T)$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    # Turn off any extra subplots
    for ax in axs[n_cent:]:
        ax.axis('off')
    plt.tight_layout()
    save_fig(fig, "prior_scaled_spectra_separated.pdf", save_folder)
    return fig

# -------------------------------
# Prior for each viscous correction (All Centralities onto a single plot)
# -------------------------------

@plot
def plot_prior_scaled_spectra_overlay(idf, save_folder, exp=False):
    """
    Plot the prior distribution of the observable U(x_T) (simulation outputs)
    for all centralities in a single plot by overlaying all design points.
    
    Parameters:
      idf : int
          Viscous correction index.
      save_folder : str
          Folder to save the resulting figure.
    
    Returns:
      fig : matplotlib.figure.Figure object.
    """
    # Load simulation data and flatten it
    Y_sim = load_moment_data('Bayesian_data', idf)
    Y_flat = prepare_simulation_data(Y_sim)  # shape (n_design, 287)
    
    # Load experimental data (each of shape (7, 41))
    xt_exp, u_xt_exp, err_u_xt = load_experimental()
    
    n_design = Y_flat.shape[0]
    n_cent = 7
    n_xt = 41

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop over each centrality bin
    for cent in range(n_cent):
        data_cent = Y_flat[:, cent*n_xt:(cent+1)*n_xt]  # shape (n_design, 41)
        
        # Plot each design point as a faint curve
        for i in range(n_design):
            ax.plot(xt_exp[cent, :], data_cent[i, :], color=color_map[idf], lw=0.5, alpha=0.2)
        
        # Compute and plot the 90% credible interval
#        lower = np.percentile(data_cent, 5, axis=0)
#        median = np.percentile(data_cent, 50, axis=0)
#        upper = np.percentile(data_cent, 95, axis=0)
        
        # Plot credible interval bands and a median
#        ax.fill_between(xt_exp[cent, :], lower, upper, color=color, alpha=0.2, label=f"{exp_centrality_labels[cent]} 5-95% CI")
#        ax.plot(xt_exp[cent, :], median, color=color, lw=2, label=f"{exp_centrality_labels[cent]} Median")

        # Overlay experimental data for this centrality
        if exp:
            ax.errorbar(xt_exp[cent, :], u_xt_exp[cent, :], yerr=err_u_xt[cent, :],
                        fmt=exp_markers[cent], color='black', capsize=3, label=f"{exp_centrality_labels[cent]} ALICE Pb-Pb 2.76 TeV.")

    ax.set_xlabel(r"$x_T$")
    ax.set_ylabel(r"$U(x_T)$")
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc='upper right', ncol=2, frameon=False)
    ax.set_title(f"Prior Scaled Spectra - All Centralities ({idf_label_short[idf]})")
    
    plt.tight_layout()
    save_fig(fig, "prior_scaled_spectra_overlay.pdf", save_folder)
    return fig

# -------------------------------
# Prior Predictive Plot (Separated by Centrality)
# -------------------------------
@plot
def plot_prior_observables_separated(idf, save_folder):
    """
    Create a figure with one subplot per centrality showing simulation (design point)
    predictions as curves, their 5th-95th percentile envelope, and overlay experimental data.
    """
    # Load simulation data and experimental data.
    Y_sim = load_moment_data('Bayesian_data', idf)
    Y_flat = prepare_simulation_data(Y_sim)  # (n_design, 287)
    xt_exp, u_xt_exp, err_u_xt = load_experimental()  # each: (7,41)
    
    n_design = Y_flat.shape[0]
    n_cent = 7
    n_xt = 41
    
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
    axs = axs.flatten()
    for cent in range(n_cent):
        ax = axs[cent]
        data_cent = Y_flat[:, cent*n_xt:(cent+1)*n_xt]
        lower = np.percentile(data_cent, 5, axis=0)
        median = np.percentile(data_cent, 50, axis=0)
        upper = np.percentile(data_cent, 95, axis=0)
        ax.fill_between(xt_exp[cent, :], lower, upper, color='gray', alpha=0.3, label="Sim Envelope")
        ax.plot(xt_exp[cent, :], median, color='black', lw=2, label="Sim Median")
        ax.errorbar(xt_exp[cent, :], u_xt_exp[cent, :], yerr=err_u_xt[cent, :],
                    fmt=exp_markers[cent], color=color_map[idf], capsize=3, label="Experimental")
        ax.set_title(f"Centrality {exp_centrality_labels[cent]}")
        ax.set_xlabel(r"$x_T$")
        ax.set_ylabel(r"$U(x_T)$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    # Turn off any extra subplots
    for ax in axs[n_cent:]:
        ax.axis('off')
    plt.tight_layout()
    save_fig(plt.gcf(), "prior_observables_separated.pdf", save_folder)
    return plt.gcf()

# -------------------------------
# PCA Pipeline Functions
# -------------------------------
def fit_pca_model(Y, variance_threshold=0.99, n_components=None):
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y)
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(Y_scaled)
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        print(f"Selected n_components: {n_components} to capture at least {variance_threshold*100:.1f}% of variance")
    pca = PCA(n_components=n_components, whiten=False)
    Z = pca.fit_transform(Y_scaled)
    Y_reconstructed = pca.inverse_transform(Z)
    print(f"Original data shape: {Y.shape}")
    print(f"PC-transformed data shape: {Z.shape}")
    return scaler, pca, Z, Y_reconstructed

def plot_explained_variance(pca, n_pc_to_plot=None):
    explained = pca.explained_variance_ratio_
    if n_pc_to_plot is None:
        n_pc_to_plot = len(explained)
    idx = np.arange(1, n_pc_to_plot+1)
    cumulative = np.cumsum(explained)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(idx, explained[:n_pc_to_plot], color='skyblue')
    ax1.set_xlabel("PC Index")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Variance per PC")
    ax2.bar(idx, cumulative[:n_pc_to_plot], color='lightgreen')
    ax2.set_xlabel("PC Index")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("Cumulative Explained Variance")
    ax2.axhline(0.99, color='r', linestyle='--', label='99% threshold')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    return fig

# -------------------------------
# Additional PCA Validation Functions
# -------------------------------
@plot
def plot_reconstruction(Y, scaler, pca, n_pc_retained, sample_index=0, save_folder=None):
    """
    Plot the original observable vs. its reconstruction for a given sample using n_pc_retained PCs.
    """
    pca_temp = PCA(n_components=n_pc_retained, whiten=False)
    Z_temp = pca_temp.fit_transform(scaler.transform(Y))
    Y_reconstructed = pca_temp.inverse_transform(Z_temp)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(Y[sample_index, :], label="Original", lw=2)
    ax.plot(Y_reconstructed[sample_index, :], '--', label=f"Reconstructed ({n_pc_retained} PCs)", lw=2)
    ax.set_xlabel("Observable Index")
    ax.set_ylabel("Value")
    ax.set_title("Original vs. Reconstructed Observable")
    ax.legend()
    if save_folder:
        save_fig(fig, f"reconstruction_sample{sample_index}_{n_pc_retained}PCs.pdf", save_folder)
    return fig

@plot
def corner_pc_scores(Z, save_folder=None):
    """
    Create a corner plot of the PC scores from the training data.
    """
    import corner
    fig = corner.corner(Z, labels=[f"PC{i+1}" for i in range(Z.shape[1])],
                        show_titles=True, title_fmt=".2f")
    if save_folder:
        save_fig(fig, "corner_PC_scores.pdf", save_folder)
    return fig

# -------------------------------
# PCA Normality Check (Scatterplots)
# -------------------------------
@plot
def plot_original_scatter_checks(Y, sample_pairs=None, save_folder=None):
    """
    Plot scatterplots for selected pairs of observables from the original simulation data.
    This helps check that the distributions are roughly normal (elliptical) and linearly correlated.
    
    Parameters:
      Y : np.ndarray, shape (n_design, n_obs)
      sample_pairs : list of tuples, each tuple is (i, j) indicating which observable indices to plot.
                     If None, default to first 4 observables.
    """
    if sample_pairs is None:
        sample_pairs = [(0,1), (0,2), (1,2), (2,3)]
    fig, axes = plt.subplots(nrows=1, ncols=len(sample_pairs), figsize=(5*len(sample_pairs), 4))
    if len(sample_pairs) == 1:
        axes = [axes]
    for idx, (i, j) in enumerate(sample_pairs):
        axes[idx].scatter(Y[:, i], Y[:, j], alpha=0.5, s=10, color='navy')
        axes[idx].set_xlabel(f"Observable {i+1}")
        axes[idx].set_ylabel(f"Observable {j+1}")
        axes[idx].set_title(f"Scatter: Obs {i+1} vs Obs {j+1}")
    plt.tight_layout()
    if save_folder:
        save_fig(plt.gcf(), "scatter_normality_checks.pdf", save_folder)
    return plt.gcf()

# -------------------------------
# Emulator Training Functions
# -------------------------------
EMULATOR_FILE = "emulators_pcs.pkl"

def train_emulators_for_pcs(idf, pc_tf_data, n_pc_train=6, emulator_output_file=EMULATOR_FILE, train_emulators=False):
    emulator_output_file = f"emulators/emulators_{idf_label_short[idf]}_{n_pc_train}_pcs.pkl"
    design = load_design(idf)
    prior_df = pd.read_csv('design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat', index_col=0)
    design_max = prior_df['max'].values
    design_min = prior_df['min'].values
    ptp = design_max - design_min
    if pc_tf_data.shape[1] < n_pc_train:
        raise ValueError("pc_tf_data has fewer components than requested.")
    pc_tf_data_train = pc_tf_data[:, :n_pc_train]
    n_pc = n_pc_train
    if (os.path.exists(emulator_output_file)) and (not train_emulators):
        print("Saved emulators exist; loading them.")
        with open(emulator_output_file, "rb") as f:
            Emulators = pickle.load(f)
    else:
        Emulators = []
        for i in range(n_pc):
            start_time = time.time()
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=ptp,
                        length_scale_bounds=np.outer(ptp, (0.4, 1e2))) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e2))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4, alpha=1e-10)
            gpr.fit(design.values, pc_tf_data_train[:, i].reshape(-1, 1))
            score = gpr.score(design.values, pc_tf_data_train[:, i].reshape(-1, 1))
            print(f"Trained emulator for PC {i+1}/{n_pc} with score {score:.3f} in {time.time()-start_time:.2f} s.")
            Emulators.append(gpr)
        with open(emulator_output_file, "wb") as f:
            pickle.dump(Emulators, f)
    return Emulators

# -------------------------------
# Observable Prediction Function
# -------------------------------
#def predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS):
#    model_parameters = np.array(model_parameters).flatten()
#    if model_parameters.shape[0] != 17:
#        raise ValueError("Input model parameters must be a 17-dimensional array.")
#    theta = model_parameters.reshape(1, -1)
#    n_pc = len(Emulators)
#    pc_means = []
#    pc_vars = []
#    for emulator in Emulators:
#        mn, std = emulator.predict(theta, return_std=True)
#        pc_means.append(mn.flatten()[0])
#        pc_vars.append(std.flatten()[0]**2)
#    pc_means = np.array(pc_means).reshape(1, -1)
#    variance_matrix = np.diag(np.array(pc_vars))
#    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + SS.mean_.reshape(1, -1)
#    A = inverse_tf_matrix[:n_pc, :]
#    inverse_transformed_variance = np.einsum('ik,kl,lj->ij', A.T, variance_matrix, A)
#    return inverse_transformed_mean.flatten(), inverse_transformed_variance

def predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS):
    model_parameters = np.array(model_parameters).flatten()
    if model_parameters.shape[0] != 17:
        raise ValueError("Input model parameters must be a 17-dimensional array.")
    theta = model_parameters.reshape(1, -1)
    n_pc = len(Emulators)
    pc_means = []
    pc_vars = []
    for emulator in Emulators:
        mn, std = emulator.predict(theta, return_std=True)
        pc_means.append(mn.flatten()[0])
        pc_vars.append(std.flatten()[0]**2)
    pc_means = np.array(pc_means).reshape(1, -1)
    variance_matrix = np.diag(np.array(pc_vars))
    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + SS.mean_.reshape(1, -1)
    A = inverse_tf_matrix[:n_pc, :]
    inverse_transformed_variance = np.einsum('ik,kl,lj->ij', A.T, variance_matrix, A)
    return inverse_transformed_mean.flatten(), inverse_transformed_variance

# -------------------------------
# Leave-P-Out Cross-Validation for Emulator Validation
# -------------------------------
def leave_p_out_validation(design, pc_tf_data, P=10, n_splits=5, save_folder=None):
    from sklearn.metrics import mean_squared_error
    n_design = design.shape[0]
    n_pc = pc_tf_data.shape[1]
    rmse_dict = {i: [] for i in range(n_pc)}
    indices = np.arange(n_design)
    for split in range(n_splits):
        val_idx = np.random.choice(indices, size=P, replace=False)
        train_idx = np.setdiff1d(indices, val_idx)
        train_design = design.values[train_idx]
        val_design = design.values[val_idx]
        for i in range(n_pc):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ptp(train_design, axis=0).mean(),
                        length_scale_bounds=(0.4, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e2))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-10)
            gpr.fit(train_design, pc_tf_data[train_idx, i].reshape(-1, 1))
            preds = gpr.predict(val_design).flatten()
            true_vals = pc_tf_data[val_idx, i]
            rmse = np.sqrt(mean_squared_error(true_vals, preds))
            rmse_dict[i].append(rmse)
    fig, ax = plt.subplots(figsize=(8,4))
    data_to_plot = [rmse_dict[i] for i in range(n_pc)]
    ax.boxplot(data_to_plot, labels=[f"PC{i+1}" for i in range(n_pc)])
    ax.set_xlabel("PC Index")
    ax.set_ylabel("RMSE")
    ax.set_title("Leave-P-Out Cross-Validation RMSE per PC")
    plt.tight_layout()
    if save_folder:
        save_fig(fig, "LPOCV_RMSE.pdf", save_folder)
    return rmse_dict, fig

# -------------------------------
# Emulator Validation by Centrality
# -------------------------------
@plot
def plot_emulator_validation_by_centrality(design, Y_flat, Emulators, inverse_tf_matrix, scaler, save_folder):
    """
    For each centrality, plot a scatter of true vs. emulator-predicted observables (over all design points).
    """
    n_design, n_obs = Y_flat.shape
    n_cent = 7
    n_xt = 41
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()
    for cent in range(n_cent):
        true_vals = Y_flat[:, cent*n_xt:(cent+1)*n_xt]  # shape (n_design, 41)
        # For each design point, predict observable using its design parameters:
        pred_vals = []
        design_array = design.values
        for j in range(n_design):
            y_pred, _ = predict_observables(design_array[j, :], Emulators, inverse_tf_matrix, scaler)
            pred_vals.append(y_pred[cent*n_xt:(cent+1)*n_xt])
        pred_vals = np.array(pred_vals)  # shape (n_design, 41)
        # Flatten both arrays:
        true_flat = true_vals.flatten()
        pred_flat = pred_vals.flatten()
        axes[cent].scatter(true_flat, pred_flat, color=color_map[0], alpha=0.5, s=10)  # Use color for Grad if idf==0
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        axes[cent].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[cent].set_title(f"Centrality {exp_centrality_labels[cent]}", fontsize=12)
        axes[cent].set_xlabel("True U(x_T)")
        axes[cent].set_ylabel("Predicted U(x_T)")
    for ax in axes[n_cent:]:
        ax.axis('off')
    plt.tight_layout()
    save_fig(plt.gcf(), "emulator_validation_by_centrality.pdf", save_folder)
    return plt.gcf()

# -------------------------------
# Prior Predictive Check (Separated by Centrality)
# -------------------------------
@plot
def plot_prior_predictive_separated(n_samples, design_min, design_max, Emulators, inverse_tf_matrix, SS, save_folder):
    """
    Perform a prior predictive check by sampling parameters uniformly from the prior,
    predicting observables via the emulator surrogate, and plotting the envelope for each centrality.
    """
    ndim = len(design_min)
    samples = design_min + (design_max - design_min) * np.random.rand(n_samples, ndim)
    predictions = []
    for theta in samples:
        y_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, SS)
        predictions.append(y_pred)
    predictions = np.array(predictions)  # (n_samples, n_obs)
    n_cent = exp_centrality_labels.__len__()  # should be 7
    n_xt = 41
    predictions_reshaped = predictions.reshape(n_samples, n_cent, n_xt)
    
    xt_exp, u_xt_exp, err_u_xt = load_experimental()  # each: (7,41)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
    axs = axs.flatten()
    for cent in range(n_cent):
        ax = axs[cent]
        env = np.percentile(predictions_reshaped[:, cent, :], [5, 50, 95], axis=0)
        ax.fill_between(xt_exp[cent, :], env[0, :], env[2, :], color='gray', alpha=0.3, label="Prior Envelope")
        ax.plot(xt_exp[cent, :], env[1, :], 'k-', lw=2, label="Prior Median")
        ax.errorbar(xt_exp[cent, :], u_xt_exp[cent, :], yerr=err_u_xt[cent, :],
                    fmt=exp_markers[cent], color=color_map[0], capsize=3, label="Experimental")
        ax.set_title(f"Centrality {exp_centrality_labels[cent]}")
        ax.set_xlabel(r"$x_T$")
        ax.set_ylabel(r"$U(x_T)$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    for ax in axs[n_cent:]:
        ax.axis('off')
    plt.tight_layout()
    save_fig(plt.gcf(), "prior_predictive_separated.pdf", save_folder)
    return plt.gcf()

# -------------------------------
# Prior and design point predictions for each viscous correction (Separated by Centrality)
# -------------------------------

@plot
def plot_prior_CI(
    n_pr_samples, 
    show_exp=False,
    bound_min=5,
    bound_max=95,
    save_folder="prior"
):
    """
    Plot prior and Bayesian design point predictions overlaid with 90% credible intervals.
    """
    
    # Read prior min/max values from my design-range file
    data_path = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat' 
    prior_df = pd.read_csv(data_path, index_col=0)
    
    # Read design points (Optional)
    design_bayesian = load_design(idf)
    
    # Read Full design points (Optional)
#    full_design_path = 'PbPb2760_design'
#    full_design = pd.read_csv = pd.read_csv(full_design_path, index_col=0)
    
    # Extracting the min and max prior values
    design_min = prior_df['min'].values
    design_max = prior_df['max'].values
    
    # Extracting design samples
    design_samples = design_bayesian.values
    
    # Number of prior draws 
    # n_pr_samples = 10000
    np.random.seed(1) # for reproducibility
    
    # Collect scaled spectra predictions for each prior sample
    pr_predictions = []
    
    # Collect scaled spectra predictions for each design sample
    ds_predictions = []
    
    # Looping over all prior samples: supposing a uniform prior
    # Use the emulator surrogate to predict observables for each prior sample
    for params in np.random.uniform(design_min, design_max, (n_pr_samples, 17)):
        y_pred, cov_pred = predict_observables(params, Emulators, inverse_tf_matrix, scaler)
        pr_predictions.append(y_pred.flatten())
    
    pr_predictions = np.array(pr_predictions) # shape (n_samples, n_obs)
    
    # Looping over all design points
    for theta in design_samples:
        y_model, cov_model = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
        ds_predictions.append(y_model.flatten())
    
    ds_predictions = np.array(ds_predictions)  # shape (n_samples, n_obs) 
      
    # Load experimental data
    xt_exp, u_xt_exp, err_u_xt = load_experimental()  # each: (7, 41)
    
    # Reshape predictions into (n_pr_samples, 7, 41)
    pr_n_cent = len(exp_centrality_labels)
    pr_n_xt = len(xt_exp[0])    
    pr_predictions_reshaped = pr_predictions.reshape(n_pr_samples, pr_n_cent, pr_n_xt)

    # Reshape predictions into (ds_n_samples, 7, 41)
    ds_n_samples = ds_predictions.shape[0]
    ds_n_cent = len(exp_centrality_labels) 
    ds_n_xt = len(xt_exp[0])
    ds_predictions_reshaped = ds_predictions.reshape(ds_n_samples, ds_n_cent, ds_n_xt)

    # Plotting
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=1200)
    axs = axs.flatten()

    for cent in range(pr_n_cent):
        ax = axs[cent]
        
        # Compute credible intervals
        pr_env = np.percentile(pr_predictions_reshaped[:, cent, :], 
                               [bound_min, 50, bound_max], axis=0)
        ds_env = np.percentile(ds_predictions_reshaped[:, cent, :], 
                               [bound_min, 50, bound_max], axis=0)
        
        # Plot prior band
        ax.fill_between(xt_exp[cent, :], pr_env[0, :], pr_env[2, :], 
                        color='gray', alpha=0.6, label=f"Prior {bound_max - bound_min}% C.I.")
        
        
        # Plot design band
        ax.fill_between(xt_exp[cent, :], ds_env[0, :], ds_env[2, :],
                        color=color_map[idf], alpha=0.3,
                        label=f"Bayesian Design {idf_label_short[idf]} {bound_max - bound_min}% C.I.")
        
        # Plot Full design band
#        ax.fill_between(xt_exp[cent], ds_perc[0, cent], ds_perc[2, cent],
#                        color=color_map[idf], alpha=0.3,
#                        label=f"Full Design {idf_name} {bound_max - bound_min}% C.I.")

        if show_exp:
            ax.errorbar(xt_exp[cent], u_xt_exp[cent], yerr=err_u_xt[cent],
                        fmt=exp_markers[cent], color='black', capsize=3,
                        label="ALICE PbPb 2.76 TeV")

        ax.set_title(f"Centrality {exp_centrality_labels[cent]}")
        ax.set_xlabel(r"$x_T$")
        ax.set_ylabel(r"$U(x_T)$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
#        ax.grid(True, which="both", ls="--", lw=0.3)

    for ax in axs[pr_n_cent:]:
        ax.axis("off")

    plt.tight_layout()

    # Save to appropriate folder
    if show_exp:
        save_fig(plt.gcf(), "prior_design_exp_CI.pdf", save_folder)
    else: 
        save_fig(plt.gcf(), "prior_design_CI.pdf", save_folder)
    
    return plt.gcf()

# -------------------------------
# Sensitivity Plot for All Centralities
# -------------------------------
@plot
def plot_sensitivity(idf, Emulators, inverse_tf_matrix, scaler, save_folder):
    """
    Performs Sobol sensitivity analysis for all seven centralities,
    finding the peak for each centrality and plotting sensitivity indices.
    """
    # Load experimental data (xt_exp shape: (7, 41))
    xt_exp, u_xt_exp, _ = load_experimental()
    
    # Find peak indices for each centrality
    peak_indices = np.argmax(u_xt_exp, axis=1)  # Shape: (7,)
    print(f"Peak x_T indices: {peak_indices}")
    
    # Define the Sobol problem using parameter bounds from design ranges:
    prior_df = pd.read_csv('design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat', index_col=0)
    param_names = list(prior_df.index)
    bounds = prior_df[['min', 'max']].values.tolist()
    problem = {'num_vars': 17, 'names': param_names, 'bounds': bounds}
    
    # Generate parameter samples using Saltelli:
    param_samples = saltelli.sample(problem, 500)
    
    # Prepare subplots for all centralities
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    for i, peak_idx in enumerate(peak_indices):
        preds = []
        for theta in param_samples:
            y_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
            preds.append(y_pred[peak_idx])
        preds = np.array(preds)
        
        # Perform Sobol sensitivity analysis
        Si = sobol.analyze(problem, preds, print_to_console=False)
        
        # Plot results
        axes[i].bar(labels, Si['S1'], yerr=Si['S1_conf'], color=color_map[idf], edgecolor='black', alpha=0.8)
        axes[i].set_title(f"{exp_centrality_labels[i]}", fontsize=14)
        axes[i].tick_params(axis='x', labelrotation=60, labelsize=10)
    
    # Remove the last empty subplot
    fig.delaxes(axes[-1])
    
    # Set overall figure properties
    fig.supylabel("First-Order Sensitivity Index (S1)", fontsize=16)
    fig.supxlabel("Model Parameters", fontsize=16)
    fig.suptitle(f"Sensitivity Analysis at Peak x_T for {idf_label_short[idf]}", fontsize=18)
    plt.tight_layout()
    save_fig(fig, "sensitivity_plot_all.pdf", save_folder)
    return fig

# -------------------------------
# Sensitivity Plot for All Centralities
# -------------------------------
@plot
def plot_sensitivity(idf, Emulators, inverse_tf_matrix, scaler, save_folder):
    """
    Performs Sobol sensitivity analysis for all seven centralities,
    finding the peak for each centrality and plotting sensitivity indices.
    """
    
    labels = [r'$N$', r'$p$', r'$\sigma_k$', r'$w$', r'$d_{\mathrm{min}}^3$', r'$\tau_R$', r'$\alpha$', r'$T_{\eta,\mathrm{kink}}$',
    r'$a_{\eta,\mathrm{low}}$', r'$a_{\eta,\mathrm{high}}$', r'$(\eta/s)_{\mathrm{kink}}$', r'$(\zeta/s)_{\max}$', r'$T_{\zeta,c}$',
    r'$w_{\zeta}$', r'$\lambda_{\zeta}$', r'$b_{\pi}$', r'$T_{\mathrm{sw}}$']
    label_indx = np.arange(len(labels))
    
    # Load experimental data (xt_exp shape: (7, 41))
    xt_exp, u_xt_exp, _ = load_experimental()
    
    # Find peak indices for each centrality
    peak_indices = np.argmax(u_xt_exp, axis=1)  # Shape: (7,)
    print(f"Peak x_T indices: {peak_indices}")
    
    # Define the Sobol problem using parameter bounds from design ranges:
    prior_df = pd.read_csv('design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat', index_col=0)
    param_names = list(prior_df.index)
    bounds = prior_df[['min', 'max']].values.tolist()
    problem = {'num_vars': 17, 'names': param_names, 'bounds': bounds}
    
    # Generate parameter samples using Saltelli:
    param_samples = saltelli.sample(problem, 500)
    
    # Prepare subplots for all centralities
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()
    
    for i, peak_idx in enumerate(peak_indices):
        preds = []
        for theta in param_samples:
            y_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
            preds.append(y_pred[peak_idx])
        preds = np.array(preds)
        
        # Perform Sobol sensitivity analysis
        Si = sobol.analyze(problem, preds, print_to_console=False)
        
        # Plot results
        axes[i].bar(labels, Si['S1'], yerr=Si['S1_conf'], color=color_map[idf], edgecolor='black', alpha=0.8)
        axes[i].set_title(f"{exp_centrality_labels[i]}", fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Set overall figure properties
    fig.supylabel("First-Order Sensitivity Index (S1)", fontsize=16)
    fig.supxlabel("Model Parameters", fontsize=16)
    fig.suptitle(f"Sensitivity Analysis at Peak x_T for {idf_label_short[idf]}", fontsize=18)
    plt.tight_layout()
    save_fig(fig, "sensitivity_plot_all.pdf", save_folder)
    return fig

# -------------------------------
# Sensitivity Plot for Principal Components (PCs)
# -------------------------------
@plot
def plot_sensitivity_pcs(idf, Emulators, inverse_tf_matrix, scaler, save_folder):
    """
    Performs Sobol sensitivity analysis for the first two principal components (PCs).
    """
    
    labels = [r'$N$', r'$p$', r'$\sigma_k$', r'$w$', r'$d_{\mathrm{min}}^3$', r'$\tau_R$', r'$\alpha$', r'$T_{\eta,\mathrm{kink}}$',
    r'$a_{\eta,\mathrm{low}}$', r'$a_{\eta,\mathrm{high}}$', r'$(\eta/s)_{\mathrm{kink}}$', r'$(\zeta/s)_{\max}$', r'$T_{\zeta,c}$',
    r'$w_{\zeta}$', r'$\lambda_{\zeta}$', r'$b_{\pi}$', r'$T_{\mathrm{sw}}$']
    label_indx = np.arange(len(labels))
    
    # Load the first two PCs
#    pcs = f"emulators_name_{idf_label_short[idf]}_pcs"
    print(f"Sensitivity analysis for {idf_label_short[idf]} viscous correction and first two PCs.")
    
    # Define the Sobol problem using parameter bounds from prior
    prior_df = pd.read_csv('design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat', index_col=0)
    bounds = prior_df[['min', 'max']].values.tolist()
    problem = {'num_vars': 17, 'names': labels, 'bounds': bounds}
    
    # Generate parameter samples using Saltelli
    param_samples = saltelli.sample(problem, 500)
    
    preds_pc1, preds_pc2 = [], []
    for theta in param_samples:
        y_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
        preds_pc1.append(y_pred[0])  # First PC
        preds_pc2.append(y_pred[1])  # Second PC
    
    preds_pc1, preds_pc2 = np.array(preds_pc1), np.array(preds_pc2)
    
    # Perform Sobol sensitivity analysis
    Si_pc1 = sobol.analyze(problem, preds_pc1, print_to_console=False)
    Si_pc2 = sobol.analyze(problem, preds_pc2, print_to_console=False)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for i, (Si, title) in enumerate(zip([Si_pc1, Si_pc2], ["First PC", "Second PC"])):
        axes[i].bar(labels, Si['S1'], yerr=Si['S1_conf'], color=color_map[idf], edgecolor='black', alpha=0.8)
        axes[i].set_title(title, fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Set overall figure properties
    fig.supylabel("First-Order Sensitivity Index (S1)", fontsize=16)
    fig.supxlabel("Model Parameters", fontsize=16)
    fig.suptitle(f"Sensitivity Analysis for PCs ({idf_label_short[idf]})", fontsize=18)
    plt.tight_layout()
    save_fig(fig, "sensitivity_plot_pcs.pdf", save_folder)
    return fig

# -------------------------------
# MCMC Sampling & Diagnostics 
# -------------------------------
#def log_prior(model_parameters):
#    X = np.array(model_parameters).reshape(1, -1)
#    lower_check = np.all(X >= design_min)
#    upper_check = np.all(X <= design_max)
#    return 0.0 if (lower_check and upper_check) else -np.inf

#def mvn_loglike(y, cov, initial_jitter=1e-8, max_jitter=1e-5):
#    jitter = initial_jitter
#    while jitter < max_jitter:
#        cov_jittered = cov + np.eye(cov.shape[0]) * jitter
#        try:
#            L, info = lapack.dpotrf(cov_jittered, lower=True, clean=True)
#            if info == 0:
#                break
#        except Exception as e:
#            pass
#        jitter *= 10
#    if jitter >= max_jitter:
#        raise np.linalg.LinAlgError("Covariance matrix is not positive definite even after adding maximum jitter.")
#    alpha, info = lapack.dpotrs(L, y, lower=True)
#    if info != 0:
#        raise ValueError("Error in solving linear system.")
#    log_det = 2.0 * np.sum(np.log(np.diag(L)))
#    return -0.5 * (np.dot(y, alpha) + log_det)

#def log_like(model_parameters):
#    mn, var = predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS)
#    delta_y = (mn - y_exp.reshape(1, -1)).flatten()
#    exp_cov = np.diag(np.diag(y_exp_variance))
#    components_omitted = pca.components_[n_pc_train:, :]
#    var_omitted = pca.explained_variance_[n_pc_train:]
#    cov_omitted_scaled = components_omitted.T @ np.diag(var_omitted) @ components_omitted
#    trunc_cov = np.diag(SS.scale_) @ cov_omitted_scaled @ np.diag(SS.scale_)
#    total_cov = var + exp_cov + trunc_cov
#    return mvn_loglike(delta_y, total_cov)

#def log_posterior(model_parameters):
#    lp = log_prior(model_parameters)
#    if not np.isfinite(lp):
#        return -np.inf
#    return lp + log_like(model_parameters)

def log_prior(model_parameters):
    """
    Uniform Prior. Evaluvate prior for model. 
    
    Parameters
    ----------
    model_parameters : 17 dimensional list of floats
    
    Return
    ----------
    unnormalized probability : float 
    
    If all parameters are inside bounds function will return 0 otherwise -inf"""
    X = np.array(model_parameters).reshape(1,-1)
    lower = np.all(X >= design_min)
    upper = np.all(X <= design_max)
    if (lower and upper):
        lp=0
    else:
        lp = -np.inf
    return lp

def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
  #  print(L.diagonal())
    a=np.ones(len(L.diagonal()))*1e-10
    #print(a)
    #print(L)
   # L=L+np.diag(a)
    if np.all(L.diagonal()>0):
        return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()
    else:
        print(L.diagonal())
        raise ValueError(
            'L has negative values on diagonal {}'.format(L.diagonal())
        )

def log_like(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats
    
    Return
    ----------
    unnormalized probability : float 
        
    """
    mn,var=predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()
    
    exp_var=np.diag(y_exp_variance)
    
    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return mvn_loglike(delta_y,total_var)        
        
def log_posterior(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats
    
    Return
    ----------
    unnormalized probability : float 
    """
    
    mn,var=predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()
    
    exp_var=np.diag(y_exp_variance)
    
    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return log_prior(model_parameters) + mvn_loglike(delta_y,total_var)

# -------------------------------
# MCMC Sampling and Diagnostics
# -------------------------------
#def run_mcmc_sampling(design_min, design_max, ndim=17, nwalkers=200, ntemps=20, nburnin=500, niterations=1000, nthin=10, nthreads=3):
#    pos0 = design_min + (design_max - design_min) * np.random.rand(ntemps, nwalkers, ndim)
#    sampler = ptemcee.Sampler(nwalkers, ndim, log_like, log_prior, ntemps, threads=nthreads, Tmax=np.inf)
#    t0 = time.time()
#    print("Running burn-in...")
#    for pos, lnprob, lnlike in sampler.sample(pos0, iterations=nburnin, adapt=True):
#        pass
#    sampler.reset()
#    print("Running production MCMC...")
#    for pos, lnprob, lnlike in sampler.sample(pos, iterations=niterations, thin=nthin, adapt=True):
#        pass
#    t1 = time.time()
#    sampling_time = t1 - t0
#    print(f"MCMC sampling completed in {sampling_time:.2f} seconds.")
#    samples = sampler.chain[0, :, :, :].reshape((-1, ndim))
#    return samples, sampling_time

# MCMC sampling function
def run_mcmc_sampling(design_min, design_max, ndim=17, nwalkers=200,
                       ntemps=20, nburnin=500, niterations=1000,
                       nthin=10, nthreads=3):
    
    pos0 = design_min + (design_max - design_min) * np.random.rand(ntemps, nwalkers, ndim)
    
    sampler = ptemcee.Sampler(nwalkers, ndim, log_like,
                               log_prior, ntemps,
                               threads=nthreads,
                               Tmax=np.inf)
    
    t0 = time.time()
    
    print("Running burn-in...")
    for pos, lnprob, lnlike in sampler.sample(pos0,
                                               iterations=nburnin,
                                               adapt=True):
        pass
    
    sampler.reset()
    
    print("Running production MCMC...")
    for pos, lnprob, lnlike in sampler.sample(pos,
                                               iterations=niterations,
                                               thin=nthin,
                                               adapt=True):
        pass
    
    t1 = time.time()
    
    sampling_time = t1 - t0
    print(f"MCMC sampling completed in {sampling_time:.2f} seconds.")
    
    samples = sampler.chain[0, :, :, :].reshape((-1, ndim))
    
    return samples, sampling_time

def plot_mcmc_diagnostics(samples_df):
    sns.set_context("notebook", font_scale=1.3)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(samples_df.columns[:9]):
        axes[i].plot(samples_df[col].values, color='blue', alpha=0.7)
        axes[i].set_title(f"Trace of {col}")
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(samples_df.columns[:9]):
        sns.histplot(samples_df[col].values, bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Marginal of {col}")
    plt.tight_layout()
    plt.show()
    def autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0,1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(samples_df.columns[:9]):
        x = samples_df[col].values
        lags = np.arange(1, 51)
        acs = [autocorr(x, lag) for lag in lags]
        axes[i].plot(lags, acs, marker='o', linestyle='-')
        axes[i].set_title(f"Autocorr of {col}")
        axes[i].set_xlabel("Lag")
        axes[i].set_ylabel("ACF")
    plt.tight_layout()
    plt.show()
    
# -------------------------------
# Data Analysis: Transport Coefficients
# -------------------------------
    
def zeta_over_s(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2)
zeta_over_s = np.vectorize(zeta_over_s)

def eta_over_s(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow*(T-T_k)
    else:
        y = etas_k + ahigh*(T-T_k)
    if y > 0:
        return y
    else:
        return 0.
eta_over_s = np.vectorize(eta_over_s)

# -------------------------------
# Data Analysis: Plotting Transport Coefficients
# -------------------------------

@plot
def plot_shear_posterior(save_folder):
    Tt = np.linspace(0.1, 0.4, 100)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6),
                             sharex=False, sharey=False, constrained_layout=True, dpi=900)
    fig.suptitle("Specific shear viscosity posterior", wrap=True)

    # True temperature dependence of the viscosity

    #[T_k, alow, ahigh, etas_k] = truth[[7,8,9,10]]
    #true_shear = eta_over_s(Tt, T_k, alow, ahigh, etas_k)


    prior_etas = []

    for row in np.random.uniform(design_min, design_max,(10000,17))[:,[7,8,9,10]]:
        [T_k, alow, ahigh, etas_k] = row
        prior=[]
        for T in Tt:
            prior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        prior_etas.append(prior)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_etas,[0,5,20,80,95,100], axis=0)

    n_samples_posterior = 1400
    #prune = 10
    posterior_etas = []
    
    for row in posterior_Grad_df.iloc[0:n_samples_posterior,[7,8,9,10]].values:
        [T_k, alow, ahigh, etas_k] = row
        posterior=[]
        for T in Tt:
            posterior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        posterior_etas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per20_pr,per80_pr,color='gray', alpha=0.3, label='60% C.I. Prior')
    axes.fill_between(Tt,per20,per80, color='blue', alpha=0.3, label='60% C.I. Grad')

    for row in posterior_CE_df.iloc[0:n_samples_posterior,[7,8,9,10]].values:
        [T_k, alow, ahigh, etas_k] = row
        posterior=[]
        for T in Tt:
            posterior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        posterior_etas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    #axes.fill_between(Tt,per5,per95,color='red', alpha=0.2, label=r'90% C.I. CE')
    axes.fill_between(Tt,per20,per80, color='red', alpha=0.3, label='60% C.I. CE')

    for row in posterior_PTB_df.iloc[0:n_samples_posterior,[7,8,9,10]].values:
        [T_k, alow, ahigh, etas_k] = row
        posterior=[]
        for T in Tt:
            posterior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        posterior_etas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    #axes.fill_between(Tt,per5,per95,color='green', alpha=0.2, label=r'90% C.I. PTB')
    axes.fill_between(Tt,per20,per80, color='green', alpha=0.3, label='60% C.I. PTB')

    axes.legend(loc='upper left')
    #axes.set_ylim(0,1.2)
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\eta/s$')
    save_fig(fig, "shear_posterior_plot.pdf", save_folder)
    return fig

@plot
def plot_bulk_posterior(save_folder):
    
    Tt = np.linspace(0.1, 0.4, 100)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6),
                             sharex=False, sharey=False, constrained_layout=True, dpi=900)
    fig.suptitle("Specific bulk viscosity posterior", wrap=True)

   # True temperature dependece of the viscosity
   #[zmax, T0, width, asym] = truth[[11,12,13,14]]
   #true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)


    prior_zetas = []

    for row in np.random.uniform(design_min, design_max,(10000,17))[:,[11,12,13,14]]:
        [zmax, T0, width, asym] = row   
        prior=[]
        for T in Tt:
            prior.append(zeta_over_s(T,zmax, T0, width, asym))
        prior_zetas.append(prior)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_zetas,[0,5,20,80,95,100], axis=0)

    n_samples_posterior = 1400
    #prune = 10
    posterior_zetas = []
    
    for row in posterior_Grad_df.iloc[0:n_samples_posterior,[11,12,13,14]].values:
        [zmax, T0, width, asym] = row   
        posterior=[]
        for T in Tt:
            posterior.append(zeta_over_s(T,zmax, T0, width, asym))
        posterior_zetas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per20_pr,per80_pr,color='gray', alpha=0.3, label='60% C.I. Prior')
    axes.fill_between(Tt,per20,per80, color='blue', alpha=0.3, label='60% C.I. Grad')


    for row in posterior_CE_df.iloc[0:n_samples_posterior,[11,12,13,14]].values:
        [zmax, T0, width, asym] = row   
        posterior=[]
        for T in Tt:
            posterior.append(zeta_over_s(T,zmax, T0, width, asym))
        posterior_zetas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)
    #axes.fill_between(Tt,per5,per95,color='red', alpha=0.2, label='90% C.I. CE')
    axes.fill_between(Tt,per20,per80, color='red', alpha=0.3, label='60% C.I. CE')

    for row in posterior_PTB_df.iloc[0:n_samples_posterior,[11,12,13,14]].values:
        [zmax, T0, width, asym] = row   
        posterior=[]
        for T in Tt:
            posterior.append(zeta_over_s(T,zmax, T0, width, asym))
        posterior_zetas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)
    #axes.fill_between(Tt,per5,per95,color='green', alpha=0.2, label='90% C.I. PTB')
    axes.fill_between(Tt,per20,per80, color='green', alpha=0.3, label='60% C.I. PTB')

    axes.legend(loc='upper right')
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\zeta/s$')
    save_fig(fig, "shear_posterior_plot.pdf", save_folder)
    return fig

# -------------------------------
# Data Analysis: Universality ratio prior and posterior
# -------------------------------

@plot
def plot_universality_ratio_prior_posterior(
    exp_u_xt,                 # shape (n_cent, n_xT)
    prior_u_xt,               # array (n_design, n_cent, n_xT)
    post_u_xt,                # array (n_post, n_cent, n_xT)
    x_T_vals,                 # 1D array of length n_xT
    centrality_labels,        # list of strings
    ref_cent_index=0,         # Reference centrality index
    tgt_cent_index=-1,        # Target centrality index
    prior_percentiles=(5,95), 
    post_percentiles=(5,95),
    colors=('gray','blue','black'),
    delta_f=0,
    figsize=(8,4), dpi=600
):
    """
    Plot universality ratio R(x_T) = U(target)/U(reference) 
    for prior, posterior, and experiment.
    """
    # Experimental ratio
    R_exp = exp_u_xt[tgt_cent_index] / exp_u_xt[ref_cent_index]

    # Prior ratios
    R_prior = prior_u_xt[:, tgt_cent_index, :] / prior_u_xt[:, ref_cent_index, :]

    # Posterior ratios
    R_post = post_u_xt[:, tgt_cent_index, :] / post_u_xt[:, ref_cent_index, :]

    # Calculate percentiles
    p_low, p_high = np.percentile(R_prior, prior_percentiles, axis=0)
    q_low, q_high = np.percentile(R_post, post_percentiles, axis=0)
    q_med = np.median(R_post, axis=0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Prior band
    ax.fill_between(x_T_vals, p_low, p_high, 
                   color=colors[0], alpha=0.3,
#                   label=f"Prior {prior_percentiles[0]}–{prior_percentiles[1]}%")
                   label=f"Prior {prior_percentiles[1] - prior_percentiles[0]}% C.I.")
    # Posterior band and median
    ax.fill_between(x_T_vals, q_low, q_high,
                   color=colors[1], alpha=0.3,
#                   label=f"Posterior {post_percentiles[0]}–{post_percentiles[1]}%")
                    label=f"Posterior {post_percentiles[1] - post_percentiles[0]}% C.I.") 
    ax.plot(x_T_vals, q_med, color=colors[1], lw=1.5, label="Posterior median")
    
    # Experimental data
    ax.plot(x_T_vals, R_exp, 'o', color=colors[2], markersize=4,
           label="ALICE PbPb 2.76 TeV")
    
    # Reference line
    ax.axhline(1.0, color='k', linestyle='-', lw=1)
    
    # Formatting
    ax.set_xscale("log")
#    ax.set_xlim(x_T_vals.min(), x_T_vals.max())
    ax.set_xlim(0, x_T_vals.max()+0.5)
    ax.set_ylim(0.4, 1.5)
    ax.set_xlabel(r"$x_T = p_T / \langle p_T\rangle$")
    ax.set_ylabel(f"$U / U({centrality_labels[ref_cent_index]}\%)$")
    ax.set_title(f"Universality Ratio ({centrality_labels[tgt_cent_index]}): {idf_label_short[delta_f]}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    return fig

# -------------------------------
# MAIN: Closure Test, Posterior Sampling, and Additional Validations
# -------------------------------
if __name__ == "__main__":
    idf = 0  # viscous correction index (Bayesian pipeline)
    main_fig_folder = "Bayesian_figures"
    visc_folder = os.path.join(main_fig_folder, f"{idf}_{idf_label_short[idf]}")
    folders = {
        "prior": os.path.join(visc_folder, "prior"),
        "PCA": os.path.join(visc_folder, "PCA"),
        "sensitivity_plot": os.path.join(visc_folder, "sensitivity_plot"),
        "emulators_validation": os.path.join(visc_folder, "emulators_validation"),
        "corner_PC_scores": os.path.join(visc_folder, "corner_PC_scores"),
        "normality_check": os.path.join(visc_folder, "normality_check")
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # ---- Load Training Data ----
    design = load_design(idf)
#    print("Training design shape:", design.shape)
    Y_sim = load_moment_data('Bayesian_data', idf)
#    print("Raw simulation data shape:", Y_sim.shape)
    Y_flat = prepare_simulation_data(Y_sim)
#    print("Flattened simulation data shape:", Y_flat.shape)
#    print(Y_flat)
    
    # ---- Prior Predictive Plot: Observables vs. x_T, Separated by Centrality ----
#    fig_prior = plot_prior_observables_separated(idf, save_folder=folders["prior"])
#    print("Saved prior predictive observables plot.")
    
    # ---- Prior Plot: Separated by Centrality ----
#    prior_folder = make_save_dir(main_fig_folder, visc_folder, "prior")
#    fig_prior = plot_prior_scaled_spectra_by_centrality(idf, folders["prior"])
#    print("Saved prior plot separated by centrality.")
    
    # ---- Prior Plot: All Centralities onto a single plot ----
#    fig_prior = plot_prior_scaled_spectra_overlay(idf, folders["prior"], exp=True)
#    print("Saved prior plot all centralities onto a single plot.")
    
    # Scaling the data to be zero mean and unit variance for each observables
    SS  =  StandardScaler(copy=True)

    # Singular Value Decomposition
    u, s, vh = np.linalg.svd(SS.fit_transform(Y_flat), full_matrices=True) # scaling Y_flat, covariance matrix
#    print(f'shape of u {u.shape} shape of s {s.shape} shape of vh {vh.shape}')
    
    # whiten and project data to principal component axis (only keeping first 6 PCs)
    pc_tf_data=u[:,0:6] * math.sqrt(u.shape[0]-1)
#    print(f'Shape of PC transformed data {pc_tf_data.shape}')
    
    # Scale Transformation from PC space to original data space
    inverse_tf_matrix= np.diag(s[0:6]) @ vh[0:6,:] * SS.scale_.reshape(1,287)/ math.sqrt(u.shape[0]-1)
    
    # Reconstruct observables from PCA
    reconstructed_data = pc_tf_data @ inverse_tf_matrix + SS.mean_
    #    scaler, pca, Z, Y_reconstructed = fit_pca_model(Y_flat, variance_threshold=0.99)
     
#    print(Z[:,0].shape)
#    print(Z.shape, Y_reconstructed.shape)
#    fig_var = plot_explained_variance(pca, n_pc_to_plot=10)
#    save_fig(fig_var, "explained_variance.pdf", folders["PCA"])
    
    # ---- PCA Validation: Reconstruction Plot & Corner Plot of PC Scores ----
#    fig_rec = plot_reconstruction(Y_flat, scaler, pca, n_pc_retained=10, sample_index=100, save_folder=folders["PCA"])
#    fig_rec = plot_reconstruction(Y_flat, scaler, pca, n_pc_retained=55, sample_index=, save_folder=folders["PCA"])
#    print("Saved reconstruction plot for sample 0.")
#    fig_corner = corner_pc_scores(Z[:, :5], save_folder=folders["corner_PC_scores"])
#    print("Saved corner plot of PC scores.")
    
    # ---- Scatter Checks for Normality of Original Observables ----
    # (Select a few pairs of observable indices to check for roughly elliptical scatter)
#    sample_pairs = [(0,1), (2,3), (4,5), (6,7)]
#    fig_scatter = plot_original_scatter_checks(Y_flat, sample_pairs=sample_pairs, save_folder=folders["normality_check"])
#    print("Saved scatter plots for normality check.")
    
    # ---- Train Emulators on Training Data ----
    n_pc_train = 6
#    pc_tf_data
#    pc_tf_data = Z[:, :n_pc_train]
#    inverse_tf_matrix = pca.components_[:n_pc_train, :]
    Emulators = train_emulators_for_pcs(idf, pc_tf_data, n_pc_train=n_pc_train,
                                        emulator_output_file=EMULATOR_FILE, train_emulators=False)
    
#    design_min = design.min(axis=0).values
#    design_max = design.max(axis=0).values

    scaler = SS

    # ---- Prior Predictive Plot ----
#    fig_pred_prior = plot_prior_predictive_separated(1000, design_min, design_max, Emulators, 
#                                                     inverse_tf_matrix, SS, folders["prior"])
#    print("Saved prior predictive observables plot.")

    # ---- Prior Predictive Plot: Full prior vs. Design points ----
#    plot_prior_CI(
#    1000,         # shape (n_prior_samples, n_cent, n_xt)
#    show_exp=False,         # if True, overlays experimental data
#    bound_min=5,
#    bound_max=95,
#    save_folder=folders["prior"]
#)
#    plot_prior_CI(
#    1000,         # shape (n_prior_samples, n_cent, n_xt)
#    show_exp=True,         # if True, overlays experimental data
#    bound_min=5,
#    bound_max=95,
#    save_folder=folders["prior"]
#)
    
    # ---- Emulator Validation: Leave-One-Out Cross-Validation ----

    # ---- Emulator Validation: Leave-P-Out Cross-Validation ----
#    rmse_dict, fig_lpo = leave_p_out_validation(design, pc_tf_data, P=10, n_splits=5, save_folder=folders["emulators_validation"])
#    print("LPOCV RMSE per PC:", rmse_dict)
    
    # ---- Emulator Validation: True vs. Predicted Observables by Centrality ----
#    fig_val = plot_emulator_validation_by_centrality(design, Y_flat, Emulators, inverse_tf_matrix, scaler, save_folder=folders["emulators_validation"])
#    print("Saved emulator validation plot")
    
    # ---- Sensitivity Plot at Peak x_T ----
#    fig_sens = plot_sensitivity(idf, Emulators, inverse_tf_matrix, SS, save_folder=folders["sensitivity_plot"])
#    print("Saved sensitivity plot at peak x_T.")
    
    # ---- Sensitivity Plot for the first two PCs ----
#    fig_sens_pcs = plot_sensitivity_pcs(idf, Emulators, inverse_tf_matrix, SS, save_folder=folders["sensitivity_plot"])
#    print("Saved sensitivity plot for the first two PCs.")
    
    # ---- MCMC Sampling & Diagnostics for Closure Test ----
    # Uncomment below to run posterior sampling if pseudo-experimental data are defined.
    # For a closure test, use the first design point as the true parameters.
#    scaler = SS
#    theta_true = design.iloc[0].values
#    y_exp, y_exp_var = predict_observables(theta_true, Emulators, inverse_tf_matrix, scaler)
#    y_exp_variance = np.diag(y_exp_var)
#    print("Pseudo-experimental data generated.")
    
    # Uncomment below to run posterior sampling if experimental data are defined.
    # Using experimental data
#    _, y_exp, y_exp_err = load_experimental()  # each: (7,41)
#    y_exp_variance = np.square(y_exp_err)
    
#    y_exp = y_exp.flatten()
#    y_exp_variance = y_exp_variance.flatten()
    
#    print("Using experimental data")
    
#    samples, sampling_time = run_mcmc_sampling(design_min, design_max, ndim=17, nwalkers=200,
#                                            ntemps=20, nburnin=100, niterations=75, nthin=10, nthreads=3)

#    print(f"Total MCMC sampling time: {sampling_time:.2f} seconds")
    
#    samples_df = pd.DataFrame(samples, columns=model_param_dsgn)
#    samples_df.to_csv(os.path.join("posterior", f"mcmc_chain_{idf_label_short[idf]}.csv"), index=False)
#    print (f"The posterior samples for {idf_label_short[idf]} model is saved!")
#    plot_mcmc_diagnostics(samples_df)
#    print("MCMC diagnostics complete.")

    # ---- Data Analysis: analyzing the posterior ----
# comparing two posteriors: integrated observables calibration vs. scaled spectra calibration 
#    posterior_Grad_df = pd.read_csv("posterior/mcmc_chain_Grad.csv")
#    posterior_CE_df = pd.read_csv("posterior/mcmc_chain_CE.csv")
#    posterior_PTB_df = pd.read_csv("posterior/mcmc_chain_PTB.csv")
#    posterior_PTM_df = pd.read_csv("posterior/mcmc_chain_PTM.csv")
#    posterior_Grad_org_df = pd.read_csv("new_LHC_posterior_samples.csv")
    
    # ---- Data Analysis (results interpretation): Universality ratio - prior and posterior ----
        # Load and prepare data
#    xt_exp, u_exp, err_exp = load_experimental()
#    Y_flat = prepare_simulation_data(load_moment_data('Bayesian_data', idf))
    
    # Reshape prior predictions
#    prior_u_xt = Y_flat.reshape(-1, 7, 41)  # (n_design, n_cent, n_xT)

    # Generate posterior predictions
#    post_samples = posterior_PTM_df.values  # Get MCMC samples
#    post_u_xt = np.array([predict_observables(theta, Emulators, inverse_tf_matrix, SS)[0]
#                      for theta in post_samples])
#    post_u_xt = post_u_xt.reshape(-1, 7, 41)  # (n_post, n_cent, n_xT)

    # Create and save plot
#    fig = plot_universality_ratio_prior_posterior(
#        exp_u_xt=u_exp,
#        prior_u_xt=prior_u_xt,
#        post_u_xt=post_u_xt,
#        x_T_vals=xt_exp[0],  # Assuming same x_T bins for all centralities
#        centrality_labels=exp_centrality_labels,
#        tgt_cent_index=5,  # 50-60% centrality
#        colors=('gray', color_map[idf], 'black'), # colors for each df
#        delta_f=idf
#    )
#    save_fig(fig, "universality_ratio.pdf", folders["prior"])
    
    # ---- Data Analysis (results interpretation): Observables predictions ----
#    prior and posterior overlaid - Observables drawn from posterior with prior overlaid
    
    # ---- Data Analysis (results interpretation): Corner plots ----
#    corner plots: single df or comparisons btw two or more     
    
    # ---- Data Analysis (results interpretation): Transport coefficients ----
#    specific shear and zeta viscosities: single df or comparisons btw two or more 
    
#    print("All figures have been saved under", main_fig_folder)
