### Author: OptimusThi 
"""
Temporary file to plot figures from the paper. I will organize it and modify it!
"""

from matplotlib import pyplot as plt
# Grad posterior 
### centralities to plot: 0-5%, 10-20%, 50-60%
### add MAP for Pb-Pb 2.76 TeV

# Plotting style that Gardim used:
#Plot parameters
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1.2  # default 0.8
plt.rcParams["ytick.major.width"] = 1.2  # default 0.8
plt.rcParams["xtick.minor.width"] = 1  # default 0.6
plt.rcParams["ytick.minor.width"] = 1  # default 0.6
plt.rcParams["axes.linewidth"]    = 1  # default 0.8 
plt.rcParams["lines.linewidth"]   = 1  # default 1.5 
plt.rcParams["xtick.major.pad"] = 4
plt.rcParams["ytick.major.pad"] = 4
plt.rcParams["xtick.minor.pad"] = 4
plt.rcParams["ytick.minor.pad"] = 4
plt.rcParams["legend.handletextpad"] = 0.0
plt.rcParams["xtick.direction"]     = 'in'
plt.rcParams["ytick.direction"]     = 'in'
plt.rcParams["xtick.minor.visible"] = 'True'
plt.rcParams["ytick.minor.visible"] = 'True'

plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')

# fig
fontsize = 23
qc       = 7  # quantidade de centralidades
particulas_plot = ["pi", "ka", "pr"]

cor  = ['red', 'black', 'blue', 'green', 'orange', 'cyan', 'gray', 'pink', 'gold', 'chocolate']
simb = ['o', '^', 's', 'v', 'd', 'H', '*', 'h', 'p', 'P']
rot  = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%']
rot  = [r.replace('%', r'\%') for r in rot]

def configure_axis(ax, fontsize):
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelsize=fontsize)

### Function to plot the Fig. without MAP parameters
#@plot
def plot_qm_proc(posterior_file, Emulators, inverse_tf_matrix, scaler, save_folder):
    """
    Load posterior samples from a CSV file, use the emulator surrogate to predict the full observable
    for each sample, and then for each centrality plot the 90% credible interval and median 
    over x_T. Experimental data are overlaid.
    
    Parameters:
      posterior_file : str
          Path to the CSV file containing posterior samples.
      Emulators : list
          List of trained GP emulators.
      inverse_tf_matrix : np.ndarray
          Matrix to inverse transform from PC space to original observable space.
      scaler : StandardScaler
          The scaler used in the PCA pipeline.
      save_folder : str
          Folder to save the resulting figure.
    
    Returns:
      fig : matplotlib.figure.Figure object.
    """
    # Load the original posterior samples; try first without index
    orig_data_df = pd.read_csv("new_LHC_posterior_samples.csv")
    orig_df = orig_data_df.iloc[:, :-1]    

    # Load the posterior samples; try first without index
    samples_df = pd.read_csv(posterior_file)
    
    # Read prior min/max values from my design-range file
    data_path = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat' 
    prior_df = pd.read_csv(data_path, index_col=0)
    
    # Extracting the min and max prior values
    design_min = prior_df['min'].values
    design_max = prior_df['max'].values

    np.random.seed(1) # for reproducibility

    # Collect scaled spectra predictions for each prior sample
    pr_predictions = []
    orig_predictions = []
    post_predictions = []

    n_pr_samples = 1000
        
    # Extracting the posterior samples values 
    posterior_samples = samples_df.values  # shape (n_samples, 17)
    orig_samples = orig_df.values
    
    # Load experimental data
    xt_exp, u_xt_exp, err_u_xt = load_experimental()  # each: (7, 41)
    
    # Looping over all prior samples: supposing a uniform prior
    for params in np.random.uniform(design_min, design_max, (n_pr_samples, 17)):
        y_pred, cov_pred = predict_observables(params, Emulators, inverse_tf_matrix, scaler)
        pr_predictions.append(y_pred.flatten())
    pr_predictions = np.array(pr_predictions) # shape (n_pr_samples, n_obs)
    
    # Use the emulator surrogate to predict observables for each posterior sample
    for theta in orig_samples:
        # theta is expected to be a 17-D vector.
        if theta.shape[0] != 17:
            raise ValueError("Posterior sample does not have 17 elements after adjustment.")
        y_orig_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
        orig_predictions.append(y_orig_pred)
    orig_predictions = np.array(orig_predictions)  # shape (n_samples, n_obs)
    
    
    # Use the emulator surrogate to predict observables for each posterior sample
    for theta in posterior_samples:
        # theta is expected to be a 17-D vector.
        if theta.shape[0] != 17:
            raise ValueError("Posterior sample does not have 17 elements after adjustment.")
        y_post_pred, _ = predict_observables(theta, Emulators, inverse_tf_matrix, scaler)
        post_predictions.append(y_post_pred)
    post_predictions = np.array(post_predictions)  # shape (n_samples, n_obs)
    
    
    # Reshape predictions into (n_pr_samples, 7, 41)
    pr_n_cent = len(exp_centrality_labels)
    pr_n_xt = len(xt_exp[0])    
    pr_predictions_reshaped = pr_predictions.reshape(n_pr_samples, pr_n_cent, pr_n_xt)
    
    # Reshape predictions into (n_samples, 7, 41)
    n_samples = orig_predictions.shape[0]
    n_cent = len(exp_centrality_labels)  # should be 7
    n_xt = len(xt_exp[0]) # should be 41
    orig_predictions_reshaped = orig_predictions.reshape(n_samples, n_cent, n_xt)
    
    # Reshape predictions into (n_samples, 7, 41)
    n_samples = post_predictions.shape[0]
    n_cent = len(exp_centrality_labels)  # should be 7
    n_xt = len(xt_exp[0]) # should be 41
    post_predictions_reshaped = post_predictions.reshape(n_samples, n_cent, n_xt)
    
    centrality_indices = [0, 6]   # 0-5%, 50-60%
    n_cent = len(centrality_indices) # 3 centralities
    
    # prepare figure grid (1 row x 3 cols)
#    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True, 
#                            sharex='col', dpi=600)
#    axs = axs.flatten()
#    fontsize=23

    # --- Prepare figure grid: 2 rows x 2 cols (top = spectra, bottom = ratio)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=600,
                            sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    # Separate top and bottom axes
    axs_top = axs[0, :]
    axs_bot = axs[1, :]

    fontsize = 23

    
    # centrality loop
#    for cent in range(n_cent):
#        ax = axs[cent]
        # Compute credible intervals: 5th, 50th, 95th percentiles
#        pr_env = np.percentile(pr_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)
#        ax.fill_between(xt_exp[centrality_indices[cent], :], pr_env[0, :], pr_env[2, :], color='gray', alpha=0.3, label="Prior 90% C.I.")
        
#        orig_env = np.percentile(orig_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)
#        ax.fill_between(xt_exp[centrality_indices[cent], :], orig_env[0, :], orig_env[2, :], color='orange', alpha=0.4, label="Original Posterior 90% C.I.")
#        ax.plot(xt_exp[centrality_indices[cent], :], env[1, :], 'k-', lw=.6, label=f"{idf_label_short[idf]} Posterior Median")
        
#        post_env = np.percentile(post_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)
#        ax.fill_between(xt_exp[centrality_indices[cent], :], post_env[0, :], post_env[2, :], color=color_map[idf], alpha=0.4, label="Grad Posterior 90% C.I.")

#        ax.errorbar(xt_exp[centrality_indices[cent], :], u_xt_exp[centrality_indices[cent], :], yerr=err_u_xt[centrality_indices[cent], :],
#                    fmt=simb[centrality_indices[cent]], color=cor[centrality_indices[cent]], capsize=3, label="ALICE PbPb 2.76 TeV")
#        ax.set_title(f"Centrality {exp_centrality_labels[centrality_indices[cent]]}", fontsize=fontsize+10)
#        ax.set_xlabel(r"$x_T$", fontsize=fontsize+5)
#        ax.set_xscale("log")
#        if centrality_indices[cent] == 0: # only the first column has a legend
#            ax.set_ylabel(r"$U(x_T)$", rotation=90, fontsize=fontsize+10, labelpad=30)
#            ax.legend(loc='best', frameon=False, fontsize=20)

    for cent in range(n_cent):
        ax = axs_top[cent]
        ax_ratio = axs_bot[cent]

        # Compute credible intervals: 5th, 50th, 95th percentiles
        pr_env = np.percentile(pr_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)
        orig_env = np.percentile(orig_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)
        post_env = np.percentile(post_predictions_reshaped[:, centrality_indices[cent], :], [5, 50, 95], axis=0)

        # --- Top panel: spectra ---
        ax.fill_between(xt_exp[centrality_indices[cent], :], pr_env[0, :], pr_env[2, :],
                        hatch='///', edgecolor='black', facecolor='gray', alpha=0.3, label="Prior 90% C.I.")
        ax.fill_between(xt_exp[centrality_indices[cent], :], orig_env[0, :], orig_env[2, :],
                        color='orange', alpha=0.4, label="Original Posterior 90% C.I.")
        ax.fill_between(xt_exp[centrality_indices[cent], :], post_env[0, :], post_env[2, :],
                        color='blue', alpha=0.4, label="Grad Posterior 90% C.I.")

        ax.errorbar(xt_exp[centrality_indices[cent], :], u_xt_exp[centrality_indices[cent], :],
                    yerr=err_u_xt[centrality_indices[cent], :],
                    fmt=simb[centrality_indices[cent]], color=cor[centrality_indices[cent]],
                    capsize=3, label=f"ALICE PbPb 2.76 TeV {exp_centrality_labels[centrality_indices[cent]]}")
        
#        ax.set_title(f"Centrality {exp_centrality_labels[centrality_indices[cent]]}",
#                     fontsize=fontsize+6)
#        ax.set_ylabel(r"$U(x_T)$", fontsize=fontsize+4)
        configure_axis(ax, fontsize)
        ax.legend(loc='best', frameon=False, fontsize=16) 

        if cent == 0:
#            ax.legend(loc='best', frameon=False, fontsize=16)
            ax.set_ylabel(r"$U(x_T)$", fontsize=fontsize+8)
    
        # --- Bottom panel: ratio (prior / exp data) ---
        pr_ratio_low = pr_env[0, :] / u_xt_exp[centrality_indices[cent], :]
        pr_ratio_high = pr_env[2, :] / u_xt_exp[centrality_indices[cent], :]
        
        # --- Bottom panel: ratio (posterior median / exp data) ---
#        ratio_median = post_env[1, :] / u_xt_exp[centrality_indices[cent], :]
        orig_ratio_low = orig_env[0, :] / u_xt_exp[centrality_indices[cent], :]
        orig_ratio_high = orig_env[2, :] / u_xt_exp[centrality_indices[cent], :]
        
        # --- Bottom panel: ratio (posterior median / exp data) ---
#        orig_ratio_median = post_env[1, :] / u_xt_exp[centrality_indices[cent], :]
        post_ratio_low = post_env[0, :] / u_xt_exp[centrality_indices[cent], :]
        post_ratio_high = post_env[2, :] / u_xt_exp[centrality_indices[cent], :]
        
        # Experimental relative uncertainty
        ratio_err = err_u_xt[cent, :] / u_xt_exp[cent, :]

        ax_ratio.fill_between(xt_exp[centrality_indices[cent], :],
                              pr_ratio_low, pr_ratio_high, 
                              hatch='///', edgecolor='black', facecolor='gray', alpha=0.3)
        
        ax_ratio.fill_between(xt_exp[centrality_indices[cent], :],
                              orig_ratio_low, orig_ratio_high, color='orange', alpha=0.4)
        
        ax_ratio.fill_between(xt_exp[centrality_indices[cent], :],
                              post_ratio_low, post_ratio_high, color='blue', alpha=0.4)
        
        # Add experimental uncertainty bars centered at ratio=1
        ax_ratio.errorbar(xt_exp[cent, :], np.ones_like(xt_exp[cent, :]),
                              yerr=ratio_err, fmt=simb[centrality_indices[cent]], color=cor[centrality_indices[cent]], capsize=3)
                
        ax_ratio.axhline(1.0, color='k', lw=1.2, ls='--')
        ax_ratio.set_ylim(0.5, 1.4)
        ax_ratio.set_xlabel(r"$x_T$", fontsize=fontsize)
        ax_ratio.set_xscale("log")
        if cent == 0:
            ax_ratio.set_ylabel("Model/Data", fontsize=fontsize-3)
        configure_axis(ax_ratio, fontsize-3)

    
    plt.tight_layout(h_pad=0.05, w_pad=0.2)
    plt.subplots_adjust(wspace=0, hspace=0.05) # optional: wspace=0
    # Saving       
    #save_fig(plt.gcf(), "qm_proc_post_scaled_spectra_separated.pdf", save_folder)
    plt.savefig('posterior_Grad_paper_model_to_data.pdf')
    plt.show() 
    return fig # optional: plt.gcf()

main_dir = "Bayesian_figures"

# How to use: scaled spectra posterior
posterior_folder = make_save_dir(f"{main_dir}", f"{idf}_{idf_label_short[idf]}", "posterior") # idf = 0,1,2,3 

# Generating posterior predictions for all 4 viscous corrections
fig_post = plot_qm_proc(f"posterior/mcmc_chain_{idf_label_short[idf]}.csv", Emulators, inverse_tf_matrix, scaler, posterior_folder)
