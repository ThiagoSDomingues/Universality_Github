import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load all four chains into a dict:
 
path = '/sysroot/home/rafaela/Projects/JETSCAPE_studies/Universality_pion_scale_spectra/posterior'

Grad_posterior = pd.read_csv(f"{path}/mcmc_chain_Grad.csv") 
CE_posterior = pd.read_csv(f"{path}/mcmc_chain_CE.csv") 
PTM_posterior = pd.read_csv(f"{path}/mcmc_chain_PTM.csv")
PTB_posterior = pd.read_csv(f"{path}/mcmc_chain_PTB.csv")

posterior_dfs = {
    0: Grad_posterior,
    1: CE_posterior,
    2: PTM_posterior,
    3: PTB_posterior
}
idf_names = ['Grad','CE','PTM','PTB']

def plot_pt_integrated_posterior_chain(
    posterior_df,
    predict_observables,
    obs_groups,
    obs_cent_list,
    index,
    y_exp,
    y_exp_variance,
    obs_tex_labels,
    obs_group_labels,
    colors,
    height_ratios,
    system='Pb-Pb-2760'
):
    """
    For a single particlization model, take its posterior-chain DataFrame,
    run predict_observables on each sample, and plot the pt-integrated
    observables with 90% credible bands + median + experiment.

    Parameters
    ----------
    posterior_df : pd.DataFrame
      shape = (n_samples, n_params)
    predict_observables : func(params) -> (mm, vv)
      mm.flatten() is length N_obs_total = sum over all obs of n_cent_bins.
    obs_groups : OrderedDict[str, List[str]]
      grouping of observable keys by row (e.g. 'yields', 'mean_pT', …).
    obs_cent_list : dict[system][obs] -> array of shape (n_cent,2)
    index : dict[obs] -> (start, end) slice into mm.
    y_exp : 1D array, length = N_obs_total
    y_exp_variance : 1D array, same length
    obs_tex_labels : dict[obs]->LaTeX label
    obs_group_labels : dict[group]->y-axis label
    colors : list of colors, one per observable within each group
    height_ratios : list of 4 floats (one per row)
    system : str, e.g. 'Pb-Pb-2760'

    Returns
    -------
    fig : matplotlib.Figure
    """
    # number of posterior samples
    params_matrix = posterior_df.values
    n_samples = len(params_matrix)
    n_groups = len(obs_groups)
    column = 1

    # set up figure
    fig, axes = plt.subplots(
        nrows=n_groups, ncols=column,
        figsize=(6, 8),
        squeeze=False,
        gridspec_kw={'height_ratios': height_ratios},
        dpi=300
    )

    # precompute all predictions: we only need the mm part (ignore vv)
    # mm_all: shape (n_samples, N_obs_total)
    mm_all = []
    for params in params_matrix:
        mm, vv = predict_observables(params)
        mm_all.append(mm.flatten())
    mm_all = np.vstack(mm_all)  # (n_samples, N_obs_total)

    # loop over groups
    for row, (obs_group, obs_list) in enumerate(obs_groups.items()):
        ax = axes[row][0]
        ax.tick_params(labelsize=9)

        # get slice predictions for each obs in this row,
        # compute median & 5–95 band
        for (obs, color) in zip(obs_list, colors):
            start, end = index[obs]
            # shape = (n_samples, n_cent_bins)
            preds = mm_all[:, start:end]
            # apply scaling if needed
            scale = 1.0
            if obs_group == 'yields':
                if obs == 'dET_deta':    scale = 5.0
                if obs == 'dNch_deta':   scale = 2.0

            # percentiles + median
            lower = np.percentile(preds, 5, axis=0) * scale
            upper = np.percentile(preds, 95, axis=0) * scale
            median = np.median(preds, axis=0) * scale

            # centrality x‐axis
            xbins = np.array(obs_cent_list[system][obs])
            x = (xbins[:,0] + xbins[:,1]) / 2.0

            # plot band + median
            ax.fill_between(x, lower, upper, color=color, alpha=0.2)
            ax.plot(x, median, color=color, lw=1.5, label=obs_tex_labels[obs])

            # experimental data
            exp_mean = y_exp[start:end] * scale
            exp_err  = np.sqrt(y_exp_variance[start:end]) * scale
            ax.errorbar(
                x, exp_mean, exp_err,
                fmt='v', color='black', markersize=4, elinewidth=1
            )

        # styling per your original
        if obs_group == 'yields':
            ax.set_yscale('log')
            ax.set_title(f"{idf_names[idf]} Posterior 90% C.I. & median", fontsize=20)
        ax.set_ylabel(obs_group_labels[obs_group], fontsize=14)
        ax.set_xlim(0, 70)

        # y‐limits by group
        if obs_group == 'yields':    ax.set_ylim(1,   1e5)
        if obs_group == 'mean_pT':   ax.set_ylim(0.,  2)
        if obs_group == 'fluct':     ax.set_ylim(0.,  0.06)
        if obs_group == 'flows':     ax.set_ylim(0.,  0.15)

        # legend
        leg = ax.legend(
            fontsize=10, borderpad=0, labelspacing=0,
            handlelength=1, handletextpad=0.2
        )
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_alpha(1.0)

        # xlabel only on last row
        if row == n_groups - 1:
            ax.set_xlabel('Centrality %', fontsize=18)

    plt.tight_layout()
    # if you have a set_tight helper:
    try:
        set_tight(fig, rect=[0,0,1,1])
    except NameError:
        pass
    return fig

# assume you have:
#   Grad_posterior, CE_posterior, PTM_posterior, PTB_posterior
#   predict_observables, obs_groups, obs_cent_list, index,
#   y_exp, y_exp_variance, obs_tex_labels, obs_group_labels, colors,
#   height_ratios = [2,1.4,1.4,0.7]
height_ratios = [2, 1.4, 1.4, 0.7]
idf = 0 
fig = plot_pt_integrated_posterior_chain(
    posterior_dfs[idf],
#    posterior_df,
    predict_observables,
    obs_groups,
    obs_cent_list,
    index,
    y_exp,
    y_exp_variance,
    obs_tex_labels,
    obs_group_labels,
    colors,
    height_ratios,
    system='Pb-Pb-2760'
)

fig.savefig(
    "pt_integrated_posterior_Grad.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()

#plot_sim_exp_discrepancy_all_idf(
#    Ymodels, y_exp, y_exp_variance,
#    obs_groups, index,
#    df_choices=[0,1,2,3],
#    color_idf=color_idf,
#    ls_idf=ls_idf
#)
