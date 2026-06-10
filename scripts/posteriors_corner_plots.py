import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

### Loading posteriors ### 
path = 'posterior'

# Only U(x_T) calibration
Grad_posterior = pd.read_csv(f"{path}/mcmc_chain_Grad.csv") 
CE_posterior = pd.read_csv(f"{path}/mcmc_chain_CE.csv") 
PTM_posterior = pd.read_csv(f"{path}/mcmc_chain_PTM.csv")
PTB_posterior = pd.read_csv(f"{path}/mcmc_chain_PTB.csv")

# Only pt-integrated observables calibration
data_df = pd.read_csv(f"{path}/new_LHC_posterior_samples.csv")
posterior_original = data_df.iloc[:, :-1]

def plot_corner_posteriors(
    df1,
    observables_to_plot,
    idf1=0,
    idf2=None,
    df2=None,
    map_parameters=None,
    truth=None,
    prune=1,
    diag_sharey=False,
    n_samples=1400,
    idf_colors={0: 'blue', 1: 'red', 2: 'magenta', 3: 'green', 4: 'orange'},
    idf_labels={0: 'Grad', 1: 'CE', 2: 'PTM', 3: 'PTB', 4: 'Original'},
    title=None,
    figsize=(12, 12)
):
    """
    Plot a corner plot (pair plot) for posterior samples from one or two viscous correction models.

    Parameters:
    - df1, df2: posterior DataFrames
    - observables_to_plot: list of indices or column names to include
    - idf1, idf2: integers (0–3), ID of the correction model
    - map_parameters, truth: optional reference values (array-like)
    - prune: thinning factor for samples
    - diag_sharey: share y-axis in diagonals
    - idf_colors, idf_labels: mapping from idf index to color/label
    - title: optional plot title
    - figsize: tuple for figure size
    """
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    # Subsample
    df1_sub = df1.iloc[:n_samples:prune, observables_to_plot].copy()
    df1_sub['Model'] = idf_labels[idf1]

    if df2 is not None and idf2 is not None:
        df2_sub = df2.iloc[:n_samples:prune, observables_to_plot].copy()
        df2_sub['Model'] = idf_labels[idf2]
        combined = pd.concat([df1_sub, df2_sub], ignore_index=True)
        palette = [idf_colors[idf1], idf_colors[idf2]]
    else:
        combined = df1_sub
        palette = [idf_colors[idf1]]

    # Plot
    g = sns.PairGrid(
        combined,
        hue='Model' if 'Model' in combined else None,
        corner=True,
        diag_sharey=diag_sharey,
        height=figsize[0] / len(observables_to_plot),
        palette=palette
    )

#    g.map_lower(sns.kdeplot, fill=True, alpha=0.5)
    g.map_lower(lambda x, y, **kwargs: sns.kdeplot(x=x, y=y, fill=True, **kwargs), alpha=0.5)
    g.map_diag(sns.kdeplot, linewidth=2, shade=True, alpha=0.7)

    # Add vertical reference lines
    for n, col in enumerate(combined.columns[:-1]):  # Exclude 'Model'
        ax = g.axes[n][n]

        if map_parameters is not None:
            ax.axvline(map_parameters[n], linestyle='--', color='black', label='MAP')
        if truth is not None:
            ax.axvline(truth[n], linestyle='--', color='orange', label='Truth')
        if n == 0 and (map_parameters is not None or truth is not None):
            ax.legend(loc='upper right', fontsize='small')

    if 'Model' in combined:
        g.add_legend()

    if title:
        plt.suptitle(title, fontsize=16)
#        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()
    else:
        plt.tight_layout()
    
    plt.savefig('corner_Grad_Grad_2.png', dpi=600)
    plt.show()

plot_corner_posteriors(
    df1=Grad_posterior,
    df2=posterior_df,
    idf1=0,
    idf2=4,
#    observables_to_plot=[0, 1, 2, 3, 4, 5, 6, 15, 16],
    observables_to_plot=[7, 8 , 9, 10, 11, 12, 13, 14],
    map_parameters=None,  # optional
    truth=None,  # optional
    prune=1,
    n_samples=1400,
    title="Grad vs Original Posterior Comparison"
)   
