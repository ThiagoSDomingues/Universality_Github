### Author: OptimusThi
"""
Script to plot pt-integrated observable predictions + experimental data.
"""

import sys
import os
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from numpy.linalg import inv
import sklearn, matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl
import scipy.stats as st
from scipy import optimize

import emcee
import ptemcee
import h5py
from scipy.linalg import lapack
from multiprocessing import Pool
from multiprocessing import cpu_count
import time
sns.set("notebook")

name="JETSCAPE_bayes"
#Saved emulator name
EMU='PbPb2760_emulators_scikit.dat'
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "JETSCAPE_bayes/Data/"

# Define folder structure 

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')
    
# Design points
design = pd.read_csv(filepath_or_buffer=data_path("PbPb2760_design"))    

# Simulation outputs at the design points
simulation = pd.read_csv(filepath_or_buffer=data_path("PbPb2760_simulation"))

X = design.values
Y = simulation.values

print( "X.shape : "+ str(X.shape) )
print( "Y.shape : "+ str(Y.shape) )

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
 
# Scaling the data to be zero mean and unit variance for each observables
SS  =  StandardScaler(copy=True)

# Singular Value Decomposition
u, s, vh = np.linalg.svd(SS.fit_transform(Y), full_matrices=True)
print(f'shape of u {u.shape} shape of s {s.shape} shape of vh {vh.shape}')   

# print the explained raito of variance
# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
#importance = pca_analysis.explained_variance_
importance = np.square(s[:10]/math.sqrt(u.shape[0]-1))
cumulateive_importance = np.cumsum(importance)/np.sum(importance)
idx = np.arange(1,1+len(importance))
ax1.bar(idx,importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")
ax2.bar(idx,cumulateive_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
plt.tight_layout()

# whiten and project data to principal component axis (only keeping first 10 PCs)
pc_tf_data=u[:,0:10] * math.sqrt(u.shape[0]-1)
print(f'Shape of PC transformed data {pc_tf_data.shape}')

# Scale Transformation from PC space to original data space
inverse_tf_matrix= np.diag(s[0:10]) @ vh[0:10,:] * SS.scale_.reshape(1,110)/ math.sqrt(u.shape[0]-1)

#This is how you can load the actual experimental data instead of pseudo experimental data
experiment=pd.read_csv(filepath_or_buffer=data_path("PbPb2760_experiment"),index_col=0)
experiment.head()

y_exp = experiment.loc['mean'].values
y_exp_variance= experiment.loc['variance'].values
print(f'Shape of the experiment observables {y_exp.shape} and shape of the experimental error variance{y_exp_variance.shape}')

from collections import OrderedDict

colors = OrderedDict([
    ('blue', '#4e79a7'),
    ('orange', '#f28e2b'),
    ('green', '#59a14f'),
    ('red', '#e15759'),
    ('cyan', '#76b7b2'),
    ('purple', '#b07aa1'),
    ('brown', '#9c755f'),
    ('yellow', '#edc948'),
    ('pink', '#ff9da7'),
    ('gray', '#bab0ac')
])

fontsize = dict(
    large=11,
    normal=10,
    small=9,
    tiny=8
)

offblack = '.15'

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic:medium',
    'mathtext.cal': 'sans',
    'font.size': fontsize['normal'],
    'legend.fontsize': fontsize['normal'],
    'axes.labelsize': fontsize['normal'],
    'axes.titlesize': fontsize['large'],
    'xtick.labelsize': fontsize['small'],
    'ytick.labelsize': fontsize['small'],
    #'font.weight': 400,
    'axes.labelweight': 400,
    'axes.titleweight': 400,
    'axes.prop_cycle': plt.cycler('color', list(colors.values())),
    'lines.linewidth': .8,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .8,
    'axes.linewidth': .6,
    'xtick.major.width': .6,
    'ytick.major.width': .6,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 3.,
    'ytick.major.size': 3.,
    'xtick.minor.size': 2.,
    'ytick.minor.size': 2.,
    'xtick.major.pad': 3.5,
    'ytick.major.pad': 3.5,
    'axes.labelpad': 4.,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
})

def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.

    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)
    
# 8 bins
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]]) 

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}

obs_groups = {'yields' : ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton'],
              'mean_pT' : ['mean_pT_pion', 'mean_pT_kaon','mean_pT_proton', ],
              'fluct' : ['pT_fluct'],
              'flows' : ['v22', 'v32', 'v42']}

obs_group_labels = {'yields' : r'$dN_\mathrm{id}/dy_p$, $dN_\mathrm{ch}/d\eta$, $dE_T/d\eta$ [GeV]',
                    'mean_pT' : r'$ \langle p_T \rangle_\mathrm{id}$' + ' [GeV]',
                    'fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                    'flows' : r'$v^{(\mathrm{ch})}_k\{2\} $'}

colors = ['b', 'g', 'r', 'c', 'm', 'tan', 'gray']

obs_tex_labels = {'dNch_deta' : r'$dN_\mathrm{ch}/d\eta$' + ' x 2',
                  'dN_dy_pion' : r'$dN_{\pi}/dy_p$',
                  'dN_dy_kaon' : r'$dN_{K}/dy_p$',
                  'dN_dy_proton' : r'$dN_{p}/dy_p$',
                  'dET_deta' : r'$dE_{T}/d\eta$' + ' x 5',
                  
                  'mean_pT_proton' : r'$\langle p_T \rangle_p$',
                  'mean_pT_kaon' : r'$\langle p_T \rangle_K$',
                  'mean_pT_pion' : r'$\langle p_T \rangle_\pi$',
                 
                  'pT_fluct' : None,
                  'v22' : r'$v^{(\mathrm{ch})}_2\{2\}$',
                  'v32' : r'$v^{(\mathrm{ch})}_3\{2\}$',
                  'v42' : r'$v^{(\mathrm{ch})}_4\{2\}$'}
                  
index={}
st_index=0
for obs_group in  obs_groups.keys():
    for obs in obs_groups[obs_group]:
        #print(obs)
        n_centrality= len(obs_cent_list['Pb-Pb-2760'][obs])
        #print(n_centrality)
        index[obs]=[st_index,st_index+n_centrality]
        st_index = st_index+n_centrality
print(index)

# height_ratios = [1.8, 1.2, 1.5, 1.]
height_ratios = [2, 1.4, 1.4, 0.7]
column = 1
fig, axes = plt.subplots(nrows=4, ncols=column, figsize=(6, 8), squeeze=False, 
                         gridspec_kw={'height_ratios': height_ratios})

for row, obs_group in enumerate(obs_groups.keys()):
    for obs, color in zip(obs_groups[obs_group], colors):
        expt_label = 'ALICE'
        
        axes[row][0].tick_params(labelsize=9)
       # axes[row][1].tick_params(labelsize=9)
        
        scale = 1.0
        
        if obs_group == 'yields':
            axes[row][0].set_yscale('log')
          #  axes[row][1].set_yscale('log')
            
            axes[row][0].set_title("Experimental Data", fontsize = 11)
          #  axes[row][1].set_title("VAH + PTMA", fontsize = 11)
           
            if obs == 'dET_deta':
                scale = 5.
            if obs == 'dNch_deta':
                scale = 2.
        
        axes[row][0].set_ylabel(obs_group_labels[obs_group], fontsize = 11)
        
        xbins = np.array(obs_cent_list['Pb-Pb-2760'][obs])
        x = (xbins[:,0] + xbins[:,1]) / 2.

        #Y1 = Ymodel1['Pb-Pb-2760'][obs]['mean'][0][0]
        #Yerr1 = Ymodel1['Pb-Pb-2760'][obs]['err'][0][0]
        
        #Y2 = Ymodel2['Pb-Pb-2760'][obs]['mean'][0][0]
        #Yerr2 = Ymodel2['Pb-Pb-2760'][obs]['err'][0][0]
        
        label = obs_tex_labels[obs]
            
       # axes[row][0].plot(x, Y1*scale, color = color, label = label, lw = 1.5)
       # axes[row][0].fill_between(x, (Y1-Yerr1)*scale, (Y1+Yerr1)*scale, color=color, alpha=0.2)
        
       # axes[row][1].plot(x, Y2*scale, color = color, label = label, lw = 1.5)
       # axes[row][1].fill_between(x, (Y2-Yerr2)*scale, (Y2+Yerr2)*scale, color=color, alpha=0.2)
        
        exp_mean = y_exp[index[obs][0]:index[obs][1]]
        exp_err = np.sqrt(y_exp_variance[index[obs][0]:index[obs][1]])
        #exp_mean = Yexp['Pb-Pb-2760'][obs]['mean'][0]
        #exp_err = Yexp['Pb-Pb-2760'][obs]['err'][0]
    
        axes[row][0].errorbar(x, exp_mean*scale, exp_err, color=color, fmt='v', markersize='4', elinewidth=1, label=label)
        #axes[row][1].errorbar(x, exp_mean*scale, exp_err, color='black', fmt='v', markersize='4', elinewidth=1)
            

        
    leg = axes[row][0].legend(fontsize=9, borderpad=0, labelspacing=0, handlelength=1, handletextpad=0.2)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
        legobj.set_alpha(1.0)

    axes[row][0].set_xlim(0, 70)

    if obs_group == 'yields':
        axes[row][0].set_ylim(1, 1e5)
       # axes[row][1].set_ylim(1, 1e4)
    if obs_group == 'mean_pT':
        axes[row][0].set_ylim(0., 1.5)
       # axes[row][1].set_ylim(0., 1.5)
    if obs_group == 'fluct':
        axes[row][0].set_ylim(0.0, 0.06)
      #  axes[row][1].set_ylim(0.0, 0.04)
    if obs_group == 'flows':
        axes[row][0].set_ylim(0.0, 0.15)
       # axes[row][1].set_ylim(0.0, 0.12)
    if axes[row][0].is_last_row():
        axes[row][0].set_xlabel('Centrality %', fontsize = 11)
       # axes[row][1].set_xlabel('Centrality %', fontsize = 11)
        
plt.tight_layout()
# set_tight(fig, rect=[0, 0, 1, 0.95])
set_tight(fig, rect=[0, 0, 1, 1])
#save_fig("Experimental_data.png")

print("Done")

prior_df = pd.read_csv(filepath_or_buffer=data_path("PbPb2760_prior"), index_col=0)

design_max=prior_df.loc['max'].values
design_min=prior_df.loc['min'].values

# If false, uses pre-trained emulators.
# If true, retrain emulators.
train_emulators = False
import time
design=X
input_dim=len(design_max)
ptp = design_max - design_min
bound=zip(design_min,design_max)
if (os.path.exists(data_path(EMU))) and (train_emulators==False):
    print('Saved emulators exists and overide is prohibited')
    with open(data_path(EMU),"rb") as f:
        Emulators=pickle.load(f)
else:
    Emulators=[]
    for i in range(0,10):
        start_time = time.time()
        kernel=1*krnl.RBF(length_scale=ptp,length_scale_bounds=np.outer(ptp, (4e-1, 1e2)))+ krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-2, 1e2))
        GPR=gpr(kernel=kernel,n_restarts_optimizer=4,alpha=0.0000000001)
        GPR.fit(design,pc_tf_data[:,i].reshape(-1,1))
        print(f'GPR score is {GPR.score(design,pc_tf_data[:,i])} \n')
        #print(f'GPR log_marginal likelihood {GPR.log_marginal_likelihood()} \n')
        print("--- %s seconds ---" % (time.time() - start_time))
        Emulators.append(GPR)

if (train_emulators==True) or not(os.path.exists(data_path(EMU))):
    with open(data_path(EMU),"wb") as f:
        pickle.dump(Emulators,f)
        
        
def predict_observables(model_parameters):
    """Predicts the observables for any model parameter value using the trained emulators.
    
    Parameters
    ----------
    Theta_input : Model parameter values. Should be an 1D array of 17 model parametrs.
    
    Return
    ----------
    Mean value and full error covaraiance matrix of the prediction is returened. """
    
    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()
    
    if len(theta)!=17:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else: 
        theta=np.array(theta).reshape(1,17)
        for i in range(0,10):
            mn,std=Emulators[i].predict(theta,return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
    variance_matrix=np.diag(np.array(variance).flatten())
    A_p=inverse_tf_matrix
    inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance

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
        
# Covariance truncation error from keeping subset of PC is not included
def log_like(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats
    
    Return
    ----------
    unnormalized probability : float 
        
    """
    mn,var=predict_observables(model_parameters)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()
    
    exp_var=np.diag(y_exp_variance)
    
    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return mvn_loglike(delta_y,total_var)
    
# Covariance truncation error from keeping subset of PC is not included
def log_posterior(model_parameters):
    """
        Parameters
    ----------
    model_parameters : 17 dimensional list of floats
    
    Return
    ----------
    unnormalized probability : float 
    """
    
    mn,var=predict_observables(model_parameters)
    delta_y=mn-y_exp.reshape(1,-1)
    delta_y=delta_y.flatten()
    
    exp_var=np.diag(y_exp_variance)
    
    total_var=var + exp_var
    #only_diagonal=np.diag(total_var.diagonal())
    return log_prior(model_parameters) + mvn_loglike(delta_y,total_var)
    
# If false, uses pre-generated MCMC chains.
# If true, runs MCMC.
run_mcmc = False

# Here we actually perform the MCMC Sampling
filename = data_path(name+'ptemcee_closure') # to save in a different place

ntemps=20
Tmax = np.inf

nwalkers = 200 # Typically 10*ndim
ndim = 17
nburnin = 500 # The number of steps it takes for the walkers to thermalize
niterations= 1000 # The number of samples to draw once thermalized
nthin = 10 # Record every nthin-th iteration

nthreads = 8 # Easy parallelization! 

min_theta = [1.625] # Lower bound for initializing walkers
max_theta = [24.79] # Upper bound for initializing walkers

if run_mcmc:

    # 2. Instantiate the sampler object with the parameters, data, likelihood, and prior.
    
    #sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=nthreads, betas=betas)
    ptsampler_ex=ptemcee.Sampler(nwalkers, ndim, log_like, log_prior, ntemps, 
                      threads=nthreads, Tmax=Tmax)

    # 3. Initialize the walkers at random positions in our 99% prior range
    pos0 = design_min + (design_max - design_min) * np.random.rand(ntemps, nwalkers, ndim)

    # 4. Run the sampler's burn-in iterations
    print("Running burn-in phase")
    for p, lnprob, lnlike in ptsampler_ex.sample(pos0, iterations=nburnin,adapt=True):
        pass
    ptsampler_ex.reset() # Discard previous samples from the chain, but keep the position

    print("Running MCMC chains")
    # 5. Now we sample for nwalkers*niterations, recording every nthin-th sample
    for p, lnprob, lnlike in ptsampler_ex.sample(p, iterations=niterations, thin=nthin,adapt=True):
        pass 

    print('Done MCMC')

    mean_acc_frac = np.mean(ptsampler_ex.acceptance_fraction)
    print(f"Mean acceptance fraction: {mean_acc_frac:.3f}",
          f"(in total {nwalkers*niterations} steps)")
    
    # We only analyze the zero temperature MCMC samples
    
    #np.save(name+'ptemcee', )
   # closure_ex_chain = ptsampler_ex.chain[0, :, :, :].reshape((-1, ndim))
    samples = ptsampler_ex.chain[0, :, :, :].reshape((-1, ndim))
    samples_df = pd.DataFrame(samples, columns=model_param_dsgn)
    samples_df.to_csv(filename)
else:
    samples_df = pd.read_csv(filename, index_col=0)
    
et = time.time()
#print(f'Time it took to generate MCMC chain {et-st}')

data_df = pd.read_csv(filepath_or_buffer=data_path("new_LHC_posterior_samples.csv"))

posterior_original = data_df.iloc[:, :-1]

path = '/sysroot/home/rafaela/Projects/JETSCAPE_studies/Universality_pion_scale_spectra/posterior'

Grad_posterior = pd.read_csv(f"{path}/mcmc_chain_Grad.csv") 
CE_posterior = pd.read_csv(f"{path}/mcmc_chain_CE.csv") 
PTM_posterior = pd.read_csv(f"{path}/mcmc_chain_PTM.csv")
PTB_posterior = pd.read_csv(f"{path}/mcmc_chain_PTB.csv")

# Modify here
def plot_prior_posterior_chain(
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
    system='Pb-Pb-2760',
    design: pd.DataFrame = None,
    overlay_prior: bool = False,
    n_prior_samples: int = 200,
    random_state: int = 1 # generate the same state for reproducibility
):
    """
    Plot pt-integrated observables posterior band+median vs experiment,
    with optional overlay of prior (design-point) predictions.

    Parameters
    ----------
    posterior_df : pd.DataFrame, shape (n_post, n_params)
    predict_observables : func(params) -> (mm, vv)
    obs_groups : OrderedDict[group_name -> list of obs keys]
    obs_cent_list : dict[system][obs] -> ndarray (n_cent,2)
    index : dict[obs -> (start,end)] slices into mm.flatten()
    y_exp : 1D ndarray, length = total bins
    y_exp_variance : 1D ndarray, same length
    obs_tex_labels : dict[obs -> str]
    obs_group_labels : dict[group_name -> str]
    colors : list of colors per obs in each group
    height_ratios : list of floats, one per group
    system : str
    design : pd.DataFrame
        The “design” file containing prior points in same parameter order
    overlay_prior : bool
        If True, overlay raw design-point predictions
    n_prior_samples : int
        How many design points to plot (random subset)
    random_state : int
        Seed for selecting subset
    """
    # 1) Posterior predictions
    params_post = posterior_df.values
    mm_list = [predict_observables(p)[0].flatten() for p in params_post]
    mm_post = np.vstack(mm_list)  # shape (n_post, N_total)

    # 2) Optional: prior predictions
    # Modify here! 
    if overlay_prior:
        rng = np.random.default_rng(random_state)
        # if design has fewer rows, take all
        idxs = (rng.choice(len(design), size=min(n_prior_samples, len(design)), replace=False)
                if len(design) > n_prior_samples else np.arange(len(design)))
        param_prior = design.iloc[idxs].values
        mm_prior = np.vstack([predict_observables(p)[0].flatten() for p in param_prior])
    else:
        mm_prior = None

    # 3) Setup figure
    n_groups = len(obs_groups)
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(6, 8),
        squeeze=False,
        gridspec_kw={'height_ratios': height_ratios},
        dpi=600
    )

    # 4) Loop over groups
    for row, (group, obs_list) in enumerate(obs_groups.items()):
        ax = axes[row][0]
        ax.tick_params(labelsize=9)
        ax.set_ylabel(obs_group_labels[group], fontsize=11)

        for obs, color in zip(obs_list, colors):
            start, end = index[obs]
            # prior band
            prior_preds = mm_prior[:, start:end]
            # posterior band
            preds = mm_post[:, start:end]
            scale = 1.0
            if group == 'yields':
                if obs == 'dET_deta':  scale = 5.
                if obs == 'dNch_deta': scale = 2.

            lower = np.percentile(preds, 5, axis=0) * scale
            upper = np.percentile(preds, 95, axis=0) * scale
            median = np.median(preds, axis=0) * scale
            
            pr_lower = np.percentile(prior_preds, 5, axis=0) * scale
            pr_upper = np.percentile(prior_preds, 95, axis=0) * scale

            xbins = np.array(obs_cent_list[system][obs])
            x = (xbins[:,0] + xbins[:,1]) / 2.0

            # prior overlay
#            if mm_prior is not None:
#                for mm_i in mm_prior:
#                    y_prior = mm_i[start:end] * scale
#                    ax.plot(x, y_prior, color='gray', alpha=0.2, lw=0.5)
            
            # prior credible band
            ax.fill_between(x, pr_lower, pr_upper, color='gray', alpha=0.3) 
            # posterior credible band + median
#            ax.fill_between(x, lower, upper, color=color, alpha=0.4)
#            ax.plot(x, median, color=color, lw=1.5, label=obs_tex_labels[obs])

            # experimental data
            exp_mean = y_exp[start:end] * scale
            exp_err  = np.sqrt(y_exp_variance[start:end]) * scale
            ax.errorbar(
                x, exp_mean, exp_err,
                fmt='v', color='black', markersize=4, elinewidth=1
            )

        # styling
        if group == 'yields':
            ax.set_yscale('log')
#            ax.set_title(f"{system} {idf_label_short[idf]} Posterior & Prior", fontsize=11)
            ax.set_title(f"{system} {idf_label_short[idf]} Prior", fontsize=11)
        ax.set_xlim(0, 70)
        if group == 'yields':    ax.set_ylim(1,   1e5)
        if group == 'mean_pT':   ax.set_ylim(0.,  2)
        if group == 'fluct':     ax.set_ylim(0.,  0.06)
        if group == 'flows':     ax.set_ylim(0.,  0.15)

        leg = ax.legend(
            fontsize=9, borderpad=0, labelspacing=0,
            handlelength=1, handletextpad=0.2
        )
        for lh in leg.legendHandles:
            lh.set_linewidth(2.0)
            lh.set_alpha(1.0)

        if row == n_groups - 1:
            ax.set_xlabel('Centrality %', fontsize=11)

    plt.tight_layout()
    plt.savefig('prior_Grad.pdf', dpi=300, bbox_inches='tight')
    return fig

# 1) load design file
design = pd.read_csv(data_path("PbPb2760_design"))

# 2) overlay_prior=True to see prior
fig = plot_prior_posterior_chain(
    posterior_df = Grad_posterior,
#    posterior_df = CE_posterior,
#    posterior_df = posterior_original, 
    predict_observables = predict_observables,
    obs_groups = obs_groups,
    obs_cent_list = obs_cent_list,
    index = index,
    y_exp = y_exp,
    y_exp_variance = y_exp_variance,
    obs_tex_labels = obs_tex_labels,
    obs_group_labels = obs_group_labels,
    colors = colors,
    height_ratios = [2,1.4,1.4,0.7],
    system = 'Pb-Pb-2760',
    design = design,
    overlay_prior = True,      # <-- turn on prior overlay
    n_prior_samples = 485      # <--- how many design points to show
)
plt.show()
