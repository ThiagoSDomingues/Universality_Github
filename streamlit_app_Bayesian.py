import streamlit as st
import pandas as pd 
import numpy as np
import time
import os
import subprocess
#import matplotlib
import altair as alt
import pickle
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

# -------------------------------
# Global Definitions for Labels & Colors
# -------------------------------

idf_label = {
            0 : 'Grad',
            1 : 'Chapman-Enskog R.T.A',
            2 : 'Pratt-Torrieri-McNelis',
            3 : 'Pratt-Torrieri-Bernhard'
            }
idf_label_short = {
            0 : 'Grad',
            1 : 'CE',
            2 : 'PTM',
            3 : 'PTB'
            }

short_names = {
                'norm' : r'Energy Normalization', #0
                'trento_p' : r'TRENTo Reduced Thickness', #1
                'sigma_k' : r'Multiplicity Fluctuation', #2
                'nucleon_width' : r'Nucleon width [fm]', #3
                'dmin3' : r'Min. Distance btw. nucleons cubed [fm^3]', #4
                'tau_R' : r'Free-streaming time scale [fm/c]', #5
                'alpha' : r'Free-streaming energy dep.', #6
                'eta_over_s_T_kink_in_GeV' : r'Temperature of shear kink [GeV]', #7
                'eta_over_s_low_T_slope_in_GeV' : r'Low-temp. shear slope [GeV^-1]', #8
                'eta_over_s_high_T_slope_in_GeV' : r'High-temp shear slope [GeV^-1]', #9
                'eta_over_s_at_kink' : r'Shear viscosity at kink', #10
                'zeta_over_s_max' : r'Bulk viscosity max.', #11
                'zeta_over_s_T_peak_in_GeV' : r'Temperature of max. bulk viscosity [GeV]', #12
                'zeta_over_s_width_in_GeV' : r'Width of bulk viscosity [GeV]', #13
                'zeta_over_s_lambda_asymm' : r'Skewness of bulk viscosity', #14
                'shear_relax_time_factor' : r'Shear relaxation time normalization', #15
                'Tswitch' : 'Particlization temperature [GeV]', #16
}

exp_centrality_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"]
exp_markers = ['o', 's', '^', 'D', 'v', '<', '>']

color_map = {0: 'blue', 1: 'red', 2: 'magenta', 3: 'green'}

MAP_params = {}
MAP_params['Pb-Pb-2760'] = {}
MAP_params['Au-Au-200'] = {}
MAP_params['Xe-Xe-5440'] = {}

#values from ptemcee sampler with 500 walkers, 2k step adaptive burn in, 10k steps, 20 temperatures
#                                     N      p   sigma_k   w     d3   tau_R  alpha T_eta,kink a_low   a_high eta_kink zeta_max T_(zeta,peak) w_zeta lambda_zeta    b_pi   T_s
MAP_params['Pb-Pb-2760']['Grad'] = [14.2,  0.06,  1.05,  1.12,  3.00,  1.46,  0.031,  0.223,  -0.78,   0.37,    0.096,   0.13,      0.12,      0.072,    -0.12,   4.65 , 0.136]
MAP_params['Au-Au-200']['Grad'] =  [5.73,  0.06,  1.05,  1.12,  3.00,  1.46,  0.031,  0.223,  -0.78,   0.37,    0.096,   0.13,      0.12,      0.072,    -0.12,   4.65 , 0.136]

MAP_params['Pb-Pb-2760']['CE'] = [15.6,  0.06,  1.00,  1.19,  2.60,  1.04,  0.024,  0.268,  -0.73,   0.38,    0.042,   0.127,     0.12,      0.025,    0.095,   5.6,  0.146]
MAP_params['Au-Au-200']['CE'] =  [6.24,  0.06,  1.00,  1.19,  2.60,  1.04,  0.024,  0.268,  -0.73,   0.38,    0.042,   0.127,     0.12,      0.025,    0.095,   5.6,  0.146]

MAP_params['Pb-Pb-2760']['PTB'] = [13.2,  0.14,  0.98,  0.81,  3.11,  1.46,  0.017,  0.194,  -0.47,   1.62,    0.105,   0.165,     0.194,      0.026,    -0.072,  5.54,  0.147]
MAP_params['Au-Au-200']['PTB'] =  [5.31,  0.14,  0.98,  0.81,  3.11,  1.46,  0.017,  0.194,  -0.47,   1.62,    0.105,   0.165,     0.194,      0.026,    -0.072,  5.54,  0.147]

system = 'Pb-Pb-2760'

# -------------------------------
# Added System Selection & Viscosity Plots
# -------------------------------
def system_selection():
    systems = list(MAP_params.keys())
    return st.selectbox('Collision System', systems)

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
    design =  design.drop(labels=list(delete_design_pts_set), errors='ignore')
    labels = design.keys()    
    data_path = 'design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat'    
    design_range = pd.read_csv(data_path, index_col=0)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, labels, design_max, design_min
    
# -------------------------------
# Optimized Calculations & Caching
# -------------------------------    
@st.cache_data(show_spinner=False)
def load_design_cached(idf):
    return load_design(idf)    
       
def prepare_simulation_data(Y):
    Y_reduced = Y[:, :7, :]
    n_design, n_cent, n_xt = Y_reduced.shape
    Y_flat = Y_reduced.reshape(n_design, n_cent * n_xt)
    return Y_flat

@st.cache_data(show_spinner=False)
def scaler_cached(idf):
    Y_sim = load_moment_data('Bayesian_data', idf)
    Y_flat = prepare_simulation_data(Y_sim)
    return StandardScaler().fit(Y_flat)
    
def inverse_tf_matrix_cached(idf):
    Y_sim = load_moment_data('Bayesian_data', idf)
    Y_flat = prepare_simulation_data(Y_sim)
    SS = StandardScaler().fit(Y_flat)
    u, s, vh = np.linalg.svd(SS.transform(Y_flat), full_matrices=True)
    return np.diag(s[0:6]) @ vh[0:6,:] * SS.scale_.reshape(1,287)/np.sqrt(u.shape[0]-1)        
      
@st.cache_data()
def load_emu_cached(system, idf):
    #load the emulator
    n_pc_train=6
    emulator_output_file = f"emulators/emulators_{idf_label_short[idf]}_{n_pc_train}_pcs.pkl"
    with open(emulator_output_file, "rb") as f:
        emu = pickle.load(f)
    return emu

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

@st.cache_data(persist=True) 
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

@st.cache_data(show_spinner=False)
def load_experimental_cached():
    return load_experimental()
    
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
# Visualization Optimizations
# -------------------------------
def plot_spectra_comparison(y_pred, xt_exp, u_xt_exp, err_u_xt):
    n_cent = len(exp_centrality_labels)
    predictions = y_pred.reshape(n_cent, 41)
    
    charts = []
    for cent in range(n_cent):
        source = pd.DataFrame({
            'xT': xt_exp[cent],
            'Prediction': predictions[cent],
            'Experiment': u_xt_exp[cent],
            'Error': err_u_xt[cent]
        })
        
        base = alt.Chart(source).encode(
            x=alt.X('xT:Q', scale=alt.Scale(type='log'))
        )
        
        line = base.mark_line().encode(y='Prediction:Q')
        points = base.mark_point(size=50).encode(y='Experiment:Q')
        error_bars = base.mark_errorbar().encode(
            y=alt.Y('Experiment:Q', title='U(xT)'),
            yError='Error:Q'
        )
        
        charts.append((line + points + error_bars).properties(
            title=f'Centrality {exp_centrality_labels[cent]}',
            width=300,
            height=200
        ))
    
    st.altair_chart(alt.vconcat(
        alt.hconcat(*charts[:3]),
        alt.hconcat(*charts[3:6]),
        alt.hconcat(charts[6])
    ))    

# -------------------------------
# Viscosity Plots
# -------------------------------
def make_viscosity_plots(params):
    T = np.linspace(0.1, 0.35, 100)
    eta_s = eta_over_s(T, *params[7:11])
    zeta_s = zeta_over_s(T, *params[11:15])
    
    df = pd.DataFrame({'T (GeV)': T, 'η/s': eta_s, 'ζ/s': zeta_s}).melt('T (GeV)')
    
    chart = alt.Chart(df).mark_line().encode(
        x='T (GeV):Q',
        y='value:Q',
        color='variable:N'
    ).properties(width=600, height=300)
    
    st.altair_chart(chart)

# Depracted
#def make_plot_eta_zeta(params):
    
    # Defining temperature range
#    T_low = 0.1 
#    T_high = 0.35
#    T = np.linspace(T_low, T_high, 100) # linearly spaced values of temperature
    
    # Calling the functions
#    eta_s = eta_over_s(T, *params[7:11]) # params 7,8,9,10
#    zeta_s = zeta_over_s(T, *params[11:15]) # params 11, 12, 13, 14
    
    # Create a dataframe for eta and zeta
#    df_eta_zeta = pd.DataFrame({'T': T, 'eta':eta_s, 'zeta':zeta_s})  
    
    # Creating chart plots for eta and zeta
#    chart_eta = 
#    chart_zeta = 
    
#    charts = alt.hconcat(chart_zeta, chart_eta)
#    st.write(charts)

# -------------------------------
# Visualization Optimizations
# -------------------------------
def plot_spectra_comparison(y_pred, xt_exp, u_xt_exp, err_u_xt):
    n_cent = len(exp_centrality_labels)
    predictions = y_pred.reshape(n_cent, 41)
    
    charts = []
    for cent in range(n_cent):
        source = pd.DataFrame({
            'xT': xt_exp[cent],
            'Prediction': predictions[cent],
            'Experiment': u_xt_exp[cent],
            'Error': err_u_xt[cent]
        })
        
        base = alt.Chart(source).encode(
            x=alt.X('xT:Q', scale=alt.Scale(type='log'))
        )
        
        line = base.mark_line().encode(y='Prediction:Q')
        points = base.mark_point(size=50).encode(y='Experiment:Q')
        error_bars = base.mark_errorbar().encode(
            y=alt.Y('Experiment:Q', title='U(xT)'),
            yError='Error:Q'
        )
        
        charts.append((line + points + error_bars).properties(
            title=f'Centrality {exp_centrality_labels[cent]}',
            width=300,
            height=200
        ))
    
    st.altair_chart(alt.vconcat(
        alt.hconcat(*charts[:3]),
        alt.hconcat(*charts[3:6]),
        alt.hconcat(charts[6])
    ))
    
# -------------------------------
# Optimized Main Function
# -------------------------------    
def main():
    st.set_page_config(layout="wide")   
    st.title('Scaled Particle Spectra Observable Emulator for Heavy-Ion Collisions')
    
    # System selection
    system = system_selection()
    
    # Model selection
    idf_names = ['Grad', 'Chapman-Enskog R.T.A', 'Pratt-Torrieri-McNelis', 'Pratt-Torrieri-Bernhard']
    idf_name = st.selectbox('Particlization model', idf_names)
    
    # Reset button
#    st.markdown('<a href="javascript:window.location.href=window.location.href">Reset</a>', unsafe_allow_html=True)
    
    inverted_idf_label = dict([[v,k] for k,v in idf_label.items()])
    idf = inverted_idf_label[idf_name]
    
    # Cached data loading
    design, labels, design_max, design_min = load_design_cached(idf)
    xt_exp, u_xt_exp, err_u_xt = load_experimental_cached()
    emu = load_emu_cached(system, idf)
#    design, labels, design_max, design_min = load_design(idf)
    
    # get emu predictions
#    emu = load_emu(system, idf)
    
    # load the experimental data
#    xt_exp, u_xt_exp, err_u_xt = load_experimental()
    
    # Parameter sliders with session state
    # initialize parameters
    if 'params' not in st.session_state:
        st.session_state.params = MAP_params[system][idf_label_short[idf]]
    
    with st.sidebar:
        st.header("Model Parameters")
        for i, (s_name, label) in enumerate(short_names.items()):
            st.session_state.params[i] = st.slider(
                label, 
                min_value=design_min[i],
                max_value=design_max[i],
                value=st.session_state.params[i],
                step=(design_max[i]-design_min[i])/100,
                key=f"param_{i}"
            )
            

#    params_0 = MAP_params[system][ idf_label_short[idf] ]
#    params = []
    
    # updated params
#    for i_s, s_name in enumerate(short_names.keys()):
#        min = design_min[i_s]
#        max = design_max[i_s]
#        step = (max - min)/100.
#        p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=params_0[i_s], step=step)
 #       params.append(p)
    # Main display tabs
    tab1, tab2 = st.tabs(["Spectra Analysis", "Transport Coefficients"])
    
    with tab1:
        # Optimized prediction with progress
        with st.spinner('Calculating predictions...'):
            y_pred, cov_pred = predict_observables(
                st.session_state.params, 
                emu, 
                inverse_tf_matrix_cached(idf), 
                scaler_cached(idf)
            )
        
        # Plotting in columns
#        cols = st.columns(2)
#        with cols[0]:
            plot_spectra_comparison(y_pred, xt_exp, u_xt_exp, err_u_xt)
#        with cols[1]:
            #make_viscosity_plots(st.session_state.params)
    
    with tab2:
        st.header("Shear and Bulk Viscosities")
        make_viscosity_plots(st.session_state.params)
    
# This can be change. I don't know if I'm doing it correctly.
#    Y_sim = load_moment_data('Bayesian_data', idf)
#    Y_flat = prepare_simulation_data(Y_sim)    
    
    # Scaling the data to be zero mean and unit variance for each observables
#    SS  =  StandardScaler(copy=True)
    
    # Singular Value Decomposition
#    u, s, vh = np.linalg.svd(SS.fit_transform(Y_flat), full_matrices=True) # scaling Y_flat, covariance matrix
    
    # whiten and project data to principal component axis (only keeping first 6 PCs)
#    pc_tf_data=u[:,0:6] * math.sqrt(u.shape[0]-1)
    
    # Scale Transformation from PC space to original data space
#    inverse_tf_matrix= np.diag(s[0:6]) @ vh[0:6,:] * SS.scale_.reshape(1,287)/ math.sqrt(u.shape[0]-1)
    
#    scaler = SS 
        
    # get emu prediction        
#    y_pred, cov_pred = predict_observables(params, emu, inverse_tf_matrix, scaler)
#    Yemu_mean, Yemu_cov, time_emu = emu_predict(emu, params)
    
    # Plotting predictions + experimental data
#    n_cent = len(exp_centrality_labels)
    
#    print(y_pred.shape)
#    n_samples = y_pred.shape[0]
#    n_cent = len(exp_centrality_labels)  # should be 7
#    n_xt = 41
#    predictions_reshaped = y_pred.reshape(n_cent, n_xt)
#    y = predictions_reshaped
    
#    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16,8), dpi=1200)
#    axs = axs.flatten()
#    for cent in range(n_cent):
#        ax = axs[cent]
        
        # Emulators predictions
#        ax.fill_between(xt_exp[cent, :], np.percentile(y, 5, axis=0), np.percentile(y, 95, axis=0), color=color_map[idf], alpha=0.4, label="Predictions 90% C.I.")
#        ax.plot(xt_exp[cent, :], predictions_reshaped[cent, :])
#        ax.plot(xt_exp[cent, :], y_pred[cent, :])
#        ax.errorbar(xt_exp[cent, :], u_xt_exp[cent, :], yerr=err_u_xt[cent, :],
#                    fmt=exp_markers[cent], color='black', capsize=3, label="ALICE PbPb 2.76 TeV")        
#        ax.set_title(f"Centrality {exp_centrality_labels[cent]}")
#        ax.set_xlabel(r"$x_T$")
#        ax.set_ylabel(r"$U(x_T)$")
#        ax.set_xscale("log")
#        ax.legend(fontsize=8)
#    for ax in axs[n_cent:]:
#        ax.axis('off')
#    plt.tight_layout()
#    st.pyplot(fig)
    
    # redraw plots    
#    make_plot_eta_zeta(params)
            
#    st.header('Emulators predictions')
#    st.markdown('Universal Scaled of Particle Spectra')
#    st.markdown('To update the widget with latest changes, click the button below, and then refresh your webpage')
#    if st.button('(Update widget)'):
#        subprocess.run("git pull origin master", shell=True)
    
if __name__ == "__main__":
    main()     
