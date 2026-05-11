#!/usr/bin/env python3
"""
Calculate transverse momentum spectra dN/dpT and universal scaled spectra U(xT)
for all design points from Qn data.
"""
import numpy as np
import os
import pickle
import multiprocessing as mp
from functools import partial

# --- Configuration ---
base_directory = '/data/js-sims-bayes/src/Qns/' # for PbPb 2.76 TeV
#base_directory = '/data/js-sims-bayes/src/QnsAu/' # for AuAu 0.2 TeV 
#base_directory = '/data/js-sims-bayes/src/QnsXe/' # for XeXe 5.44 TeV

particle_names = np.array(['pi', 'K', 'p', 'Sigma', 'Xi'])
masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
ptlist = (ptcuts[1:]+ptcuts[:-1])/2
dp_pt = np.diff(ptcuts)
Npt = len(ptlist)

delta_f_models = 4
max_n_events = 2500 # for PbPb 2.76 TeV and AuAu 0.2 TeV
#max_n_events = 1600 # for XeXe 5.44 TeV
Ndp = 500 # for PbPb 2.76 TeV and AuAu 0.2 TeV
#Ndp = 1000 # for XeXe 5.44 TeV

# Centrality bins
centbins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]])
ncentbins = len(centbins)

SPECIES_SPECTRA = {"pi": 0, "kaon": 1, "proton": 2, "Sigma": 3, "Xi": 4}

# --- Helper functions ---
def read_design_point(base_directory, dp):
    qn_file = os.path.join(base_directory, f'Qns_{dp}.npy')
    nsamples_file = os.path.join(base_directory, f'Nsamples_{dp}.npy')
    Qn = np.load(qn_file)
    Nsamples = np.load(nsamples_file)
    return Qn, Nsamples

def sort_by_Nch(Qn_norm):
    """Sort events by charged multiplicity (descending)"""
    if Qn_norm.ndim == 4:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :3, 0, :], axis=(1,2))
    elif Qn_norm.ndim == 3:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :3, :], axis=(1,2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {Qn_norm.ndim}D")
    eventlist = np.argsort(charged_mult)[::-1]
    return eventlist

def select_centrality(eventlist, centbins):
    nev = len(eventlist)
    groups = []
    for lo, hi in centbins:
        i0 = int(np.floor(lo/100.0 * nev))
        i1 = int(np.floor(hi/100.0 * nev))
        groups.append(eventlist[i0:i1])
    return groups

def universal_spectra_average(dNdpt_cent, ptlist):
    """
    Compute the universal scaling spectra U(x_T) for a given centrality.
    
    Parameters:
    -----------
    dNdpt_cent : array (n_events_in_cent, n_pt_bins)
        Differential spectra per event for this centrality bin
    ptlist : array
        pT bin centers
    
    Returns:
    --------
    mean_pT : float
        Scaling factor <pT> (mean transverse momentum)
    N : float
        Normalization factor (total multiplicity N)
    U : ndarray
        Universal spectrum U(x_T) = <pT>/N * dN/dpT
    mean_spectrum : ndarray
        Event-averaged dN/dpT
    """
    # Average over events for given centrality
    mean_spectrum = np.mean(dNdpt_cent, axis=0)
    
    # Normalization (total multiplicity N)
    N = np.trapz(mean_spectrum, ptlist)
    
    if N == 0:
        return 0.0, 0.0, np.zeros_like(ptlist), mean_spectrum
    
    # Scaling factor = <pT>
    mean_pT = np.trapz(ptlist * mean_spectrum, ptlist) / N
    
    # Universal function U(x_T) = <pT>/N * dN/dpT
    U = mean_pT * mean_spectrum / N
    
    return mean_pT, N, U, mean_spectrum

# --- Main computation function ---
def compute_spectra_design_point(idp, design_point, delta_f=1):
    """
    Compute transverse momentum spectra and universal spectra for a single design point.
    
    Returns dictionary with:
    - dN/dpT for each species and centrality
    - U(xT) for each species and centrality
    - <pT>, N for each species and centrality
    """
    print(f'Processing spectra for design point {design_point}, which is {idp}')
    
    try:
        Qn_all, Nsamples_all = read_design_point(base_directory, design_point)
        
        # Extract dimensions
        (Ndelta_f, Nevents, Nparticles, Nharmonics, Npt_data) = Qn_all.shape
        
        nsamples = Nsamples_all[delta_f]
        nsamples = nsamples.astype(float)
        nsamples[nsamples == 0] = np.inf
        
        # Build normalized Q0 array for all species
        # Q_0 is the multiplicity (harmonic index 0)
        Qn_norm_all = np.zeros((Nevents, Nparticles, Npt_data))
        
        for pid in range(Nparticles):
            # Extract Q_0 (multiplicity) for this species
            Q_0_raw = Qn_all[delta_f, :, pid, 0, :]  # [events, pT]
            # Normalize by samples and bin width to get dN/dpT
            Qn_norm_all[:, pid, :] = Q_0_raw / nsamples[:, np.newaxis] / dp_pt[np.newaxis, :]
        
        # Sort events by charged multiplicity
        eventlist = sort_by_Nch(Qn_norm_all)
        
        # Group by centrality
        centrality_events = select_centrality(eventlist, centbins)
        
        # Initialize results
        results = {
            'design_point': design_point,
            'delta_f': delta_f,
            'centrality_data': {}
        }
        
        # Loop over centrality bins
        for centrality, events_in_bin in enumerate(centrality_events):
            if len(events_in_bin) == 0:
                continue
            
            cent_results = {}
            
            # Loop over species
            for species_name, pid in SPECIES_SPECTRA.items():
                # Extract dN/dpT for this species and centrality
                dNdpt_cent = Qn_norm_all[events_in_bin, pid, :]  # [n_events_cent, Npt]
                
                # Compute universal spectra
                mean_pT, N, U, mean_spectrum = universal_spectra_average(dNdpt_cent, ptlist)
                
                # Compute x_T = pT / <pT>
                x_T = ptlist / mean_pT if mean_pT > 0 else np.zeros_like(ptlist)
                
                # Store results
                cent_results[species_name] = {
                    'dNdpt': mean_spectrum,
                    'U_xT': U,
                    'x_T': x_T,
                    'mean_pT': mean_pT,
                    'N': N,
                    'pt': ptlist.copy(),
                    'n_events': len(events_in_bin)
                }
            
            results['centrality_data'][centrality] = cent_results
        
        return results
        
    except Exception as e:
        print(f"Error processing design point {design_point}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_spectra_wrapper(args, delta_f=1):
    idp, design_point = args
    return compute_spectra_design_point(idp, design_point, delta_f)

def compute_all_design_points_spectra(design_points_list, delta_f=1, njobs=None):
    """
    Compute spectra for all design points in parallel.
    """
    if njobs is None:
        njobs = min(48, mp.cpu_count() * 2, len(design_points_list))
    
    print(f'Computing spectra for {len(design_points_list)} design points with {njobs} processes')
    
    compute_func = partial(compute_spectra_wrapper, delta_f=delta_f)
    
    if njobs == 1:
        results = []
        for idp, design_point in enumerate(design_points_list):
            result = compute_func((idp, design_point))
            results.append(result)
    else:
        with mp.Pool(processes=njobs) as pool:
            args = [(idp, design_point) for idp, design_point in enumerate(design_points_list)]
            results = pool.map(compute_func, args)
    
    results = [r for r in results if r is not None]
    
    print(f'Successfully computed spectra for {len(results)} design points')
    
    return results

# --- Usage ---
if __name__ == "__main__":
    
    # Test with first 10 design points
    #design_points = range(10)
    design_points = range(Ndp) # for full run
    df = 0 # Grad model
    
    # Compute spectra for all design points
    all_results = compute_all_design_points_spectra(
        design_points,
        delta_f=df,  # viscous correction model: 0=Grad, 1=CE, 2=PTM, 3=PTB
        njobs=48
    )
    
    delta_f_name = {0: 'Grad', 1: 'CE', 2: 'PTM', 3: 'PTB'}
    coll_system = ['PbPb_2.76TeV', 'AuAu_0.2TeV', 'XeXe_5.44TeV']

    if base_directory.endswith('Qns/'):
        coll_system = 'PbPb_2.76TeV'
    elif base_directory.endswith('QnsAu/'):
        coll_system = 'AuAu_0.2TeV'
    elif base_directory.endswith('QnsXe/'):
        coll_system = 'XeXe_5.44TeV'
    else:
        coll_system = 'Unknown'

    # Save results
    output_file = f'spectra_design_points_results_{coll_system}_{delta_f_name[df]}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    if len(all_results) > 0:
        sample = all_results[0]
        print(f"\nProcessed {len(all_results)} design points")
        if len(sample['centrality_data']) > 0:
            first_cent = list(sample['centrality_data'].values())[0]
            print(f"Species computed: {list(first_cent.keys())}")
            if len(first_cent) > 0:
                first_species = list(first_cent.values())[0]
                print(f"Observables per species: {list(first_species.keys())}")
