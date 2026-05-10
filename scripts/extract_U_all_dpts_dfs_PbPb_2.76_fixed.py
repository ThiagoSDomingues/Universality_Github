#!/usr/bin/env python3
"""
compute_UxT_design_points.py
============================
Compute U(xT) = <pT>/N * dN/dpT for all design points from Qn arrays,
using the same pT bins and centrality classes as the experimental data.

Supported systems:
    PbPb_2760   — Pb+Pb  √sNN = 2.76 TeV  (500 design points)
    XeXe_5440   — Xe+Xe  √sNN = 5.44 TeV  (1000 design points)
    AuAu_200    — Au+Au  √sNN = 0.200 TeV  (500 design points)

Species computed:
    pi, kaon, proton, Sigma, Xi   (plus charged hadrons as sum)

Viscous correction models (delta_f):
    0 = Grad,  1 = CE,  2 = PTM,  3 = PTB

Output (one file per viscous correction model):
    UxT_results_{system}_{delta_f_name}.pkl

Each file is a list (one entry per design point) of dicts:
    {
        'design_point': int,
        'delta_f':      int,
        'centrality_data': {
            cent_idx: {
                species: {
                    'pT':      ndarray  — pT bin centers  [GeV/c]
                    'dNdpT':   ndarray  — dN/dpT (event-averaged)
                    'xT':      ndarray  — pT / <pT>
                    'U_xT':    ndarray  — U(xT) = <pT>/N * dN/dpT
                    'mean_pT': float    — <pT> [GeV/c]
                    'N':       float    — total yield N
                    'n_events':int      — events in centrality bin
                }
            }
        }
    }
"""

import numpy as np
import os
import pickle
import multiprocessing as mp
from functools import partial

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_CONFIG = {
    "PbPb_2760": {
        "base_directory": "/data/js-sims-bayes/src/Qns/",
        "Ndp":            500,
        "max_n_events":   2500,
        "label":          "PbPb_2.76TeV",
    },
    "XeXe_5440": {
        "base_directory": "/data/js-sims-bayes/src/QnsXe/",
        "Ndp":            1000,
        "max_n_events":   1600,
        "label":          "XeXe_5.44TeV",
    },
    "AuAu_200": {
        "base_directory": "/data/js-sims-bayes/src/QnsAu/",
        "Ndp":            500,
        "max_n_events":   2500,
        "label":          "AuAu_0.2TeV",
    },
}

DELTA_F_NAMES = {0: "Grad", 1: "CE", 2: "PTM", 3: "PTB"}

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE SPECIES
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping: species label → index in Qn array
# The Qn arrays contain: pi(0), K(1), p(2), Sigma(3), Xi(4)
# Charged hadrons = 2*(pi + K + p)
SPECIES = {
    "pi":      0,
    "kaon":    1,
    "proton":  2,
    "Sigma":   3,
    "Xi":      4,
    "charged": None,   # computed as 2*(pi+K+p)
}

MASSES = {
    "pi":      0.13957,
    "kaon":    0.49368,
    "proton":  0.93827,
    "Sigma":   1.18937,
    "Xi":      1.32132,
    "charged": None,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CENTRALITY BINS
# ═══════════════════════════════════════════════════════════════════════════════

CENTBINS = np.array([
    [0, 5], [5, 10], [10, 20], [20, 30], [30, 40],
    [40, 50], [50, 60], [60, 70], [70, 80], [80, 90]
])
CENT_LABELS = [f"{int(lo)}-{int(hi)}%" for lo, hi in CENTBINS]

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION pT GRID  (common to all systems, defined by the Qn files)
# ═══════════════════════════════════════════════════════════════════════════════

PTCUTS_SIM = np.array([
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 10.0
])
PTLIST_SIM = 0.5 * (PTCUTS_SIM[1:] + PTCUTS_SIM[:-1])
DPPT_SIM   = np.diff(PTCUTS_SIM)
NPT_SIM    = len(PTLIST_SIM)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL pT BIN EDGES  (match the ALICE HEPData tables)
# ═══════════════════════════════════════════════════════════════════════════════

# Pions  [0.10 – 2.96 GeV]  — from ALICE HEPData ins1222333
PTEDGES_EXP = {
    "pi": np.array([
        0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80,
        1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80,
        2.90, 3.00
    ]),
    # Kaons  [0.20 – 3.00 GeV]  — from ALICE HEPData ins1222333
    "kaon": np.array([
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30,
        1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30,
        2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00
    ]),
    # Protons  [0.30 – 4.60 GeV]  — from ALICE HEPData ins1222333
    "proton": np.array([
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50,
        1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50,
        2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00,
        4.20, 4.40, 4.60
    ]),
    # Heavier species — use full simulation range (user can override)
    "Sigma": np.array([
        0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00,
        2.20, 2.40, 2.60, 2.80, 3.00, 3.40, 3.80, 4.20, 4.60
    ]),
    "Xi": np.array([
        0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.40, 2.60,
        2.80, 3.00, 3.40, 3.80, 4.20
    ]),
}
# Charged uses pion bins as reference
PTEDGES_EXP["charged"] = PTEDGES_EXP["pi"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def map_to_exp_bins(ptedges_exp, ptlist_sim, dNdpT_fine, dpT_sim):
    """
    Re-bin a fine dN/dpT spectrum into coarser experimental pT bins.

    The input ``dNdpT_fine`` contains dN/dpT *densities* on the simulation
    grid.  To convert to counts we must weight each sim bin by its width
    (dpT_sim) before summing, then divide by the experimental bin width to
    recover a density on the experimental grid.

    Without the dpT_sim weighting the result is dimensionally wrong:
    summing densities and dividing by a *different* bin width gives values
    that are off by a factor ~dpT_exp / dpT_sim (up to 20× for the narrow
    pion bins).

    Parameters
    ----------
    ptedges_exp  : (N_exp+1,)         experimental bin edges  [GeV/c]
    ptlist_sim   : (Npt_sim,)          simulation bin centres  [GeV/c]
    dNdpT_fine   : (..., Npt_sim)      dN/dpT on the simulation grid
                                        (last axis = pT)
    dpT_sim      : (Npt_sim,)          simulation bin widths   [GeV/c]

    Returns
    -------
    dNdpT_exp    : (..., N_exp)        dN/dpT on the experimental grid
    ptcenter_exp : (N_exp,)            experimental bin centres  [GeV/c]
    dpT_exp      : (N_exp,)            experimental bin widths   [GeV/c]
    """
    N_exp     = len(ptedges_exp) - 1
    shape_out = dNdpT_fine.shape[:-1] + (N_exp,)
    out       = np.zeros(shape_out, dtype=np.float64)

    for i in range(N_exp):
        lo, hi = ptedges_exp[i], ptedges_exp[i + 1]
        mask   = (ptlist_sim >= lo) & (ptlist_sim < hi)
        if not np.any(mask):
            continue
        # counts in this exp bin = ∫ dN/dpT · dpT  (rectangle rule on sim grid)
        # then dN/dpT_exp = counts / dpT_exp
        out[..., i] = np.sum(dNdpT_fine[..., mask] * dpT_sim[mask], axis=-1)

    ptcenter_exp = 0.5 * (ptedges_exp[:-1] + ptedges_exp[1:])
    dpT_exp      = np.diff(ptedges_exp)

    # Convert counts → density
    dNdpT_exp = out / dpT_exp
    return dNdpT_exp, ptcenter_exp, dpT_exp


def compute_U_xT(dNdpT_events, pT_centers, dpT):
    """
    Compute event-averaged dN/dpT, <pT>, N, U(xT), and xT.

    Parameters
    ----------
    dNdpT_events : (n_events, N_pt)  per-event dN/dpT
    pT_centers   : (N_pt,)
    dpT          : (N_pt,)

    Returns
    -------
    dict with keys: dNdpT, mean_pT, N, U_xT, xT
    """
    mean_dNdpT = np.mean(dNdpT_events, axis=0)

    # Total yield N = ∫ dN/dpT dpT
    N = np.sum(mean_dNdpT * dpT)
    if N <= 0:
        z = np.zeros_like(pT_centers)
        return dict(dNdpT=mean_dNdpT, mean_pT=0.0, N=0.0, U_xT=z, xT=z)

    # Mean pT = ∫ pT * dN/dpT dpT / N
    mean_pT = np.sum(pT_centers * mean_dNdpT * dpT) / N

    # Universal scaling  U(xT) = <pT>/N * dN/dpT
    U_xT = (mean_pT / N) * mean_dNdpT
    xT   = pT_centers / mean_pT if mean_pT > 0 else np.zeros_like(pT_centers)

    return dict(dNdpT=mean_dNdpT, mean_pT=mean_pT, N=N, U_xT=U_xT, xT=xT)


def sort_by_Nch(q0_all_fine):
    """
    Sort events by charged multiplicity (pi+K+p) descending.

    q0_all_fine : (n_events, 5, Npt_sim)  — species index 0=pi,1=K,2=p
    """
    # factor 2 for +/-; first 3 species are pi, K, p
    Nch = 2.0 * np.sum(q0_all_fine[:, :3, :], axis=(1, 2))
    return np.argsort(Nch)[::-1]


def select_centrality_groups(eventlist, centbins, n_events):
    """Split sorted eventlist into centrality bins."""
    groups = []
    for lo, hi in centbins:
        i0 = int(np.floor(lo / 100.0 * n_events))
        i1 = int(np.floor(hi / 100.0 * n_events))
        groups.append(eventlist[i0:i1])
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# PER–DESIGN-POINT COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_one_design_point(idp, design_point, base_directory, delta_f,
                             species_to_compute):
    """
    Compute U(xT) for one design point.

    Returns a dict ready to append to the results list, or None on failure.
    """
    print(f"  [DP {design_point:04d}] delta_f={delta_f}", flush=True)

    try:
        qn_path = os.path.join(base_directory, f"Qns_{design_point}.npy")
        ns_path = os.path.join(base_directory, f"Nsamples_{design_point}.npy")
        Qn_all    = np.load(qn_path)
        Nsamples  = np.load(ns_path)

        # Shape: (n_delta_f, n_events, n_species, n_harmonics, Npt)
        ndeltaf, Nevents, Nspecies, Nharm, Npt_data = Qn_all.shape

        nsamples = Nsamples[delta_f].astype(float)
        nsamples[nsamples == 0] = np.inf   # avoid div-by-zero

        # Harmonic n=0  →  multiplicity counts,  divide by samples & bin width
        # Result shape: (n_events, Nspecies, Npt_sim)
        Q0_raw  = Qn_all[delta_f, :, :, 0, :]           # (Nev, Nsp, Npt)
        q0_fine = Q0_raw / nsamples[:, None, None] / DPPT_SIM[None, None, :]

        # Sort events by charged multiplicity
        order  = sort_by_Nch(q0_fine)
        groups = select_centrality_groups(order, CENTBINS, Nevents)

        result = {
            "design_point":    design_point,
            "delta_f":         delta_f,
            "centrality_data": {},
        }

        for cent_idx, events_in_bin in enumerate(groups):
            if len(events_in_bin) == 0:
                continue

            cent_data = {}

            # q0 for events in this centrality: (n_cent, Nspecies, Npt_sim)
            q0_cent = q0_fine[events_in_bin, :, :]

            for species, pid in species_to_compute.items():

                # ── Build per-event dN/dpT in experimental bins ──────────────
                if species == "charged":
                    # 2 × (pi + K + p)  in fine bins
                    q0_sp_fine = 2.0 * np.sum(q0_cent[:, :3, :], axis=1)
                else:
                    # 2 × particle+ = particle+ + particle-   (symmetry)
                    q0_sp_fine = 2.0 * q0_cent[:, pid, :]

                # Map to experimental pT bins
                ptedges_exp = PTEDGES_EXP[species]
                counts_exp, pT_exp, dpT_exp = map_to_exp_bins(
                    ptedges_exp, PTLIST_SIM, q0_sp_fine
                )
                # counts_exp: (n_cent_events, N_exp_bins)
                # Each bin already has units 1/dpT_sim summed → divide by dpT_exp
                dNdpT_events = counts_exp / dpT_exp[None, :]

                # ── U(xT) ─────────────────────────────────────────────────────
                obs = compute_U_xT(dNdpT_events, pT_exp, dpT_exp)
                obs["pT"]      = pT_exp
                obs["n_events"] = len(events_in_bin)

                cent_data[species] = obs

            result["centrality_data"][cent_idx] = cent_data

        return result

    except Exception as exc:
        import traceback
        print(f"  [DP {design_point:04d}] ERROR: {exc}")
        traceback.print_exc()
        return None


def _worker(args):
    """Unpacking wrapper for multiprocessing.Pool.map."""
    idp, dp, base_directory, delta_f, species = args
    return compute_one_design_point(idp, dp, base_directory, delta_f, species)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_design_points(system, delta_f, species_to_compute=None,
                              n_dp=None, n_jobs=48, output_dir="."):
    """
    Compute U(xT) for all design points of one system and one delta_f model.

    Parameters
    ----------
    system            : str   — key in SYSTEM_CONFIG
    delta_f           : int   — 0=Grad, 1=CE, 2=PTM, 3=PTB
    species_to_compute: dict  — subset of SPECIES; default = all
    n_dp              : int   — number of design points (None → from config)
    n_jobs            : int   — parallel processes
    output_dir        : str   — where to write the .pkl file

    Returns
    -------
    Path to the saved .pkl file
    """
    cfg = SYSTEM_CONFIG[system]
    base_dir = cfg["base_directory"]
    Ndp      = n_dp or cfg["Ndp"]
    label    = cfg["label"]
    dfname   = DELTA_F_NAMES[delta_f]

    if species_to_compute is None:
        species_to_compute = SPECIES

    print("=" * 70)
    print(f"System      : {system}  ({label})")
    print(f"delta_f     : {delta_f} ({dfname})")
    print(f"Design pts  : {Ndp}")
    print(f"Species     : {list(species_to_compute.keys())}")
    print(f"Parallel    : {n_jobs} processes")
    print("=" * 70)

    dp_list = list(range(Ndp))
    args_list = [
        (idp, dp, base_dir, delta_f, species_to_compute)
        for idp, dp in enumerate(dp_list)
    ]

    if n_jobs == 1:
        results = [_worker(a) for a in args_list]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_worker, args_list)

    results = [r for r in results if r is not None]
    print(f"\n✓ Finished: {len(results)}/{Ndp} design points OK")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"UxT_{label}_{dfname}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh, protocol=4)
    print(f"✓ Saved → {out_path}\n")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / NOTEBOOK ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute U(xT) for JETSCAPE design points"
    )
    parser.add_argument(
        "--system", default="PbPb_2760",
        choices=list(SYSTEM_CONFIG.keys()),
        help="Collision system",
    )
    parser.add_argument(
        "--delta-f", type=int, nargs="+", default=[0, 1, 2, 3],
        metavar="N",
        help="Viscous correction model(s): 0=Grad 1=CE 2=PTM 3=PTB (default: all 4)",
    )
    parser.add_argument(
        "--species", nargs="+",
        default=list(SPECIES.keys()),
        choices=list(SPECIES.keys()),
        help="Species to compute (default: all)",
    )
    parser.add_argument(
        "--ndp", type=int, default=None,
        help="Number of design points (default: system default)",
    )
    parser.add_argument(
        "--jobs", type=int, default=48,
        help="Parallel processes (default: 48)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output .pkl files",
    )
    return parser


def run(system="PbPb_2760", delta_f=None, species=None,
        ndp=None, jobs=48, output_dir="."):
    """
    Notebook-friendly entry point — call this directly instead of using CLI args.

    Example
    -------
    from compute_UxT_design_points import run
    run(system="PbPb_2760", delta_f=[0, 1, 2, 3], jobs=48, output_dir="results/")
    """
    if delta_f is None:
        delta_f = [0, 1, 2, 3]
    if species is None:
        species = list(SPECIES.keys())

    selected_species = {sp: SPECIES[sp] for sp in species}

    for df in delta_f:
        compute_all_design_points(
            system=system,
            delta_f=df,
            species_to_compute=selected_species,
            n_dp=ndp,
            n_jobs=jobs,
            output_dir=output_dir,
        )
    print("All done.")


if __name__ == "__main__":
    import argparse

    parser = _build_parser()
    # parse_known_args ignores Jupyter's --f=... kernel flag
    args, _ = parser.parse_known_args()

    run(
        system=args.system,
        delta_f=args.delta_f,
        species=args.species,
        ndp=args.ndp,
        jobs=args.jobs,
        output_dir=args.output_dir,
    )
