### Author: OptimusThi

"""
Load and remove problematic design points.
"""

import pandas as pd

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
    design = pd.read_csv(f'{design_path}/design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat', index_col=0)
    design = design.iloc[:325, :]
    design = design.drop(labels=list(delete_design_pts_set), errors='ignore')
    return design

idf = 0 
design_path = '/sysroot/home/rafaela/Projects/JETSCAPE_studies/Universality_pion_scale_spectra'
design = load_design(idf)
design.shape[0]
