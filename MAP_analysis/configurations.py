#!/usr/bin/env python3
import os, logging
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
#from bins_and_cuts import obs_cent_list, calibration_obs_cent_list, obs_range_list


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'
#fix the random seed for cross validation, that sets are deleted consistently
np.random.seed(1)

####################################
### USEFUL LABELS / DICTIONARIES ###
####################################

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

####################################
### SWITCHES AND OPTIONS !!!!!!! ###
####################################

#how many versions of the model are run, for instance
# 4 versions of delta-f with SMASH and a fifth model with UrQMD totals 5
number_of_models_per_run = 4

# the choice of viscous correction. 0 : 14 Moment, 1 : C.E. RTA, 2 : McNelis, 3 : Bernhard
idf = 0
print("Using idf = " + str(idf) + " : " + idf_label[idf])

#the Collision systems
systems = [
        ('Pb', 'Pb', 2760),
        #('Au', 'Au', 200),
        #('Pb', 'Pb', 5020),
        #('Xe', 'Xe', 5440)
        ]
system_strs = ['{:s}-{:s}-{:d}'.format(*s) for s in systems]
num_systems = len(system_strs)

#these are problematic points for Pb Pb 2760 and Au Au 200 with 500 design points
nan_sets_by_deltaf = {
                        0 : set([334, 341, 377, 429, 447, 483]),
                        1 : set([285, 334, 341, 447, 483, 495]),
                        2 : set([209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495]),
                        3 : set([60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495])
                    }
nan_design_pts_set = nan_sets_by_deltaf[idf]

#nan_design_pts_set = set([60, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 495])
unfinished_events_design_pts_set = set([289, 324, 326, 459, 462, 242, 406, 440, 123])
strange_features_design_pts_set = set([289, 324, 440, 459, 462])

delete_design_pts_set = nan_design_pts_set.union(
                            unfinished_events_design_pts_set.union(
                                        strange_features_design_pts_set
                                        )
                                    )

delete_design_pts_validation_set = [10, 68, 93] # idf 0


class systems_setting(dict):
    def __init__(self, A, B, sqrts):
        super().__setitem__("proj", A)
        super().__setitem__("targ", B)
        super().__setitem__("sqrts", sqrts)

#    def __setitem__(self, key, value):
#        if key == 'run_id':
#            super().__setitem__("main_obs_file",
#                str(workdir/'model_calculations/{:s}/Obs/main.dat'.format(value))
#                )
#            super().__setitem__("validation_obs_file",
#                str(workdir/'model_calculations/{:s}/Obs/validation.dat'.format(value))
#                )
#        else:
#            super().__setitem__(key, value)

SystemsInfo = {"{:s}-{:s}-{:d}".format(*s): systems_setting(*s) \
                for s in systems
               }

if 'Pb-Pb-2760' in system_strs:
    SystemsInfo["Pb-Pb-2760"]["run_id"] = "production_500pts_Pb_Pb_2760"
    SystemsInfo["Pb-Pb-2760"]["n_design"] = 500
    SystemsInfo["Pb-Pb-2760"]["n_validation"] = 100
    SystemsInfo["Pb-Pb-2760"]["design_remove_idx"]=list(delete_design_pts_set)
    SystemsInfo["Pb-Pb-2760"]["npc"]=10
    SystemsInfo["Pb-Pb-2760"]["MAP_obs_file"]="main.dat"

print("SystemsInfo = ")
print(SystemsInfo)

#
#bayes_dtype = [    (s,
#                  [(obs, [("mean",float_t,len(cent_list)),
#                          ("err",float_t,len(cent_list))]) \
#                    for obs, cent_list in obs_cent_list[s].items() ],
#                  number_of_models_per_run
#                 ) \
#                 for s in system_strs
#            ]
#
#bayes_calibration_dtype = [    (s,
#                  [(obs, [("mean",float_t,len(cent_list)),
#                          ("err",float_t,len(cent_list))]) \
#                    for obs, cent_list in calibration_obs_cent_list[s].items() ],
#                  number_of_models_per_run
#                 ) \
#                 for s in system_strs
#            ]




MAP_params = {}
MAP_params['Pb-Pb-2760'] = {}
MAP_params['Au-Au-200'] = {}


#values from ptemcee sampler with 500 walkers, 2k step adaptive burn in, 10k steps, 20 temperatures
#                                     N      p   sigma_k   w     d3   tau_R  alpha T_eta,kink a_low   a_high eta_kink zeta_max T_(zeta,peak) w_zeta lambda_zeta    b_pi   T_s
MAP_params['Pb-Pb-2760']['Grad'] = [14.2,  0.06,  1.05,  1.12,  3.00,  1.46,  0.031,  0.223,  -0.78,   0.37,    0.096,   0.13,      0.12,      0.072,    -0.12,   4.65 , 0.136]
MAP_params['Au-Au-200']['Grad'] =  [5.73,  0.06,  1.05,  1.12,  3.00,  1.46,  0.031,  0.223,  -0.78,   0.37,    0.096,   0.13,      0.12,      0.072,    -0.12,   4.65 , 0.136]

MAP_params['Pb-Pb-2760']['CE'] = [15.6,  0.06,  1.00,  1.19,  2.60,  1.04,  0.024,  0.268,  -0.73,   0.38,    0.042,   0.127,     0.12,      0.025,    0.095,   5.6,  0.146]
MAP_params['Au-Au-200']['CE'] =  [6.24,  0.06,  1.00,  1.19,  2.60,  1.04,  0.024,  0.268,  -0.73,   0.38,    0.042,   0.127,     0.12,      0.025,    0.095,   5.6,  0.146]

MAP_params['Pb-Pb-2760']['PTB'] = [13.2,  0.14,  0.98,  0.81,  3.11,  1.46,  0.017,  0.194,  -0.47,   1.62,    0.105,   0.165,     0.194,      0.026,    -0.072,  5.54,  0.147]
MAP_params['Au-Au-200']['PTB'] =  [5.31,  0.14,  0.98,  0.81,  3.11,  1.46,  0.017,  0.194,  -0.47,   1.62,    0.105,   0.165,     0.194,      0.026,    -0.072,  5.54,  0.147]
