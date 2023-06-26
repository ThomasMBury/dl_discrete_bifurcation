#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

Test DL classifier on Fox model with period-doubling bifrcation
** for different sets of parameter values **

@author: tbury
"""

# Parse command line arguments
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_sims', type=int, help='Total number of model simulations', 
#                     default=25)
# parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)

# args = parser.parse_args()
# model_sims = args.model_sims
# use_inter_classifier = True if args.use_inter_classifier=='true' else False

use_inter_classifier = False

import time
start_time = time.time()

import numpy as np
import pandas as pd

import funs_fox as funs

import ewstools

from tensorflow.keras.models import load_model


np.random.seed(0)
eval_pt = 0.8 #  percentage of way through pre-transition time series

# EWS parameters
rw = 0.5 # rolling window
span = 0.25 # Lowess span

# # Load in DL models
# if use_inter_classifier:
#     filepath_classifier = '../dl_train/output/'
# else:
#     filepath_classifier = '../../data/'

# m1 = load_model(filepath_classifier+'classifier_1.pkl')
# m2 = load_model(filepath_classifier+'classifier_2.pkl')
# print('TF models loaded')


# Simulate fox model with different trajectories


# Model parameters
A = 88
B = 122
C = 40
D = 28
tau = 180
alpha = 0.2
# alpha = 0.58


D0 = 200 
M0 = 1

sigma = 0.05

rate_forcing = 100/500
Tstart = 300
Tcrit = 200
Tfinal = 150
Tvals = np.arange(Tstart, Tfinal, -rate_forcing)

# Take transition time as time at bifurcation
transition = int((Tstart-Tcrit)/rate_forcing)
   
# For null 
# Tvals = [Tstart]*transition
    
df_traj = funs.simulate_model(A, B, C, D, tau, alpha, M0, D0, Tvals, sigma, sigma)

df_traj.set_index('time')['D'].plot()




# #--------------
# # forced trajectories
# #---------------
# print('Simulate forced trajectories and compute EWS')
# list_ktau_forced = []
# list_dl_forced = []
# list_ktau_null = []
# list_dl_null = []


# for rof in rof_vals:
#     for sigma in sigma_vals:
#         for id_val in id_vals:
         
#             s_forced, transition, s_null = funs.sim_rate_forcing(sigma, rof)
            
#             # Compute EWS for forced trajectory
#             ts = ewstools.TimeSeries(s_forced, transition=transition)
#             ts.detrend(method='Lowess', span=span)
#             ts.compute_var(rolling_window=rw)
#             ts.compute_auto(rolling_window=rw, lag=1)
#             ts.compute_ktau(tmin=0, tmax=transition*eval_pt)
#             dic_ktau = ts.ktau
#             dic_ktau['sigma'] = sigma
#             dic_ktau['rof'] = rof
#             dic_ktau['id'] = id_val
#             list_ktau_forced.append(dic_ktau)
            
#             # Get DL predictions for forced trajectory
#             ts.apply_classifier(m1, tmin=0, tmax=transition*eval_pt, name='m1', verbose=0)
#             ts.apply_classifier(m2, tmin=0, tmax=transition*eval_pt, name='m2', verbose=0)
#             df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
#             df_dl_preds['sigma'] = sigma
#             df_dl_preds['rof'] = rof
#             df_dl_preds['id'] = id_val
#             list_dl_forced.append(df_dl_preds)
        
#             # Compute EWS for null trajectory
#             ts = ewstools.TimeSeries(s_null)
#             ts.detrend(method='Lowess', span=span)
#             ts.compute_var(rolling_window=rw)
#             ts.compute_auto(rolling_window=rw, lag=1)
#             ts.compute_ktau(tmin=0, tmax=transition*eval_pt)
#             dic_ktau = ts.ktau
#             dic_ktau['sigma'] = sigma
#             dic_ktau['rof'] = rof
#             dic_ktau['id'] = id_val
#             list_ktau_null.append(ts.ktau)
            
#             # Get DL predictions for null
#             ts.apply_classifier(m1, tmin=0, tmax=transition*eval_pt, name='m1', verbose=0)
#             ts.apply_classifier(m2, tmin=0, tmax=transition*eval_pt, name='m2', verbose=0)
#             df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
#             df_dl_preds['sigma'] = sigma
#             df_dl_preds['rof'] = rof
#             df_dl_preds['id'] = id_val
#             list_dl_null.append(df_dl_preds)
            
#     print('Complete for rof={}'.format(rof))

# df_ktau_forced = pd.DataFrame(list_ktau_forced)
# df_dl_forced = pd.concat(list_dl_forced)

# df_ktau_null = pd.DataFrame(list_ktau_null)
# df_dl_null = pd.concat(list_dl_null)

# # Export data
# df_ktau_forced.to_csv('output/df_ktau_forced.csv', index=False)
# df_ktau_null.to_csv('output/df_ktau_null.csv', index=False)
# df_dl_forced.to_csv('output/df_dl_forced.csv', index=False)
# df_dl_null.to_csv('output/df_dl_null.csv', index=False)


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# print('Script took {:.2f} seconds'.format(time_taken))



