#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:44:23 2023

Compute EWS for Fox model simulations with different model parameters.
Compute ROC curves

@author: tbury
"""



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

# Load in DL models
if use_inter_classifier:
    filepath_classifier = '../dl_train/output/'
else:
    filepath_classifier = '../../data/'

m1 = load_model(filepath_classifier+'classifier_1.pkl')
m2 = load_model(filepath_classifier+'classifier_2.pkl')
print('TF models loaded')


# Load in bifurcation values
df_bif_vals_alpha = pd.read_csv('output/df_bifvalues_alpha.csv')
df_bif_vals_scale = pd.read_csv('output/df_bifvalues_scaleup.csv')

alpha_vals = df_bif_vals_alpha['alpha'].values
scale_vals = df_bif_vals_scale['scale_up'].values

model_sims = 5
id_vals = np.arange(int(model_sims/5)) # number of simulations at each combo of rof and sigma


#--------------
# forced and null trajectories at different alpha values
#---------------
print('Simulate forced trajectories and compute EWS')
list_ktau_forced = []
list_dl_forced = []
list_ktau_null = []
list_dl_null = []

# Model parameters
A = 88
B = 122
C = 40
D = 28
tau = 180
alpha_vals = [0, 0.1, 0.2, 0.3, 0.4]

D0 = 200
M0 = 1

sigma = 0.05

for idx, alpha in enumerate(alpha_vals):
    bif_val = df_bif_vals_alpha[idx]
    
    Tstart = 300
    Tcrit = 200
    Tfinal = bif_val
    niter = 500
    Tvals = np.linspace(Tstart, Tfinal, niter)    
    
    for id_val in id_vals:
        

        # Simulate forced trajectory up to bifurcation
        s_forced = funs.simulate_model(A, B, C, D, tau, alpha, M0, D0, Tvals, sigma, sigma)
        
        # Simulate a null trajectory up to bifurcation
        s_null = funs.simulate_model(A, B, C, D, tau, alpha, M0, D0, [Tstart]*niter, sigma, sigma)
                
        # Compute EWS for forced trajectory
        ts = ewstools.TimeSeries(s_forced)
        ts.detrend(method='Lowess', span=span)
        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)
        ts.compute_ktau(tmin=0, tmax=niter-1)
        dic_ktau = ts.ktau
        dic_ktau['alpha'] = alpha
        dic_ktau['id'] = id_val
        list_ktau_forced.append(dic_ktau)
        
        # Get DL predictions for forced trajectory
        ts.apply_classifier(m1, tmin=0, tmax=niter-1, name='m1', verbose=0)
        ts.apply_classifier(m2, tmin=0, tmax=niter-1, name='m2', verbose=0)
        df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
        df_dl_preds['alpha'] = alpha
        df_dl_preds['id'] = id_val
        list_dl_forced.append(df_dl_preds)
    
        # Compute EWS for null trajectory
        ts = ewstools.TimeSeries(s_null)
        ts.detrend(method='Lowess', span=span)
        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)
        ts.compute_ktau(tmin=0, tmax=niter-1)
        dic_ktau = ts.ktau
        dic_ktau['alpha'] = alpha
        dic_ktau['id'] = id_val
        list_ktau_null.append(ts.ktau)
        
        # Get DL predictions for null
        ts.apply_classifier(m1, tmin=0, tmax=niter-1, name='m1', verbose=0)
        ts.apply_classifier(m2, tmin=0, tmax=niter-1, name='m2', verbose=0)
        df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
        df_dl_preds['alpha'] = alpha
        df_dl_preds['id'] = id_val
        list_dl_null.append(df_dl_preds)
            
    print('Complete for alpha={}'.format(alpha))

df_ktau_forced = pd.DataFrame(list_ktau_forced)
df_dl_forced = pd.concat(list_dl_forced)

df_ktau_null = pd.DataFrame(list_ktau_null)
df_dl_null = pd.concat(list_dl_null)

# Export data
df_ktau_forced.to_csv('output/df_ktau_forced.csv', index=False)
df_ktau_null.to_csv('output/df_ktau_null.csv', index=False)
df_dl_forced.to_csv('output/df_dl_forced.csv', index=False)
df_dl_null.to_csv('output/df_dl_null.csv', index=False)


# Export time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Script took {:.2f} seconds'.format(time_taken))
