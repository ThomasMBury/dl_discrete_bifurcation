#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute EWS in chick heart data 
- after applying random perturbation to data of magnitude = sample error
- generate 10x number of time series

@author: tbury
"""


import time
start_time = time.time()

import sys

import numpy as np
import pandas as pd

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

import os

use_inter_classifier =  False

np.random.seed(0)

eval_pts = np.arange(0.64, 1.01, 0.04) #  percentage of way through pre-transition time series

pert_sigma = 0.0125 # smaple error is +/- 0.0125 ms (set as lower and upper value for uniform dist.)

n_perts = 10 # number of new time series to create from perturbations

path_prefix = '../../code'

# Load in DL models
if use_inter_classifier:
    filepath_classifier = path_prefix+'/dl_train/output/'
else:
    filepath_classifier = path_prefix+'/../data/'

m1 = load_model(filepath_classifier+'classifier_1.pkl')
m2 = load_model(filepath_classifier+'classifier_2.pkl')
print('TF models loaded')


# EWS parameters
rw = 0.5 # rolling window
bw = 20 # Gaussian band width (# beats)

# Load in trajectory data
df = pd.read_csv('../../data/df_chick.csv')
df_pd = df[df['type']=='pd']
df_null = df[df['type']=='neutral']

# Load in transition times
df_transition = pd.read_csv('../../code/test_chick_heart/output/df_transitions.csv')
df_transition.set_index('tsid', inplace=True)


#---------------
# period-doubling trajectories
#---------------
list_ktau = []
list_dl_preds = []

list_tsid = df_pd['tsid'].unique()
for tsid in list_tsid:
   
    df_spec = df_pd[df_pd['tsid']==tsid].set_index('Beat number')
    transition = df_transition.loc[tsid]['transition']
    series = df_spec['IBI (s)']
    
    
    for pert_num in np.arange(1,n_perts+1):
        # Apply perturbation to each data point
        perturbs = np.random.uniform(low=-pert_sigma, high=pert_sigma, size=len(series))
        series_pert = series+perturbs
        
        # Compute EWS
        ts = ewstools.TimeSeries(series_pert, transition=transition)
        # ts.detrend(method='Lowess', span=50)
        ts.detrend(method='Gaussian', bandwidth=bw)
        
        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)
        
        for eval_pt in eval_pts:
            
            eval_time = transition*eval_pt
            
            # Compute kendall tau at evaluation points
            ts.compute_ktau(tmin=0, tmax=eval_time)
            dic_ktau = ts.ktau
            dic_ktau['eval_time'] = eval_time
            dic_ktau['tsid'] = tsid
            dic_ktau['pert'] = pert_num
            list_ktau.append(dic_ktau)
        
            # Get DL predictions at eval pts
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)
             
            df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
            df_dl_preds['eval_time']=eval_time
            df_dl_preds['tsid'] = tsid
            df_dl_preds['pert'] = pert_num
            list_dl_preds.append(df_dl_preds)
            ts.clear_dl_preds()

    print('Complete for pd tsid={}'.format(tsid))

df_ktau_forced = pd.DataFrame(list_ktau)
df_dl_forced = pd.concat(list_dl_preds)


#-------------
# null trajectories
#-------------
print('Simulate null trajectories and compute EWS')

list_ktau = []
list_dl_preds = []

list_tsid = df_null['tsid'].unique()

for tsid in list_tsid:
    
    df_spec = df_null[df_null['tsid']==tsid].set_index('Beat number')
    series = df_spec['IBI (s)']   
    
    for pert_num in np.arange(1,n_perts+1):
        # Apply perturbation to each data point
        perturbs = np.random.uniform(low=-pert_sigma, high=pert_sigma, size=len(series))
        series_pert = series+perturbs
    
    
        # Compute EWS
        ts = ewstools.TimeSeries(series_pert)
        # ts.detrend(method='Lowess', span=50)
        ts.detrend(method='Gaussian', bandwidth=bw)
        
        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)
        
        for eval_pt in eval_pts:
            
            eval_time = eval_pt*series.index[-1]
            
            # Compute kendall tau at evaluation points
            ts.compute_ktau(tmin=0, tmax=eval_time)
            dic_ktau = ts.ktau
            dic_ktau['eval_time'] = eval_time
            dic_ktau['tsid'] = tsid
            dic_ktau['pert'] = pert_num
            list_ktau.append(dic_ktau)
        
            # Get DL predictions at eval pts
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)
                
            df_dl_preds = ts.dl_preds.groupby('time').mean(numeric_only=True) # use mean DL pred
            df_dl_preds['eval_time']=eval_time
            df_dl_preds['tsid']=tsid
            df_dl_preds['pert']=pert_num
            list_dl_preds.append(df_dl_preds)
            ts.clear_dl_preds()
        
    print('Complete for null tsid={}'.format(tsid))
        
    
df_ktau_null = pd.DataFrame(list_ktau)
df_dl_null = pd.concat(list_dl_preds)

# Export data
df_ktau_forced.to_csv('output/df_ktau_pd_fixed_pert_{}.csv'.format(pert_sigma), index=False)
df_ktau_null.to_csv('output/df_ktau_null_fixed_pert_{}.csv'.format(pert_sigma), index=False)
df_dl_forced.to_csv('output/df_dl_pd_fixed_pert_{}.csv'.format(pert_sigma), index=False)
df_dl_null.to_csv('output/df_dl_null_fixed_pert_{}.csv'.format(pert_sigma), index=False)

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))



