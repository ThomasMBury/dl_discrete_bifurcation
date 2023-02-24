#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute DL and kendall tau at fixed evaluation points in chick heart data

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

np.random.seed(0)


eval_pts = np.arange(0.64, 1.01, 0.04) #  percentage of way through pre-transition time series

# Load in DL models
m1 = load_model('../dl_train/classifier_1.pkl')
m2 = load_model('../dl_train/classifier_2.pkl')
print('TF models loaded')


# Load in trajectory data
df = pd.read_csv('../../data/df_chick.csv')
df_pd = df[df['type']=='pd']
df_null = df[df['type']=='neutral']

# Load in transition times
df_transition = pd.read_csv('output/df_transitions.csv')
df_transition.set_index('tsid', inplace=True)


#--------------
# period-doubling trajectories
#---------------
list_ktau = []
list_dl_preds = []

list_tsid = df_pd['tsid'].unique()
for tsid in list_tsid:
    
    df_spec = df_pd[df_pd['tsid']==tsid].set_index('Beat number')
    transition = df_transition.loc[tsid]['transition']
    series = df_spec['IBI (s)']
    
    # Compute EWS
    ts = ewstools.TimeSeries(series, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=20)
    
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    
    for eval_pt in eval_pts:
        
        eval_time = transition*eval_pt
        
        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)
    
        # Get DL predictions at eval pts
        ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
        ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)
         
        df_dl_preds = ts.dl_preds.groupby('time').mean() # use mean DL pred
        df_dl_preds['eval_time']=eval_time
        df_dl_preds['tsid'] = tsid
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
    
    # Compute EWS
    ts = ewstools.TimeSeries(series)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=20)
    
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    
    for eval_pt in eval_pts:
        
        eval_time = eval_pt*series.index[-1]
        
        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)
    
        # Get DL predictions at eval pts
        ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
        ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)
            
        df_dl_preds = ts.dl_preds.groupby('time').mean() # use mean DL pred
        df_dl_preds['eval_time']=eval_time
        df_dl_preds['tsid']=tsid
        list_dl_preds.append(df_dl_preds)
        ts.clear_dl_preds()
    
    print('Complete for null tsid={}'.format(tsid))
        
df_ktau_null = pd.DataFrame(list_ktau)
df_dl_null = pd.concat(list_dl_preds)


# Export data
df_ktau_forced.to_csv('output/ktau_preds_60_100/df_ktau_forced.csv', index=False)
df_ktau_null.to_csv('output/ktau_preds_60_100/df_ktau_null.csv', index=False)
df_dl_forced.to_csv('output/dl_preds_60_100/df_dl_forced.csv', index=False)
df_dl_null.to_csv('output/dl_preds_60_100/df_dl_null.csv', index=False)


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))



# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'output/time_compute_ews_fixed.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))





