#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:46:50 2022

-Compute EWS and DL predictions on data
-Export data to output dir

@author: tbury
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

np.random.seed(0)

# Load in DL classifiers
path = '../../dl_train/trained_models/'
classifier_names = [
    'model_1_mtype_1_nsims_10000_sigma_0.005_0.015.pkl',
    'model_1_mtype_2_nsims_10000_sigma_0.005_0.015.pkl'
    ]

list_classifiers=[]
for name in classifier_names:
    classifier = load_model(path+name)
    list_classifiers.append(classifier)

print('TF models loaded')

# Load in trajectory data
df_traj = pd.read_csv('data/df_chick.csv')
df_traj_pd = df_traj[df_traj['type']=='pd']
df_traj_null = df_traj[df_traj['type']=='neutral']

# Load in transition times
df_transition = pd.read_csv('output/df_transitions.csv')


#--------
# Compute EWS for period-doubling trajectories
#--------

list_ews = []
list_dl = []

list_tsid = df_traj_pd['tsid'].unique()
# for tsid in list_tsid:
for tsid in list_tsid:
    
    df_spec = df_traj_pd[df_traj_pd['tsid']==tsid].set_index('Beat number')
    transition = df_transition[df_transition['tsid']==tsid]['transition'].iloc[0]
    s = df_spec['IBI (s)'].iloc[:]
    
    # Compute EWS
    ts = ewstools.TimeSeries(s, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=20)
    
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    
    # Get DL predictions
    for idx, classifier in enumerate(list_classifiers):
        ts.apply_classifier_inc(classifier, inc=5, name=str(idx), verbose=0)
    
    # Save data to lists
    df_ews = ts.state.join(ts.ews)
    df_ews['tsid'] = tsid
    list_ews.append(df_ews)
    
    df_dl = ts.dl_preds
    df_dl['tsid'] = tsid
    df_dl.rename({'time':'Beat number'}, axis=1)
    list_dl.append(df_dl)
    
    # fig = ts.make_plotly(kendall_tau=False, ens_avg=True)
    # fig.write_html('figures/ews_single/{}_pd.html'.format(tsid))
    
    print('Complete for tsid={}\n'.format(tsid))

df_ews_pd = pd.concat(list_ews)
df_dl_pd = pd.concat(list_dl)

# Export period-doubling EWS
df_ews_pd.to_csv('output/df_ews_pd.csv')
df_dl_pd.to_csv('output/df_dl_pd.csv', index=False)


#--------
# Compute EWS for null trajectories
#--------

list_ews = []
list_dl = []
list_tsid = df_traj_null['tsid'].unique()
for tsid in list_tsid:
    
    df_spec = df_traj_null[df_traj_null['tsid']==tsid].set_index('Beat number')
    s = df_spec['IBI (s)'].iloc[:]
    
    # Compute EWS
    ts = ewstools.TimeSeries(s)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=20)
    
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    
    # Get DL predictions
    for idx, classifier in enumerate(list_classifiers):
        ts.apply_classifier_inc(classifier, inc=5, name=str(idx), verbose=0)
    
    # Save data to lists
    df_ews = ts.state.join(ts.ews)
    df_ews['tsid'] = tsid
    list_ews.append(df_ews)
    
    df_dl = ts.dl_preds
    df_dl['tsid'] = tsid
    df_dl.rename({'time':'Beat number'}, axis=1)
    list_dl.append(df_dl)
    
    # fig = ts.make_plotly(kendall_tau=False, ens_avg=True)
    # fig.write_html('figures/ews_single/{}_pd.html'.format(tsid))
    
    print('Complete for tsid={}\n'.format(tsid))    
    
df_ews_null = pd.concat(list_ews)
df_dl_null = pd.concat(list_dl)

# Export null EWS
df_ews_null.to_csv('output/df_ews_null.csv')
df_dl_null.to_csv('output/df_dl_null.csv', index=False)



