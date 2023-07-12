#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:47:56 2022

Generate data for fig 2 - EWS in a sample of model simulations

@author: tbury
"""


import time
start_time = time.time()

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)

args = parser.parse_args()
use_inter_classifier = True if args.use_inter_classifier=='true' else False


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
sys.path.append('../test_fox')
sys.path.append('../test_westerhoff')
sys.path.append('../test_ricker')
sys.path.append('../test_kot')
sys.path.append('../test_lorenz')
import funs_fox, funs_westerhoff, funs_ricker, funs_kot, funs_lorenz

# Time increment between ML predictions
inc = 10

# EWS paramers
span = 0.25
rw = 0.5


# Load in DL models
if use_inter_classifier:
    filepath_classifier = '../dl_train/output/'
else:
    filepath_classifier = '../../data/'

m1 = load_model(filepath_classifier+'classifier_1.pkl')
m2 = load_model(filepath_classifier+'classifier_2.pkl')
print('TF models loaded')


#----------
# Simulate models and compute EWS
#----------

## Fox period-doubling model
np.random.seed(0)
sigma = 0.05
s_forced, transition, s_null = funs_fox.sim_rate_forcing(sigma)

# Compute EWS
ts_pd = ewstools.TimeSeries(s_forced, transition=transition)
ts_pd.detrend(method='Lowess', span=span)
ts_pd.compute_var(rolling_window=rw)
ts_pd.compute_auto(rolling_window=rw, lag=1)

# Get DL predictions
ts_pd.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
ts_pd.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
ts_pd.dl_preds_mean = ts_pd.dl_preds.groupby('time').mean(numeric_only=True)
print('EWS computed for Fox model')
        


## Westerhoff NS model
np.random.seed(2)
sigma = 0.1
s_forced, transition, s_null = funs_westerhoff.sim_rate_forcing(sigma)

# Compute EWS
ts_ns = ewstools.TimeSeries(s_forced, transition=transition)
ts_ns.detrend(method='Lowess', span=span)
ts_ns.compute_var(rolling_window=rw)
ts_ns.compute_auto(rolling_window=rw, lag=1)

# Get DL predictions
ts_ns.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
ts_ns.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
ts_ns.dl_preds_mean = ts_ns.dl_preds.groupby('time').mean(numeric_only=True)
print('EWS computed for Westerhoff model')

        


## Ricker fold model
np.random.seed(0)
sigma = 0.1
s_forced, transition, s_null = funs_ricker.sim_rate_forcing(sigma)

# Compute EWS
ts_fold = ewstools.TimeSeries(s_forced, transition=transition)
ts_fold.detrend(method='Lowess', span=span)
ts_fold.compute_var(rolling_window=rw)
ts_fold.compute_auto(rolling_window=rw, lag=1)

# Get DL predictions
ts_fold.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
ts_fold.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
ts_fold.dl_preds_mean = ts_fold.dl_preds.groupby('time').mean(numeric_only=True)
print('EWS computed for Ricker model')





## Kot transcritical model
np.random.seed(2)
sigma = 0.005
s_forced, transition, s_null = funs_kot.sim_rate_forcing(sigma)

# Compute EWS
ts_tc = ewstools.TimeSeries(s_forced, transition=transition)
ts_tc.detrend(method='Lowess', span=span)
ts_tc.compute_var(rolling_window=rw)
ts_tc.compute_auto(rolling_window=rw, lag=1)

# Get DL predictions
ts_tc.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
ts_tc.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
ts_tc.dl_preds_mean = ts_tc.dl_preds.groupby('time').mean(numeric_only=True)
print('EWS computed for Kot model')




## Lorenz pitchfork model
np.random.seed(0)
sigma = 0.005
s_forced, transition, s_null = funs_lorenz.sim_rate_forcing(sigma)

# Compute EWS
ts_pf = ewstools.TimeSeries(s_forced, transition=transition)
ts_pf.detrend(method='Lowess', span=span)
ts_pf.compute_var(rolling_window=rw)
ts_pf.compute_auto(rolling_window=rw, lag=1)

# Get DL predictions
ts_pf.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
ts_pf.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
ts_pf.dl_preds_mean = ts_pf.dl_preds.groupby('time').mean(numeric_only=True)
print('EWS computed for Lorenz model')


#--------
# Collect data and save
#--------

df_pd = ts_pd.state.join(ts_pd.ews)
df_pd = df_pd.join(ts_pd.dl_preds_mean)
df_pd['model'] = 'pd'

df_ns = ts_ns.state.join(ts_ns.ews)
df_ns = df_ns.join(ts_ns.dl_preds_mean)
df_ns['model'] = 'ns'

df_fold = ts_fold.state.join(ts_fold.ews)
df_fold = df_fold.join(ts_fold.dl_preds_mean)
df_fold['model'] = 'fold'

df_tc = ts_tc.state.join(ts_tc.ews)
df_tc = df_tc.join(ts_tc.dl_preds_mean)
df_tc['model'] = 'tc'

df_pf = ts_pf.state.join(ts_pf.ews)
df_pf = df_pf.join(ts_pf.dl_preds_mean)
df_pf['model'] = 'pf'

df_plot = pd.concat([df_pd, df_ns, df_fold, df_tc, df_pf])

df_plot.to_csv('output/df_plot.csv')

# Export time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Script took {:.2f}s'.format(time_taken))


















