#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:47:56 2022

Simulare AR1 process and evaluate DL

@author: tbury
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

# # Parse command line arguments
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)

# args = parser.parse_args()
# use_inter_classifier = True if args.use_inter_classifier=='true' else False

use_inter_classifier=False

path_prefix = '../../code'

# Load in DL models
if use_inter_classifier:
    filepath_classifier = path_prefix+'/dl_train/output/'
else:
    filepath_classifier = path_prefix+'/../data/'

m1 = load_model(filepath_classifier+'classifier_1.pkl')
m2 = load_model(filepath_classifier+'classifier_2.pkl')
print('TF models loaded')


sigma = 0.01
x0 = 0

rho_vals = np.arange(-1,1, 0.1)[1:]
l_vals = np.arange(100,600,100)
nsims = 20

list_df = []

for rho in rho_vals:
    for l in l_vals:
        for sim in np.arange(nsims):
    
            # Simulate AR1 process
            list_x = []
            x = x0
            for i in np.arange(l):
                x = rho*x + sigma*np.random.normal(0,1)
                list_x.append(x)
            
            # Compute EWS
            s = pd.Series(list_x)
            ts = ewstools.TimeSeries(s)
            ts.detrend(method='Lowess', span=0.25)
            
            ts.apply_classifier(m1, tmin=s.index[0], tmax=s.index[-1], name='m1', verbose=0)
            ts.apply_classifier(m2, tmin=s.index[0], tmax=s.index[-1], name='m2', verbose=0)
            df_dl = ts.dl_preds.groupby('time').mean(numeric_only=True)
            
            df_dl['rho'] = rho
            df_dl['length'] = l
            df_dl['sim'] = sim
            
            list_df.append(df_dl)
        print('Complete for length={}'.format(l))
    print('Complete for rho={}'.format(rho))

df_dl_full = pd.concat(list_df).reset_index()

# Export DL predictions
df_dl_full.to_csv('output/df_sigma_{}.csv'.format(sigma), index=False)










