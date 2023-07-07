#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute ROC curve data using predictions at evaluation points
- Do for various values of rolling window

@author: tbury
"""


import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

import sklearn.metrics as metrics

np.random.seed(0)
rw = 0.05

# Import df predictions
df_ktau_forced = pd.read_csv('output/df_ktau_pd_fixed_rw_{}.csv'.format(rw))
df_ktau_null = pd.read_csv('output/df_ktau_null_fixed_rw_{}.csv'.format(rw))

#----------------
# compute ROC curves
#----------------
print('Compute ROC curves')

df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0

df_ktau = pd.concat([df_ktau_forced, df_ktau_null])


def roc_compute(truth_vals, indicator_vals):
    
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals,indicator_vals)
    
    # Compute AUC (area under curve)
    auc = metrics.auc(fpr, tpr)
    
    # Put into a DF
    dic_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds, 'auc':auc}
    df_roc = pd.DataFrame(dic_roc)

    return df_roc


# Initiliase list for ROC dataframes for predicting May fold bifurcation
list_roc = []

# Assign indicator and truth values for variance
indicator_vals = df_ktau['variance']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals,indicator_vals)
df_roc['ews'] = 'Variance'
list_roc.append(df_roc)


# Assign indicator and truth values for lag-1 AC
indicator_vals = -df_ktau['ac1']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals,indicator_vals)
df_roc['ews'] = 'Lag-1 AC'
list_roc.append(df_roc)

# Concatenate roc dataframes
df_roc_full = pd.concat(list_roc, ignore_index=True)

# Export ROC data
filepath = 'output/df_roc_rw_{}.csv'.format(rw)
df_roc_full.to_csv(filepath,
                   index=False,)

auc_vals = df_roc_full.groupby('ews')['auc'].max()
print(auc_vals)

print('Exported ROC data to {}'.format(filepath))

















