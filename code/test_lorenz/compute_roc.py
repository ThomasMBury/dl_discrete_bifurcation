#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:58:49 2020

Compute ROC curves for EWS and DL predictions.

@author: Thomas M. Bury
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scipy.stats as stats

import funs_lorenz as funs

#-------------
# Import predictions
#–------------

df_ktau_forced = pd.read_csv('output/df_ktau_forced.csv')
df_ktau_null = pd.read_csv('output/df_ktau_null.csv')
df_dl_forced = pd.read_csv('output/df_dl_forced.csv')
df_dl_null = pd.read_csv('output/df_dl_null.csv')


#----------------
# compute ROC curves
#----------------
print('Compute ROC curves')

df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0

df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

bif_labels = ['1','2','3','4','5']
df_dl['any_bif'] = df_dl[bif_labels].sum(axis=1)


## Get data on ML favoured bifurcation for each forced trajectory
df_dl['fav_bif'] = df_dl[bif_labels].idxmax(axis=1)

# Count each bifurcation choice for forced trajectories
counts = df_dl[df_dl['truth_value']==1]['fav_bif'].value_counts()
df_counts = pd.DataFrame(index=bif_labels)
df_counts.index.name='bif_id'
df_counts['count'] = counts
# Nan as 0
df_counts.fillna(value=0, inplace=True)

# Export data on bifurcation prediction counts
filepath = 'output/df_fav_bif.csv'
df_counts.to_csv(filepath)
print('Exported bifurcation count data to {}'.format(filepath))



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

# Assign indicator and truth values for ML prediction
indicator_vals = df_dl['any_bif']
truth_vals = df_dl['truth_value']
df_roc = roc_compute(truth_vals,indicator_vals)
df_roc['ews'] = 'DL bif'
list_roc.append(df_roc)


# Assign indicator and truth values for variance
indicator_vals = df_ktau['variance']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals,indicator_vals)
df_roc['ews'] = 'Variance'
list_roc.append(df_roc)


# Assign indicator and truth values for lag-1 AC
indicator_vals = df_ktau['ac1']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals,indicator_vals)
df_roc['ews'] = 'Lag-1 AC'
list_roc.append(df_roc)

# Concatenate roc dataframes
df_roc_full = pd.concat(list_roc, ignore_index=True)

# Export ROC data
filepath = 'output/df_roc.csv'
df_roc_full.to_csv(filepath,
                    index=False,)

print('Exported ROC data to {}'.format(filepath))



# TEMP work
# Plot a histogram with error bars of the weights
df_plot = df_dl.query('truth_value==1')[['1','2','3','4','5']]
df_plot.boxplot()




















