#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

Simulate Fox model to get bifurcation diagrams for different parameter values
Export data for bifurcation diagrams
Obtain and export bifurcation values

@author: tbury
"""


import time
start_time = time.time()

import numpy as np
import pandas as pd

import funs_fox as funs

import plotly.express as px

np.random.seed(0)

#---------
# Bifurcation diagrams with different alpha value
#----------

# Model parameters
A = 88
B = 122
C = 40
D = 28
tau = 180
alpha_vals = [0, 0.1, 0.2, 0.3, 0.4]

D0 = 200
M0 = 1

# Map out bifurcation diagram
Tstart = 300
Tcrit = 200
Tfinal = 100
Tvals = np.arange(Tstart, Tfinal, -0.1)

list_df = []

for alpha in alpha_vals:
    for Tfixed in Tvals:
        # Simulate model without noise
        niter = 100
        sigma = 0
        df_traj = funs.simulate_model(A, B, C, D, tau, alpha, M0, D0, [Tfixed]*niter, sigma, sigma)
        # Keep last 10 data points
        df_end = df_traj.iloc[-2:].copy()
        df_end['T'] = Tfixed
        df_end['alpha'] = alpha
        list_df.append(df_end)

df_bif = pd.concat(list_df)

# Export bifurcation data
df_bif.to_csv('output/df_bif_alpha.csv', index=False)

# Find bifurcation points (where two different D values emerge. Diff > eps)
eps = 0.5
def is_two_branches(df):
    max_d = df['D'].max()
    min_d = df['D'].min()
    if max_d - min_d > eps:
        return True
    else:
        return False

df_branches = df_bif.groupby(['alpha','T']).apply(is_two_branches).to_frame(name='two_branches')

df_branches_two = df_branches.query('two_branches==True')
bif_vals = []
for alpha in alpha_vals:
    bif_val = df_branches_two.loc[alpha].iloc[-1].name
    bif_vals.append((alpha, bif_val))

df_bif_vals = pd.DataFrame({'alpha':[tup[0] for tup in bif_vals],
                            'bif':[tup[1] for tup in bif_vals]})

df_bif_vals.to_csv('output/df_bifvalues_alpha.csv', index=False)



# # df_bif.plot(x='T', y='D', kind='scatter', color='alpha')
# fig = px.scatter(df_bif, x='T', y='D', color='alpha')
# fig.update_yaxes(range=[90,200])
# fig.write_html('figures/fig_bif_alpha.html')




#---------
# Bifurcation diagrams with different scaling factor
#----------

# Model parameters
scale_up_vals = [0.7, 0.8, 0.9, 1.1, 1.2]
tau = 180
alpha=0.2

D0 = 200
M0 = 1

# Map out bifurcation diagram
Tstart = 350
Tfinal = 50
Tvals = np.arange(Tstart, Tfinal, -0.1)

list_df = []

for scale_up in scale_up_vals:
    
    A = 88*scale_up
    B = 122*scale_up
    C = 40*scale_up
    D = 28*scale_up    
    
    for Tfixed in Tvals:

        # Simulate model without noise
        niter = 200
        sigma = 0
        df_traj = funs.simulate_model(A, B, C, D, tau, alpha, M0, D0, [Tfixed]*niter, sigma, sigma)
        # Keep last 10 data points
        df_end = df_traj.iloc[-2:].copy()
        df_end['T'] = Tfixed
        df_end['scale_up'] = scale_up
        list_df.append(df_end)

df_bif = pd.concat(list_df)
# df_bif['scale_up'] = df_bif['scale_up'].astype(str)

# Export bifurcation data
df_bif.to_csv('output/df_bif_scaleup.csv', index=False)


fig = px.scatter(df_bif, x='T', y='D', color='scale_up')
fig.update_yaxes(range=[90,200])
fig.write_html('figures/fig_bif_scaleup.html')
    

# Get bifurcation values
df_branches = df_bif.groupby(['scale_up','T']).apply(is_two_branches).to_frame(name='two_branches')

df_branches_two = df_branches.query('two_branches==True')
bif_vals = []
for scale_up in scale_up_vals:
    bif_val = df_branches_two.loc[scale_up].iloc[-1].name
    bif_vals.append((scale_up, bif_val))

df_bif_vals = pd.DataFrame({'scale_up':[tup[0] for tup in bif_vals],
                            'bif':[tup[1] for tup in bif_vals]})

df_bif_vals.to_csv('output/df_bifvalues_scaleup.csv', index=False)






