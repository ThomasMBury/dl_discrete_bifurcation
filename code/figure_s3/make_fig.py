#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:25:54 2022

Make plot of individual trajectories
at different values of sigma and rof

Put simulations of each model into sigle subplot

@author: tbury
"""

import numpy as np
import pandas as pd

import plotly.express as px


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append('../test_fox')
sys.path.append('../test_westerhoff')
sys.path.append('../test_ricker')
sys.path.append('../test_kot')
sys.path.append('../test_lorenz')
import funs_fox, funs_westerhoff, funs_ricker, funs_kot, funs_lorenz

cols = px.colors.qualitative.Plotly

#--------------
# Run simulations for each model
#--------------

list_df = []
list_transitions = []

## Fox model
np.random.seed(0)
sigma_vals = [0.00625, 0.025,  0.1]
rof_vals = [100/500,  100/300,  100/100]

for sigma in sigma_vals:
    for rof in rof_vals:
        s_forced, transition, s_null = funs_fox.sim_rate_forcing(sigma, rof)
        df = pd.DataFrame(s_forced)
        df['sigma'] = sigma
        df['rof'] = rof
        df['model'] = 'fox'
        df['transition'] = transition
        df['state'] = df['D']
        df.drop('D', axis=1, inplace=True)
        list_df.append(df)


## Westerhoff model
np.random.seed(0)
sigma_vals = [0.00625, 0.025,  0.1]
rof_vals = [5/500,  5/300,  5/100]

for sigma in sigma_vals:
    for rof in rof_vals:
        s_forced, transition, s_null = funs_westerhoff.sim_rate_forcing(sigma, rof)
        df = pd.DataFrame(s_forced)
        df['sigma'] = sigma
        df['rof'] = rof
        df['model'] = 'westerhoff'
        df['transition'] = transition
        df['state'] = df['y']
        df.drop('y', axis=1, inplace=True)
        list_df.append(df)


## Ricker model
np.random.seed(0)
sigma_vals = [0.0125, 0.05,  0.2]
rof_vals = [2.36/500,  2.36/300,  2.36/100]

for sigma in sigma_vals:
    for rof in rof_vals:
        s_forced, transition, s_null = funs_ricker.sim_rate_forcing(sigma, rof)
        df = pd.DataFrame(s_forced)
        df['sigma'] = sigma
        df['rof'] = rof
        df['model'] = 'ricker'
        df['transition'] = transition
        df['state'] = df['x']
        df.drop('x', axis=1, inplace=True)
        list_df.append(df)


## Lotka volterra model
np.random.seed(0)
sigma_vals = [0.000625, 0.0025,  0.01]
rof_vals = [0.5/500,  0.5/300,  0.5/100]

for sigma in sigma_vals:
    for rof in rof_vals:
        s_forced, transition, s_null = funs_kot.sim_rate_forcing(sigma, rof)
        df = pd.DataFrame(s_forced)
        df['sigma'] = sigma
        df['rof'] = rof
        df['model'] = 'lv'
        df['transition'] = transition
        df['state'] = df['x']
        df.drop('x', axis=1, inplace=True)
        list_df.append(df)


## Lorenz model
np.random.seed(0)
sigma_vals = [0.000625, 0.0025,  0.01]
rof_vals = [0.5/500,  0.5/300,  0.5/100]

for sigma in sigma_vals:
    for rof in rof_vals:
        s_forced, transition, s_null = funs_lorenz.sim_rate_forcing(sigma, rof)
        df = pd.DataFrame(s_forced)
        df['sigma'] = sigma
        df['rof'] = rof
        df['model'] = 'lorenz'
        df['transition'] = transition
        df['state'] = df['x']
        df.drop('x', axis=1, inplace=True)
        list_df.append(df)


df = pd.concat(list_df).reset_index()

df['label'] = df.apply(
    lambda x: 'sigma={}, model={}'.format(x['sigma'], x['model']), 
    axis=1)



fig = px.line(df, x='time', y='state', facet_col='label', facet_col_wrap=3,
              color='rof')


# Update trace colors
for i in range(45):
    fig.data[i]['line']['color'] = 'black'


for i in [0,1,2,9,10,11,18,19,20,27,28,29,30,31,32]:
    fig.data[i]['line']['color'] = cols[0]
for i in [3,4,5,12,13,14,21,22,23,33,34,35,36,37,38]:
    fig.data[i]['line']['color'] = cols[1]
for i in [6,7,8,15,16,17,24,25,26,39,40,41,42,43,44]:
    fig.data[i]['line']['color'] = cols[2]
    

    



sigma_vals = df.groupby('label', sort=False)['sigma'].mean().values
titles_orig = ['label={}'.format(s) for s in df['label'].unique()]
subplot_titles = ['sigma = {:.2g}'.format(sigma) for sigma in sigma_vals]
dict_titles = dict(zip(titles_orig, subplot_titles))

fig.for_each_annotation(lambda a: a.update(text = dict_titles[a.text]))


    
# for i in np.arange(3):
#     fig.data[i]['line']['color'] = cols[4]
    
# for i in np.arange(3,6):
#     fig.data[i]['line']['color'] = cols[5]
    
              
    
fig.update_traces(line=dict(width=1.2))

fig.update_xaxes(matches=None, automargin=False)
fig.update_yaxes(matches=None, automargin=False)


fig.update_yaxes(title='State', col=1)
fig.update_xaxes(title='Time', row=1)


yrange5 = [100,190]
yrange4 = [29, 47]
yrange3 = [-1,11]
yrange2 = [0.6,1.1]
yrange1 = [-0.55, 0.55]

fig.update_yaxes(range=yrange5, row=5)
fig.update_yaxes(range=yrange4, row=4)
fig.update_yaxes(range=yrange3, row=3)
fig.update_yaxes(range=yrange2, row=2)
fig.update_yaxes(range=yrange1, row=1)



fig.update_layout(font=dict(family='Times New Roman', size=15),
                  showlegend=False,
                  margin=dict(l=50,r=50,t=50,b=50),
                  height=900,
                  width=750,
                  )


# # Add lines for transitions

# list_shapes = []

# axis_numbers = ['']+[str(i) for i in np.arange(2,16)]

# for i, axis in enumerate(axis_numbers):
#     shape = {'type': 'line', 
#               'x0': 100, 
#               'y0': yrange5[0], 
#               'x1': 100, 
#               'y1': yrange5[1], 
#               'xref': 'x{}'.format(axis), 
#               'yref': 'y{}'.format(axis),
#               'line': {'width':1.5,'dash':'dot','color':cols[2]},
#               }
#     list_shapes.append(shape)
    
#     shape = {'type': 'line', 
#               'x0': 300, 
#               'y0': yrange5[0], 
#               'x1': 300, 
#               'y1': yrange5[1], 
#               'xref': 'x{}'.format(axis), 
#               'yref': 'y{}'.format(axis),
#               'line': {'width':1.5,'dash':'dot','color':cols[1]},
#               }
#     list_shapes.append(shape)    
    
#     shape = {'type': 'line', 
#               'x0': 500, 
#               'y0': yrange5[0], 
#               'x1': 500, 
#               'y1': yrange5[1], 
#               'xref': 'x{}'.format(axis), 
#               'yref': 'y{}'.format(axis),
#               'line': {'width':1.5,'dash':'dot','color':cols[0]},
#               }
#     list_shapes.append(shape)    

# fig['layout'].update(shapes=list_shapes)



## Add annotations for each model name

fig.add_annotation(
    row=5,col=3, 
    x=1.15, y=0.5, 
    xref='x domain',
    yref='y domain',
    text='Fox',
    showarrow=False,
    font = dict(color='black', size=17),
    textangle=90,
    )

fig.add_annotation(
    row=4,col=3, 
    x=1.15, y=0.5, 
    xref='x domain',
    yref='y domain',
    text='Westerhoff',
    showarrow=False,
    font = dict(color='black', size=17),
    textangle=90,
    )

fig.add_annotation(
    row=3,col=3, 
    x=1.15, y=0.5, 
    xref='x domain',
    yref='y domain',
    text='Ricker',
    showarrow=False,
    font = dict(color='black', size=17),
    textangle=90,
    )

fig.add_annotation(
    row=2,col=3, 
    x=1.15, y=0.5, 
    xref='x domain',
    yref='y domain',
    text='Lotka-Volterra',
    showarrow=False,
    font = dict(color='black', size=17),
    textangle=90,
    )

fig.add_annotation(
    row=1,col=3, 
    x=1.15, y=0.5, 
    xref='x domain',
    yref='y domain',
    text='Lorenz',
    showarrow=False,
    font = dict(color='black', size=17),
    textangle=90,
    )


fig.write_image('../../results/figure_s3.png', 
                scale=4,
                )






