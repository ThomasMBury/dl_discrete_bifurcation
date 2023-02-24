#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:40:26 2022

Use training data models to generate trajectories that go
beyond the bifurcation.
Make fig. for each type of bifurcation

Key for trajectory type:
    0 : Null trajectory
    1 : Period-doubling trajectory
    2 : Neimark-Sacker trajectory
    3 : Fold trajectory
    4 : Transcritical trajectory
    5 : Pitchfork trajectory

@author: tbury
"""


import numpy as np
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cols = px.colors.qualitative.Plotly

import sys
sys.path.append('../training_data/')
from train_funs import \
    simulate_pd, simulate_fold, simulate_ns, simulate_tc, simulate_pf


font_size = 14
font_family = 'Times New Roman'
font_size_letter_label = 14

linewidth = 0.7
linewidth_bif = 1.2
linewidth_bif_dash = '5px'
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

# dist from axis to axis label
xaxes_standoff = 50
yaxes_standoff = 50

# Scale up factor on image export
scale = 8 # default dpi=72 - nature=300-600



tmax = 700
tburn = 100
max_order = 10 # Max polynomial degree
# Noise amplitude mean (drawn from triangular dist.)
sigma_mean = 0.05

# Transition thresholds
dev_thresh_fig=1e10 # set high so we can simulate full traj for fig
dev_thresh_train=0.4 # used to get training section from full traj

#------------
# Null
#------------

# Simulate trajectory
np.random.seed(0)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -1
# Run simulation
df_null = simulate_pd(bl, bl, tmax, tburn, sigma, max_order, dev_thresh_fig)

# Compute points for bifurcation branches
df_null_bif = pd.DataFrame()
df_null_bif['time'] = np.arange(0,tmax,0.01)
df_null_bif['mu'] = np.linspace(bl,bl,len(df_null_bif))
df_null_bif['stable_1'] = 0

# Compute end of training data (transition or bifurcation time)
time_end_null = 600


#------------
# Period-doubling bifurcation
#------------

# Simulate trajectory
np.random.seed(0)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -1
bh = 1/6
supercrit=True

# Run simulation
df_pd = simulate_pd(bl, bh, tmax, tburn, sigma, max_order, dev_thresh_fig, supercrit)
df_pd['mu'] = np.linspace(bl,bh,tmax) 

# df_pd.set_index('time').plot()

# Compute points for bifurcation branches
df_pd_bif = pd.DataFrame()
df_pd_bif['time'] = np.arange(0,tmax,0.01)
df_pd_bif['mu'] = np.linspace(bl,bh,len(df_pd_bif))
df_pd_bif['stable_1'] = df_pd_bif['mu'].apply(lambda x: 0 if x<=0 else np.nan)
df_pd_bif['stable_2'] = df_pd_bif['mu'].apply(lambda x: np.sqrt(x) if x>=0 else np.nan)
df_pd_bif['stable_3'] = df_pd_bif['mu'].apply(lambda x: -np.sqrt(x) if x>=0 else np.nan)
df_pd_bif['unstable_1'] = df_pd_bif['mu'].apply(lambda x: 0 if x>=0 else np.nan)

# Compute end of training data (transition or bifurcation time)
time_end_pd = min(600, df_pd[abs(df_pd['x'])>dev_thresh_train]['time'].iloc[0])



#------------
# Neimark-Sacker bifurcation
#------------

# Simulate trajectory
np.random.seed(0)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -1
bh = 1/6
theta = 2*np.pi/16
supercrit=True

# Run simulation
df_ns = simulate_ns(bl, bh, theta, tmax, tburn, sigma, max_order, dev_thresh_fig, supercrit)
df_ns['mu'] = np.linspace(bl,bh,tmax) 

# Compute points for bifurcation branches
df_ns_bif = pd.DataFrame()
df_ns_bif['time'] = np.arange(0,tmax,0.01)
df_ns_bif['mu'] = np.linspace(bl,bh,len(df_ns_bif))
df_ns_bif['stable_1'] = df_pd_bif['mu'].apply(lambda x: 0 if x<=0 else np.nan)
df_ns_bif['stable_2'] = df_pd_bif['mu'].apply(lambda x: np.sqrt(x) if x>=0 else np.nan)
df_ns_bif['stable_3'] = df_pd_bif['mu'].apply(lambda x: -np.sqrt(x) if x>=0 else np.nan)
df_ns_bif['unstable_1'] = df_pd_bif['mu'].apply(lambda x: 0 if x>=0 else np.nan)

# Compute end of training data (transition or bifurcation time)
time_end_ns = min(600, df_ns[abs(df_ns['x'])>dev_thresh_train]['time'].iloc[0])


#------------
# Fold bifurcation
#------------

# Simulate trajectory
np.random.seed(1)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -0.5
bh = 1/12
# Run simulation
df_fold = simulate_fold(bl, bh, tmax, tburn, sigma, max_order, dev_thresh_fig,
                        return_dev=False)
df_fold['mu'] = np.linspace(bl,bh,tmax) 

# Compute points for bifurcation branches
df_fold_bif = pd.DataFrame()
df_fold_bif['time'] = np.arange(0,tmax,0.01)
df_fold_bif['mu'] = np.linspace(bl,bh,len(df_fold_bif))
df_fold_bif['stable_1'] = df_fold_bif['mu'].apply(lambda x: np.sqrt(-x) if x<=0 else np.nan)
df_fold_bif['unstable_1'] = df_fold_bif['mu'].apply(lambda x: -np.sqrt(-x) if x<=0 else np.nan)

# Compute end of training data (transition or bifurcation time)
df_fold['dev'] = df_fold['x'] - df_fold['mu'].apply(lambda x: np.sqrt(-x) if x<0 else 0)
time_end_fold = min(600, df_fold[abs(df_fold['dev'])>dev_thresh_train]['time'].iloc[0])



#------------
# Transcritical bifurcation
#------------

# Simulate trajectory
np.random.seed(3)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -1
bh = 1/6
# Run simulation
df_tc = simulate_tc(bl, bh, tmax, tburn, sigma, max_order, dev_thresh_fig)
df_tc['mu'] = np.linspace(bl,bh,tmax) 

# Compute points for bifurcation branches
df_tc_bif = pd.DataFrame()
df_tc_bif['time'] = np.arange(0,tmax,0.01)
df_tc_bif['mu'] = np.linspace(bl,bh,len(df_tc_bif))
df_tc_bif['stable_1'] = df_tc_bif['mu'].apply(lambda x: 0 if x<=0 else x)
df_tc_bif['unstable_1'] = df_tc_bif['mu'].apply(lambda x: x if x<=0 else 0)

# Compute end of training data (transition or bifurcation time)
time_end_tc = min(600, df_tc[abs(df_tc['x'])>dev_thresh_train]['time'].iloc[0])


#------------
# Pitchfork
#------------

# Simulate trajectory
np.random.seed(1)
sigma =  sigma_mean * np.random.triangular(0.5,1,1.5)
# bl = np.random.uniform(-1.8,-0.2)
bl = -1
bh = 1/6
supercrit=True

# Run simulation
df_pf = simulate_pf(bl, bh, tmax, tburn, sigma, max_order, dev_thresh_fig, supercrit)
df_pf['mu'] = np.linspace(bl,bh,tmax) 

# Compute points for bifurcation branches
df_pf_bif = pd.DataFrame()
df_pf_bif['time'] = np.arange(0,tmax,0.01)
df_pf_bif['mu'] = np.linspace(bl,bh,len(df_pf_bif))
df_pf_bif['stable_1'] = df_pf_bif['mu'].apply(lambda x: 0 if x<=0 else np.sqrt(x))
df_pf_bif['stable_2'] = df_pf_bif['mu'].apply(lambda x: np.nan if x<=0 else -np.sqrt(x))
df_pf_bif['unstable_1'] = df_pf_bif['mu'].apply(lambda x: np.nan if x<=0 else 0)

# Compute end of training data (transition or bifurcation time)
time_end_pf = min(600, df_pf[abs(df_pf['x'])>dev_thresh_train]['time'].iloc[0])



#-----------
# Make subplot
#-----------

# Add as annotations manually (easier to set position)
subplot_titles = ['Null', 'Period-doubling', 'Neimark-Sacker', 'Fold',
                  'Transcritical', 'Pitchfork']

fig = make_subplots(rows=3, cols=2,
                    shared_xaxes=True,
                    shared_yaxes=True,
                    horizontal_spacing=0.05,
                    vertical_spacing=0.05,
                    # subplot_titles=subplot_titles,
                    )


#-----------Null----------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_null_bif['time'],y=df_null_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=1,col=1,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_null['time'], y=df_null['x'],
               mode='lines', line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=1,col=1,
)


#-----------Period-doubling------------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_pd_bif['time'],y=df_pd_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid',width=linewidth_bif)),
    row=1,col=2,
)

fig.add_trace(
    go.Scatter(x=df_pd_bif['time'],y=df_pd_bif['stable_2'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=1,col=2,
)

fig.add_trace(
    go.Scatter(x=df_pd_bif['time'],y=df_pd_bif['stable_3'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=1,col=2,
)

fig.add_trace(
    go.Scatter(x=df_pd_bif['time'],y=df_pd_bif['unstable_1'],
               mode='lines', 
               line=dict(color='black',dash=linewidth_bif_dash, width=linewidth_bif)),
    row=1,col=2,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_pd['time'], y=df_pd['x'],
               mode='lines', line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=1,col=2,
)

#-----------Neimark-Sacker------------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_ns_bif['time'],y=df_pd_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=2,col=1,
)

fig.add_trace(
    go.Scatter(x=df_ns_bif['time'],y=df_pd_bif['stable_2'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=2,col=1,
)

fig.add_trace(
    go.Scatter(x=df_ns_bif['time'],y=df_pd_bif['stable_3'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=2,col=1,
)

fig.add_trace(
    go.Scatter(x=df_ns_bif['time'],y=df_pd_bif['unstable_1'],
               mode='lines', 
               line=dict(color='black',dash=linewidth_bif_dash, width=linewidth_bif)),
    row=2,col=1,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_ns['time'], y=df_ns['x'],
               mode='lines', line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=2,col=1,
)

#-----------Fold-----------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_fold_bif['time'],y=df_fold_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=2,col=2,
)

fig.add_trace(
    go.Scatter(x=df_fold_bif['time'],y=df_fold_bif['unstable_1'],
               mode='lines', 
               line=dict(color='black',dash=linewidth_bif_dash, width=linewidth_bif)),
    row=2,col=2,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_fold['time'], y=df_fold['x'],
               mode='lines', 
               line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=2,col=2,
)


#-----------Transcritical-----------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_tc_bif['time'],y=df_tc_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=3,col=1,
)

fig.add_trace(
    go.Scatter(x=df_tc_bif['time'],y=df_tc_bif['unstable_1'],
               mode='lines', 
               line=dict(color='black',dash=linewidth_bif_dash, width=linewidth_bif)),
    row=3,col=1,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_tc['time'], y=df_tc['x'],
               mode='lines',
               line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=3,col=1,
)

#-----------Pitchfork----------
# bifurcation traces
fig.add_trace(
    go.Scatter(x=df_pf_bif['time'],y=df_pf_bif['stable_1'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=3,col=2,
)
fig.add_trace(
    go.Scatter(x=df_pf_bif['time'],y=df_pf_bif['stable_2'],
               mode='lines', 
               line=dict(color='black',dash='solid', width=linewidth_bif)),
    row=3,col=2,
)
fig.add_trace(
    go.Scatter(x=df_pf_bif['time'],y=df_pf_bif['unstable_1'],
               mode='lines', 
               line=dict(color='black',dash=linewidth_bif_dash, width=linewidth_bif)),
    row=3,col=2,
)

# trajectory
fig.add_trace(
    go.Scatter(x=df_pf['time'], y=df_pf['x'],
               mode='lines', line=dict(color=cols[0],width=linewidth),
               opacity=0.7),
    row=3,col=2,
)




#--------------Shapes
list_shapes = []
# Vertical lines for section that would be used as training data

end_times = [time_end_null, time_end_pd, time_end_ns,
             time_end_fold, time_end_tc, time_end_pf]

for idx, ax in enumerate(['']+list(np.arange(2,7))):
    
    end_time = end_times[idx]
    start_time = end_time - 500
    
    line_start = {'type': 'line', 
              'x0':start_time,
              'y0': -1, 
              'x1':start_time, 
              'y1': 1,
              'xref': 'x{}'.format(ax), 
              'yref': 'y{}'.format(ax),
              'line': dict(width=linewidth,dash='solid',color='green'),
              }
    line_end = {'type': 'line', 
              'x0':end_time,
              'y0': -1, 
              'x1':end_time, 
              'y1': 1,
              'xref': 'x{}'.format(ax), 
              'yref': 'y{}'.format(ax),
              'line': dict(width=linewidth,dash='solid',color='green'),
              }
    fig.add_shape(line_start)    
    fig.add_shape(line_end)    

   



#-----------------Annotations---------

# Letter labels for each panel
import string
label_letters = string.ascii_lowercase

axes_numbers = ['']+list(np.arange(2,7))
idx=0
for idx, ax in enumerate(axes_numbers):
    label_annotation = dict(
            x=0.01,
            y=1.12,
            text='<b>{}</b>'.format(label_letters[idx]),
            xref='x{} domain'.format(ax),
            yref='y{} domain'.format(ax),
            showarrow=False,
            font = dict(
                    color = "black",
                    size = font_size_letter_label)
            )
    fig.add_annotation(label_annotation)



#---------Subplot titles----------
for idx, ax in enumerate([''] + list(np.arange(2,7))):
    fig.add_annotation(x=0.5,y=1.12,
                        text=subplot_titles[idx],
                        showarrow=False,
                        yanchor='top',
                        xanchor='center',
                        yref='y{} domain'.format(ax),
                        xref='x{} domain'.format(ax))


#----------Axes properties 
fig.update_xaxes(title='n', row=3)
fig.update_yaxes(range=[-1,1])
fig.update_yaxes(title='x', col=1)


# Global y axis properties
fig.update_yaxes(showline=True,
                 ticks="outside",
                 tickwidth=tickwidth,
                 ticklen=ticklen,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 # title_standoff=yaxes_standoff,
                 )

# Global x axis properties
fig.update_xaxes(showline=True,
                 ticks="outside",
                 tickwidth=tickwidth,
                 ticklen=ticklen,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 # title_standoff=xaxes_standoff
                 )  





fig.update_layout(showlegend=False,
                  width=400, height=500,
                  margin={'l':40,'r':5,'b':40,'t':15},
                  font=dict(size=font_size, family=font_family),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  )




fig.write_image('../../results/figure_s2.png',
                scale=4)






