#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:39:17 2022

Make heatmap of AUC value at each combo of (RoF, sigma)
for variance, lag-1 AC, and DL probability
for each discrete-time model 

@author: tbury
"""


import time
start_time = time.time()

import numpy as np
import pandas as pd
import plotly.express as px

import sklearn.metrics as metrics

def roc_compute(truth_vals, indicator_vals):
    
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals,indicator_vals)
    
    # Compute AUC (area under curve)
    auc = metrics.auc(fpr, tpr)
    
    # Put into a DF
    dic_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds, 'auc':auc}
    df_roc = pd.DataFrame(dic_roc)

    return df_roc


#-------------
# Fox model
#------------

path = '../test_fox/output/'

# Import data
df_ktau_forced = pd.read_csv(path+'df_ktau_forced.csv')
df_ktau_null = pd.read_csv(path+'df_ktau_null.csv')
df_dl_forced = pd.read_csv(path+'df_dl_forced.csv')
df_dl_null = pd.read_csv(path+'df_dl_null.csv')

# Set truth values
df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0
df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

df_dl['p_bif'] = df_dl[['1','2','3','4','5']].sum(axis=1)

# Get rof and sigma values
rof_values = df_ktau_forced['rof'].unique()
sigma_values = df_ktau_forced['sigma'].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        
        df_dl_spec = df_dl[(df_dl['sigma']==sigma)&\
                           (df_dl['rof']==rof)]
        df_ktau_spec = df_ktau[(df_ktau['sigma']==sigma)&\
                               (df_ktau['rof']==rof)]    
        # AUC for variance
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['variance'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'var', 'auc':auc})
        
        # AUC for lag-1 AC
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = -df_ktau_spec['ac1'].values # negative value since decreasing AC1 before period-doubling
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'ac1', 'auc':auc})
        
        # AUC for DL probability
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_dl_spec['p_bif'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'p_bif', 'auc':auc})

df_auc_fox = pd.DataFrame(list_dict)

# What percentage of times does the DL have a better AUC
df_auc_pivot = df_auc_fox.pivot(index=['rof','sigma'], columns='indicator')['auc']
df_auc_pivot['best'] = df_auc_pivot[['ac1','var','p_bif']].idxmax(axis=1)
prop_dl_best = len(df_auc_pivot[df_auc_pivot['best']=='p_bif'])/len(df_auc_pivot)
print('For the Fox model, the DL performs best in {}% of cases'.format(prop_dl_best*100))


#-------------
# Westerhoff model
#------------

path = '../test_westerhoff/output/'

# Import data
df_ktau_forced = pd.read_csv(path+'df_ktau_forced.csv')
df_ktau_null = pd.read_csv(path+'df_ktau_null.csv')
df_dl_forced = pd.read_csv(path+'df_dl_forced.csv')
df_dl_null = pd.read_csv(path+'df_dl_null.csv')

# Set truth values
df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0
df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

df_dl['p_bif'] = df_dl[['1','2','3','4','5']].sum(axis=1)


# Get rof and sigma values
rof_values = df_ktau_forced['rof'].unique()
sigma_values = df_ktau_forced['sigma'].unique()



list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        
        df_dl_spec = df_dl[(df_dl['sigma']==sigma)&\
                           (df_dl['rof']==rof)]
        df_ktau_spec = df_ktau[(df_ktau['sigma']==sigma)&\
                               (df_ktau['rof']==rof)]    
        # AUC for variance
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['variance'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'var', 'auc':auc})
        
        # AUC for lag-1 AC
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['ac1'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'ac1', 'auc':auc})
        
        # AUC for DL probability
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_dl_spec['p_bif'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'p_bif', 'auc':auc})

df_auc_westerhoff = pd.DataFrame(list_dict)

# What percentage of times does the DL have a better AUC
df_auc_pivot = df_auc_westerhoff.pivot(index=['rof','sigma'], columns='indicator')['auc']
df_auc_pivot['best'] = df_auc_pivot[['ac1','var','p_bif']].idxmax(axis=1)
prop_dl_best = len(df_auc_pivot[df_auc_pivot['best']=='p_bif'])/len(df_auc_pivot)
print('For the Westerhoff model, the DL performs best in {}% of cases'.format(prop_dl_best*100))



#-------------
# Ricker model
#------------

path = '../test_ricker/output/'

# Import data
df_ktau_forced = pd.read_csv(path+'df_ktau_forced.csv')
df_ktau_null = pd.read_csv(path+'df_ktau_null.csv')
df_dl_forced = pd.read_csv(path+'df_dl_forced.csv')
df_dl_null = pd.read_csv(path+'df_dl_null.csv')

# Set truth values
df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0
df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

df_dl['p_bif'] = df_dl[['1','2','3','4','5']].sum(axis=1)


# Get rof and sigma values
rof_values = df_ktau_forced['rof'].unique()
sigma_values = df_ktau_forced['sigma'].unique()



list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        
        df_dl_spec = df_dl[(df_dl['sigma']==sigma)&\
                           (df_dl['rof']==rof)]
        df_ktau_spec = df_ktau[(df_ktau['sigma']==sigma)&\
                               (df_ktau['rof']==rof)]    
        # AUC for variance
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['variance'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'var', 'auc':auc})
        
        # AUC for lag-1 AC
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['ac1'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'ac1', 'auc':auc})
        
        # AUC for DL probability
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_dl_spec['p_bif'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'p_bif', 'auc':auc})

df_auc_ricker = pd.DataFrame(list_dict)

# What percentage of times does the DL have a better AUC
df_auc_pivot = df_auc_ricker.pivot(index=['rof','sigma'], columns='indicator')['auc']
df_auc_pivot['best'] = df_auc_pivot[['ac1','var','p_bif']].idxmax(axis=1)
prop_dl_best = len(df_auc_pivot[df_auc_pivot['best']=='p_bif'])/len(df_auc_pivot)
print('For the Ricker model, the DL performs best in {}% of cases'.format(prop_dl_best*100))



#-------------
# Kot model
#------------

path = '../test_kot/output/'

# Import data
df_ktau_forced = pd.read_csv(path+'df_ktau_forced.csv')
df_ktau_null = pd.read_csv(path+'df_ktau_null.csv')
df_dl_forced = pd.read_csv(path+'df_dl_forced.csv')
df_dl_null = pd.read_csv(path+'df_dl_null.csv')

# Set truth values
df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0
df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

df_dl['p_bif'] = df_dl[['1','2','3','4','5']].sum(axis=1)


# Get rof and sigma values
rof_values = df_ktau_forced['rof'].unique()
sigma_values = df_ktau_forced['sigma'].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        
        df_dl_spec = df_dl[(df_dl['sigma']==sigma)&\
                           (df_dl['rof']==rof)]
        df_ktau_spec = df_ktau[(df_ktau['sigma']==sigma)&\
                               (df_ktau['rof']==rof)]    
        # AUC for variance
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['variance'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'var', 'auc':auc})
        
        # AUC for lag-1 AC
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['ac1'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'ac1', 'auc':auc})
        
        # AUC for DL probability
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_dl_spec['p_bif'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'p_bif', 'auc':auc})

df_auc_kot = pd.DataFrame(list_dict)

# What percentage of times does the DL have a better AUC
df_auc_pivot = df_auc_kot.pivot(index=['rof','sigma'], columns='indicator')['auc']
df_auc_pivot['best'] = df_auc_pivot[['ac1','var','p_bif']].idxmax(axis=1)
prop_dl_best = len(df_auc_pivot[df_auc_pivot['best']=='p_bif'])/len(df_auc_pivot)
print('For the predator-prey model, the DL performs best in {}% of cases'.format(prop_dl_best*100))




#-------------
# Lorenz model
#------------

path = '../test_lorenz/output/'

# Import data
df_ktau_forced = pd.read_csv(path+'df_ktau_forced.csv')
df_ktau_null = pd.read_csv(path+'df_ktau_null.csv')
df_dl_forced = pd.read_csv(path+'df_dl_forced.csv')
df_dl_null = pd.read_csv(path+'df_dl_null.csv')

# Set truth values
df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0
df_dl_forced['truth_value'] = 1
df_dl_null['truth_value'] = 0

df_dl =  pd.concat([df_dl_forced, df_dl_null])
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

df_dl['p_bif'] = df_dl[['1','2','3','4','5']].sum(axis=1)


# Get rof and sigma values
rof_values = df_ktau_forced['rof'].unique()
sigma_values = df_ktau_forced['sigma'].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        
        df_dl_spec = df_dl[(df_dl['sigma']==sigma)&\
                           (df_dl['rof']==rof)]
        df_ktau_spec = df_ktau[(df_ktau['sigma']==sigma)&\
                               (df_ktau['rof']==rof)]    
        # AUC for variance
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['variance'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'var', 'auc':auc})
        
        # AUC for lag-1 AC
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_ktau_spec['ac1'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'ac1', 'auc':auc})
        
        # AUC for DL probability
        truth_vals = df_ktau_spec['truth_value'].values
        indicator_vals = df_dl_spec['p_bif'].values
        df_roc = roc_compute(truth_vals, indicator_vals)
        auc = df_roc['auc'].iloc[0]
        list_dict.append({'rof':rof, 'sigma':sigma, 'indicator':'p_bif', 'auc':auc})

df_auc_lorenz = pd.DataFrame(list_dict)

# What percentage of times does the DL have a better AUC
df_auc_pivot = df_auc_lorenz.pivot(index=['rof','sigma'], columns='indicator')['auc']
df_auc_pivot['best'] = df_auc_pivot[['ac1','var','p_bif']].idxmax(axis=1)
prop_dl_best = len(df_auc_pivot[df_auc_pivot['best']=='p_bif'])/len(df_auc_pivot)
print('For the Lorenz model, the DL performs best in {}% of cases'.format(prop_dl_best*100))








#-------------
# Make subplot heat map
#------------

from plotly.subplots import make_subplots
import plotly.graph_objects as go



fig = make_subplots(5, 3, horizontal_spacing=0.15, vertical_spacing=0.08)
    


# Fox model
df_auc = df_auc_fox
# Only plot from 0.5 to 1
df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z_ac = df_auc[df_auc['indicator']=='ac1'].pivot(index='sigma', columns='rof',values='auc')
z_var = df_auc[df_auc['indicator']=='var'].pivot(index='sigma', columns='rof',values='auc')
z_dl = df_auc[df_auc['indicator']=='p_bif'].pivot(index='sigma', columns='rof',values='auc')

xvals = ['{:.2g}'.format(x) for x in z_ac.columns]
yvals = ['{:.2g}'.format(x) for x in z_ac.index]

row=1
fig.add_trace(go.Heatmap(z=z_ac.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                         coloraxis='coloraxis',
                         ),
              row=row,col=1)
fig.add_trace(go.Heatmap(z=z_var.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=2)
fig.add_trace(go.Heatmap(z=z_dl.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=3)

    
# westerhoff model
df_auc = df_auc_westerhoff
# Only plot from 0.5 to 1
df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))


z_ac = df_auc[df_auc['indicator']=='ac1'].pivot(index='sigma', columns='rof',values='auc')
z_var = df_auc[df_auc['indicator']=='var'].pivot(index='sigma', columns='rof',values='auc')
z_dl = df_auc[df_auc['indicator']=='p_bif'].pivot(index='sigma', columns='rof',values='auc')

xvals = ['{:.2g}'.format(x) for x in z_ac.columns]
yvals = ['{:.2g}'.format(x) for x in z_ac.index]

row=2
fig.add_trace(go.Heatmap(z=z_ac.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=1)
fig.add_trace(go.Heatmap(z=z_var.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=2)
fig.add_trace(go.Heatmap(z=z_dl.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=3)

# ricker model
df_auc = df_auc_ricker
# Only plot from 0.5 to 1
df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))


z_ac = df_auc[df_auc['indicator']=='ac1'].pivot(index='sigma', columns='rof',values='auc')
z_var = df_auc[df_auc['indicator']=='var'].pivot(index='sigma', columns='rof',values='auc')
z_dl = df_auc[df_auc['indicator']=='p_bif'].pivot(index='sigma', columns='rof',values='auc')

xvals = ['{:.2g}'.format(x) for x in z_ac.columns]
yvals = ['{:.2g}'.format(x) for x in z_ac.index]

row=3
fig.add_trace(go.Heatmap(z=z_ac.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=1)
fig.add_trace(go.Heatmap(z=z_var.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=2)
fig.add_trace(go.Heatmap(z=z_dl.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=3)


# predator prey model
df_auc = df_auc_kot
# Only plot from 0.5 to 1
df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))


z_ac = df_auc[df_auc['indicator']=='ac1'].pivot(index='sigma', columns='rof',values='auc')
z_var = df_auc[df_auc['indicator']=='var'].pivot(index='sigma', columns='rof',values='auc')
z_dl = df_auc[df_auc['indicator']=='p_bif'].pivot(index='sigma', columns='rof',values='auc')

xvals = ['{:.2g}'.format(x) for x in z_ac.columns]
yvals = ['{:.2g}'.format(x) for x in z_ac.index]

row=4
fig.add_trace(go.Heatmap(z=z_ac.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                          ),
              row=row,col=1)
fig.add_trace(go.Heatmap(z=z_var.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=2)
fig.add_trace(go.Heatmap(z=z_dl.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=3)



# lorenz model
df_auc = df_auc_lorenz
# Only plot from 0.5 to 1
df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z_ac = df_auc[df_auc['indicator']=='ac1'].pivot(index='sigma', columns='rof',values='auc')
z_var = df_auc[df_auc['indicator']=='var'].pivot(index='sigma', columns='rof',values='auc')
z_dl = df_auc[df_auc['indicator']=='p_bif'].pivot(index='sigma', columns='rof',values='auc')

xvals = ['{:.2g}'.format(x) for x in z_ac.columns]
yvals = ['{:.2g}'.format(x) for x in z_ac.index]

row=5
fig.add_trace(go.Heatmap(z=z_ac.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=1)
fig.add_trace(go.Heatmap(z=z_var.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=2)
fig.add_trace(go.Heatmap(z=z_dl.values, x=xvals, y=yvals, 
                         zmin=0.5, zmax=1, 
                          coloraxis='coloraxis',
                         ),
              row=row,col=3)


font_annotation = 14
fig.add_annotation(
            x=0.05, y=1.04,
            xref='paper', yref='paper',
            text='Lag-1 AC',
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=0.49, y=1.04,
            xref='paper', yref='paper',
            text='Variance',
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=0.98, y=1.04,
            xref='paper', yref='paper',
            text='DL probability',
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=1.07, y=0.94,
            xref='paper', yref='paper',
            text='Fox',
            yanchor='middle',
            textangle=90,
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=1.07, y=0.76,
            xref='paper', yref='paper',
            text='Westerhoff',
            textangle=90,
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=1.07, y=0.5,
            xref='paper', yref='paper',
            text='Ricker',
            textangle=90,
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=1.07, y=0.23,
            xref='paper', yref='paper',
            text='Lotka-Volterra',
            textangle=90,
            showarrow=False,
            font=dict(size=font_annotation),
            )

fig.add_annotation(
            x=1.07, y=0.04,
            xref='paper', yref='paper',
            text='Lorenz',
            textangle=90,
            showarrow=False,
            font=dict(size=font_annotation),
            )



# Axes properties
fig.update_xaxes(title='RoF', row=5)
fig.update_yaxes(title='sigma', col=1)

fig.update_xaxes(automargin=False)
fig.update_yaxes(automargin=False)


fig.update_layout(coloraxis_colorbar=dict(x=1.1, title='AUC<br>score<br> '),
                  # automargin=False,
                   margin=dict(l=60, r=10, b=75, t=30),
                  )



fig.update_layout(width=650, height=900,
                  font=dict(family='Times New Roman'))



fig.write_image('../../results/figure_s4.png', scale=8)


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))



























