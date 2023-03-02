#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:03:01 2021

Make fig of ROC curves with inset showing histogram of highest DL probability

@author: Thoams M. Bury

"""


import time
start_time = time.time()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

# Import PIL for image tools
from PIL import Image

#-----------
# General fig params
#------------

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = px.colors.qualitative.Plotly # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
        'state':'gray',
        'smoothing':col_grays[2],
        'dl_bif':cols[0],
        'variance':cols[1],
        'ac':cols[2],
        'dl_fold':cols[3],  
        'dl_hopf':cols[4],
        'dl_branch':cols[5],
        'dl_null':'black',
     }

# Pixels to mm
mm_to_pixel = 96/25.4 # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183*mm_to_pixel/3 # 3 panels wide
fig_height = fig_width


font_size = 8
font_family = 'Times New Roman'
font_size_letter_label = 14
font_size_auc_text = 10


# AUC annotations
x_auc = 0.98
y_auc = 0.6
y_auc_sep = 0.065

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
linewidth_axes_inset = 0.5

axes_standoff = 0


# Scale up factor on image export
scale = 8 # default dpi=72 - nature=300-600



def make_roc_figure(df_roc, letter_label, title=''):
    ''' Make ROC figure (no inset)'''
        
    fig = go.Figure()
    
    
    # DL prediction any bif
    df_trace = df_roc[df_roc['ews']=='DL bif']
    auc_dl = df_trace.round(2)['auc'].iloc[0]
    fig.add_trace(
        go.Scatter(x=df_trace['fpr'],
                    y=df_trace['tpr'],
                    showlegend=False,
                    mode='lines',
                    line=dict(width=linewidth,
                              color=dic_colours['dl_bif'],
                              ),
                    )
        )
    
    
    # Variance plot
    df_trace = df_roc[df_roc['ews']=='Variance']
    auc_var = df_trace.round(2)['auc'].iloc[0]
    fig.add_trace(
        go.Scatter(x=df_trace['fpr'],
                    y=df_trace['tpr'],
                    showlegend=False,
                    mode='lines',
                    line=dict(width=linewidth,
                              color=dic_colours['variance'],
                              ),
                    )
        )
    
    # Lag-1  AC plot
    df_trace = df_roc[df_roc['ews']=='Lag-1 AC']
    auc_ac = df_trace.round(2)['auc'].iloc[0]
    fig.add_trace(
        go.Scatter(x=df_trace['fpr'],
                    y=df_trace['tpr'],
                    showlegend=False,
                    mode='lines',
                    line=dict(width=linewidth,
                              color=dic_colours['ac'],
                              ),
                    )
        )
    
    # Line y=x
    fig.add_trace(
        go.Scatter(x=np.linspace(0,1,100),
                    y=np.linspace(0,1,100),
                    showlegend=False,
                    line=dict(color='black',
                              dash='dot',
                              width=linewidth,
                              ),
                    )
        )
    

    #--------------
    # Add labels and titles
    #----------------------
    
    list_annotations = []
    
    label_annotation = dict(
            # x=sum(xrange)/2,
            x=0.02,
            y=1,
            text='<b>{}</b>'.format(letter_label),
            xref='paper',
            yref='paper',
            showarrow=False,
            font = dict(
                    color = 'black',
                    size = font_size_letter_label,
                    ),
            )


    
    annotation_auc_dl = dict(
            # x=sum(xrange)/2,
            x=x_auc,
            y=y_auc,
            text='A<sub>DL</sub>={:.2f}'.format(auc_dl),
            xref='paper',
            yref='paper',
            showarrow=False,
            font = dict(
                    color = 'black',
                    size = font_size_auc_text,
                    )
            )
        
    
    annotation_auc_var = dict(
            # x=sum(xrange)/2,
            x=x_auc,
            y=y_auc-y_auc_sep,
            text='A<sub>Var</sub>={:.2f}'.format(auc_var),
            xref='paper',
            yref='paper',
            showarrow=False,
            font = dict(
                    color = 'black',
                    size = font_size_auc_text,
                    )
            )    
    
    
    
    annotation_auc_ac = dict(
            # x=sum(xrange)/2,
            x=x_auc,
            y=y_auc-2*y_auc_sep,
            text='A<sub>AC</sub>={:.2f}'.format(auc_ac),
            xref='paper',
            yref='paper',
            showarrow=False,
            font = dict(
                    color = 'black',
                    size = font_size_auc_text,
                    )
            )    
    title_annotation = dict(
            # x=sum(xrange)/2,
            x=0.5,
            y=1,
            text=title,
            xref='paper',
            yref='paper',
            showarrow=False,
            font = dict(
                    color = "black",
                    size = font_size)
            ) 
      
    
    list_annotations.append(label_annotation)
    list_annotations.append(annotation_auc_dl)
    list_annotations.append(annotation_auc_var)
    list_annotations.append(annotation_auc_ac)
    # list_annotations.append(title_annotation)

    fig['layout'].update(annotations=list_annotations)
        
    
    #-------------
    # General layout properties
    #--------------
    
    # X axes properties
    fig.update_xaxes(
        title=dict(text='False positive',
                   standoff=axes_standoff,
                   ),
        range=[-0.04,1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals =np.arange(0,1.1,0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor='black',
        mirror=False,
        )
    
    
    # Y axes properties
    fig.update_yaxes(
        title=dict(text='True positive',
                   standoff=axes_standoff,
                   ),
        range=[-0.04,1.04],
        ticks="outside",
        tickvals=np.arange(0,1.1,0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor='black',
        mirror=False,
        )
    
    
    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30,r=5,b=15,t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        )

    return fig



import seaborn as sns

def make_inset_boxplot(df_dl_forced, target_bif, save_dir):
    '''
    Make inset boxplot that shows the value of the 
    DL weights where the predictions are made

    '''
    
    sns.set(
        style="ticks",
        rc={'figure.figsize': (2.5*1.05, 1.5*1.05),
            'axes.linewidth':0.5,
            'axes.edgecolor':'#333F4B',
            'xtick.color':'#333F4B',
            'xtick.major.width':0.5,
            'xtick.major.size':3,
            'text.color':'#333F4B',
            'font.family':'Times New Roman',
            # 'font.size':20,
            },
        font_scale=1.2,
        )
            

    plt.figure()
    
    bif_types = [str(i) for i in np.arange(1,6)]
    bif_labels = ['PD', 'NS', 'Fold', 'TC', 'PF']
    map_bif = dict(zip(bif_types, bif_labels))
    df_plot = df_dl_forced[bif_types].melt(var_name='bif_type', value_name='DL prob')
    df_plot['bif_label'] = df_plot['bif_type'].map(map_bif)
    
    color_main = '#A9A9A9'
    color_target = '#FFA15A'
    col_palette = {bif: color_main for bif in bif_labels}
    col_palette[target_bif] = color_target


    b = sns.boxplot(df_plot,
                    orient = 'h',
                    x = "DL prob", 
                    y = "bif_label",
                    width = 0.8,
                    palette = col_palette,
                    linewidth = 0.8,
                    showfliers=False,
                    )
    
    b.set(xlabel=None)
    b.set(ylabel=None)
    b.set_xticks([0, 0.5, 1])
    b.set_xticklabels(['0', '0.5', '1'])

    sns.despine(offset = 3, trim = True)
    b.tick_params(left=False, bottom=True)



    fig = b.get_figure()
    # fig.tight_layout()

    fig.savefig(save_dir, dpi=330, bbox_inches='tight', pad_inches=0)
        
    
    
    


def combine_roc_inset(path_roc, path_inset, path_out):
    ''' 
    Combine ROC plot and inset, and export to path_out
    '''
    
    # Import image
    img_roc = Image.open(path_roc)
    img_inset = Image.open(path_inset)
    
    # Get height and width of frame (in pixels)
    height = img_roc.height
    width = img_roc.width
    
    # Create frame
    dst = Image.new('RGB', (width,height), (255,255,255))
    
    # Pasete in images
    dst.paste(img_roc,(0,0))
    dst.paste(img_inset,(width-img_inset.width-60, 1100))
    
    dpi=96*8 # (default dpi) * (scaling factor)
    dst.save(path_out, dpi=(dpi,dpi))
    
    return
    


#-------
# Fox period-doubling
#--------

df_roc = pd.read_csv('../test_fox/output/df_roc.csv')
# df_counts = pd.read_csv('../test_fox/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_fox/output/df_dl_forced.csv')

fig_roc = make_roc_figure(df_roc, 'a')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 1, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'PD', 'temp_inset.png')

# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_fox_pd.png'

combine_roc_inset(path_roc, path_inset, path_out)


#-------
# Westerhoff NS
#--------

df_roc = pd.read_csv('../test_westerhoff/output/df_roc.csv')
# df_counts = pd.read_csv('../test_westerhoff/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_westerhoff/output/df_dl_forced.csv')

fig_roc = make_roc_figure(df_roc, 'b')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 2, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'NS', 'temp_inset.png')

# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_westerhoff_ns.png'

combine_roc_inset(path_roc, path_inset, path_out)



#-------
# Ricker fold
#--------
df_roc = pd.read_csv('../test_ricker/output/df_roc.csv')
# df_counts = pd.read_csv('../test_ricker/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_ricker/output/df_dl_forced.csv')

fig_roc = make_roc_figure(df_roc, 'c')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 3, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'Fold', 'temp_inset.png')


# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_ricker_fold.png'

combine_roc_inset(path_roc, path_inset, path_out)



#-------
# Kot transcritical
#--------
df_roc = pd.read_csv('../test_kot/output/df_roc.csv')
# df_counts = pd.read_csv('../test_kot/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_kot/output/df_dl_forced.csv')


fig_roc = make_roc_figure(df_roc, 'd')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 4, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'TC', 'temp_inset.png')


# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_kot_transcritical.png'

combine_roc_inset(path_roc, path_inset, path_out)


#-------
# Lorenz pitchfork
#--------
nsims = 2500
df_roc = pd.read_csv('../test_lorenz/output/df_roc.csv')
# df_counts = pd.read_csv('../test_lorenz/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_lorenz/output/df_dl_forced.csv')


fig_roc = make_roc_figure(df_roc, 'e')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 5, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'PF', 'temp_inset.png')


# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_lorenz_pitchfork.png'

combine_roc_inset(path_roc, path_inset, path_out)


#-------
# Heart data
#--------
df_roc = pd.read_csv('../test_chick_heart/output/df_roc.csv')
# df_counts = pd.read_csv('../test_chick_heart/output/df_fav_bif.csv')
df_dl_forced = pd.read_csv('../test_chick_heart/output/df_dl_pd_fixed.csv')

fig_roc = make_roc_figure(df_roc, 'f')
fig_roc.write_image('temp_roc.png', scale=scale)

# make_inset_bar(df_counts, 1, 'temp_inset.png')
make_inset_boxplot(df_dl_forced, 'PD', 'temp_inset.png')


# Combine figs and export
path_roc = 'temp_roc.png'
path_inset = 'temp_inset.png'
path_out = 'output/roc_heart.png'

combine_roc_inset(path_roc, path_inset, path_out)

#------------
# Combine ROC plots
#------------

#-----------------
# Fig 2 of manuscript: 8-panel figure for all models and empirical data
#-----------------

# # Early or late predictions
# timing = 'late'

list_filenames = ['roc_fox_pd',
                  'roc_westerhoff_ns',
                  'roc_ricker_fold',
                  'roc_kot_transcritical',
                  'roc_lorenz_pitchfork',
                  'roc_heart',
                  ]
list_filenames = ['output/{}.png'.format(s) for s in list_filenames]

list_img = []
for filename in list_filenames:
    img = Image.open(filename)
    list_img.append(img)

# Get heght and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width


# Create frame
dst = Image.new('RGB',(3*ind_width, 2*ind_height), (255,255,255))

# Paste in images
i=0
for y in np.arange(2)*ind_height:
    for x in np.arange(3)*ind_width:
        dst.paste(list_img[i], (x,y))
        i+=1


dpi=96*8 # (default dpi) * (scaling factor)
dst.save('../../results/figure_4_2.png',
          dpi=(dpi,dpi))

# Remove temporary images
import os
for filename in list_filenames+['temp_inset.png','temp_roc.png']:
    try:
        os.remove(filename)
    except:
        pass

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))











