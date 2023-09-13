#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:39:17 2022

Make heatmap of %(correct DL class) for each combo of (RoF, sigma)
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
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals, indicator_vals)

    # Compute AUC (area under curve)
    auc = metrics.auc(fpr, tpr)

    # Put into a DF
    dic_roc = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}
    df_roc = pd.DataFrame(dic_roc)

    return df_roc


# -------------
# Fox model
# ------------

path = "../test_fox/output/"

# Import data
df = pd.read_csv(path + "df_dl_forced.csv")
df = df.rename({str(i): i for i in np.arange(6)}, axis=1)

df["fav_bif"] = df[[1, 2, 3, 4, 5]].idxmax(axis=1)

correct_bif = 1

# Get rof and sigma values
rof_values = df["rof"].unique()
sigma_values = df["sigma"].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        df_spec = df[(df["sigma"] == sigma) & (df["rof"] == rof)]
        # % correct
        num_correct = len(df_spec[df_spec["fav_bif"] == correct_bif])
        num_total = len(df_spec)
        prop_correct = num_correct / num_total
        list_dict.append({"rof": rof, "sigma": sigma, "prop_correct": prop_correct})

df_pred_fox = pd.DataFrame(list_dict)


# -------------
# Westerhoff model
# ------------


path = "../test_westerhoff/output/"

# Import data
df = pd.read_csv(path + "df_dl_forced.csv")
df = df.rename({str(i): i for i in np.arange(6)}, axis=1)

df["fav_bif"] = df[[1, 2, 3, 4, 5]].idxmax(axis=1)

correct_bif = 2

# Get rof and sigma values
rof_values = df["rof"].unique()
sigma_values = df["sigma"].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        df_spec = df[(df["sigma"] == sigma) & (df["rof"] == rof)]
        # % correct
        num_correct = len(df_spec[df_spec["fav_bif"] == correct_bif])
        num_total = len(df_spec)
        prop_correct = num_correct / num_total
        list_dict.append({"rof": rof, "sigma": sigma, "prop_correct": prop_correct})

df_pred_westerhoff = pd.DataFrame(list_dict)


# -------------
# Ricker model
# ------------


path = "../test_ricker/output/"

# Import data
df = pd.read_csv(path + "df_dl_forced.csv")
df = df.rename({str(i): i for i in np.arange(6)}, axis=1)

df["fav_bif"] = df[[1, 2, 3, 4, 5]].idxmax(axis=1)

correct_bif = 3

# Get rof and sigma values
rof_values = df["rof"].unique()
sigma_values = df["sigma"].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        df_spec = df[(df["sigma"] == sigma) & (df["rof"] == rof)]
        # % correct
        num_correct = len(df_spec[df_spec["fav_bif"] == correct_bif])
        num_total = len(df_spec)
        prop_correct = num_correct / num_total
        list_dict.append({"rof": rof, "sigma": sigma, "prop_correct": prop_correct})

df_pred_ricker = pd.DataFrame(list_dict)


# -------------
# Kot model
# ------------


path = "../test_kot/output/"

# Import data
df = pd.read_csv(path + "df_dl_forced.csv")
df = df.rename({str(i): i for i in np.arange(6)}, axis=1)

df["fav_bif"] = df[[1, 2, 3, 4, 5]].idxmax(axis=1)

correct_bif = 4

# Get rof and sigma values
rof_values = df["rof"].unique()
sigma_values = df["sigma"].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        df_spec = df[(df["sigma"] == sigma) & (df["rof"] == rof)]
        # % correct
        num_correct = len(df_spec[df_spec["fav_bif"] == correct_bif])
        num_total = len(df_spec)
        prop_correct = num_correct / num_total
        list_dict.append({"rof": rof, "sigma": sigma, "prop_correct": prop_correct})

df_pred_kot = pd.DataFrame(list_dict)


# -------------
# Lorenz model
# ------------


path = "../test_lorenz/output/"

# Import data
df = pd.read_csv(path + "df_dl_forced.csv")
df = df.rename({str(i): i for i in np.arange(6)}, axis=1)

df["fav_bif"] = df[[1, 2, 3, 4, 5]].idxmax(axis=1)

correct_bif = 5

# Get rof and sigma values
rof_values = df["rof"].unique()
sigma_values = df["sigma"].unique()

list_dict = []

for rof in rof_values:
    for sigma in sigma_values:
        df_spec = df[(df["sigma"] == sigma) & (df["rof"] == rof)]
        # % correct
        num_correct = len(df_spec[df_spec["fav_bif"] == correct_bif])
        num_total = len(df_spec)
        prop_correct = num_correct / num_total
        list_dict.append({"rof": rof, "sigma": sigma, "prop_correct": prop_correct})

df_pred_lorenz = pd.DataFrame(list_dict)


# -------------
# Make subplot heat map
# ------------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(5, 1, vertical_spacing=0.08)


# Fox model
df = df_pred_fox
row = 1
# # Only plot from 0.5 to 1
# df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z = df.pivot(index="sigma", columns="rof", values="prop_correct")
xvals = ["{:.2g}".format(x) for x in z.columns]
yvals = ["{:.2g}".format(x) for x in z.index]

fig.add_trace(
    go.Heatmap(
        z=z.values,
        x=xvals,
        y=yvals,
        zmin=0,
        zmax=1,
        coloraxis="coloraxis",
    ),
    row=row,
    col=1,
)


# westerhoff model
df = df_pred_westerhoff
row = 2
# # Only plot from 0.5 to 1
# df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z = df.pivot(index="sigma", columns="rof", values="prop_correct")
xvals = ["{:.2g}".format(x) for x in z.columns]
yvals = ["{:.2g}".format(x) for x in z.index]

fig.add_trace(
    go.Heatmap(
        z=z.values,
        x=xvals,
        y=yvals,
        zmin=0,
        zmax=1,
        coloraxis="coloraxis",
    ),
    row=row,
    col=1,
)


# ricker model
df = df_pred_ricker
row = 3
# # Only plot from 0.5 to 1
# df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z = df.pivot(index="sigma", columns="rof", values="prop_correct")
xvals = ["{:.2g}".format(x) for x in z.columns]
yvals = ["{:.2g}".format(x) for x in z.index]

fig.add_trace(
    go.Heatmap(
        z=z.values,
        x=xvals,
        y=yvals,
        zmin=0,
        zmax=1,
        coloraxis="coloraxis",
    ),
    row=row,
    col=1,
)


# Kot model
df = df_pred_kot
row = 4
# # Only plot from 0.5 to 1
# df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z = df.pivot(index="sigma", columns="rof", values="prop_correct")
xvals = ["{:.2g}".format(x) for x in z.columns]
yvals = ["{:.2g}".format(x) for x in z.index]

fig.add_trace(
    go.Heatmap(
        z=z.values,
        x=xvals,
        y=yvals,
        zmin=0,
        zmax=1,
        coloraxis="coloraxis",
    ),
    row=row,
    col=1,
)


# lorenz model
df = df_pred_lorenz
row = 5
# # Only plot from 0.5 to 1
# df_auc['auc'] = df_auc['auc'].apply(lambda x: max(0.5, x))

z = df.pivot(index="sigma", columns="rof", values="prop_correct")
xvals = ["{:.2g}".format(x) for x in z.columns]
yvals = ["{:.2g}".format(x) for x in z.index]

fig.add_trace(
    go.Heatmap(
        z=z.values,
        x=xvals,
        y=yvals,
        zmin=0,
        zmax=1,
        coloraxis="coloraxis",
    ),
    row=row,
    col=1,
)


font_annotation = 14

x_annotation = 1.25
fig.add_annotation(
    x=x_annotation,
    y=0.94,
    xref="paper",
    yref="paper",
    text="Fox",
    yanchor="middle",
    textangle=90,
    showarrow=False,
    font=dict(size=font_annotation),
)

fig.add_annotation(
    x=x_annotation,
    y=0.76,
    xref="paper",
    yref="paper",
    text="Westerhoff",
    textangle=90,
    showarrow=False,
    font=dict(size=font_annotation),
)

fig.add_annotation(
    x=x_annotation,
    y=0.5,
    xref="paper",
    yref="paper",
    text="Ricker",
    textangle=90,
    showarrow=False,
    font=dict(size=font_annotation),
)

fig.add_annotation(
    x=x_annotation,
    y=0.23,
    xref="paper",
    yref="paper",
    text="Lotka-Volterra",
    textangle=90,
    showarrow=False,
    font=dict(size=font_annotation),
)

fig.add_annotation(
    x=x_annotation,
    y=0.04,
    xref="paper",
    yref="paper",
    text="Lorenz",
    textangle=90,
    showarrow=False,
    font=dict(size=font_annotation),
)


# Axes properties
fig.update_xaxes(title="RoF", row=5)
fig.update_yaxes(title="sigma", col=1)

fig.update_xaxes(automargin=False)
fig.update_yaxes(automargin=False)


fig.update_layout(
    width=320,
    height=900,
    font=dict(family="Times New Roman"),
    coloraxis=dict(cmin=0, cmax=1),
    coloraxis_colorbar=dict(
        x=1.5,
        title="Proportion<br>correct<br>",
    ),
    margin=dict(l=60, r=10, b=75, t=30),
)


fig.write_image("../../results/figure_s6.png", scale=8)


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))
