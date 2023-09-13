#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:42:44 2023

Fig 2x2 with rows showing
1. bifurcation diagrams for Fox model with different parameters
2. ROC curves for DL performance

Cols showing
1. varying alpha
2. varying scaling factor

@author: tbury
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

# -----------
# General fig params
# ------------

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "alpha_0": cols[0],
    "alpha_0.1": cols[1],
    "alpha_0.2": cols[2],
    "alpha_0.3": cols[3],
    "alpha_0.4": cols[4],
    "scale_up_0.8": cols[0],
    "scale_up_0.9": cols[1],
    "scale_up_1": cols[2],
    "scale_up_1.1": cols[3],
    "scale_up_1.2": cols[4],
}


# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 2  # 3 panels wide
fig_height = fig_width * 0.7


font_size = 12
font_family = "Times New Roman"
font_size_letter_label = 16
font_size_auc_text = 12


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
scale = 8  # default dpi=72 - nature=300-600


# -------
# Bifurcation fig for different alpha
# --------

df_bif_alpha = pd.read_csv("output/df_bif_alpha.csv")

marker_size = 1.5

fig = go.Figure()

# Alpha 0
alpha = 0
fig.add_trace(
    go.Scatter(
        x=df_bif_alpha.query("alpha==@alpha")["T"],
        y=df_bif_alpha.query("alpha==@alpha")["D"],
        # showlegend=False,
        name="0",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["alpha_0"],
        ),
    )
)

alpha = 0.1
fig.add_trace(
    go.Scatter(
        x=df_bif_alpha.query("alpha==@alpha")["T"],
        y=df_bif_alpha.query("alpha==@alpha")["D"],
        # showlegend=False,
        name="0.1",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["alpha_0.1"],
        ),
    )
)

alpha = 0.2
fig.add_trace(
    go.Scatter(
        x=df_bif_alpha.query("alpha==@alpha")["T"],
        y=df_bif_alpha.query("alpha==@alpha")["D"],
        # showlegend=False,
        name="0.2",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["alpha_0.2"],
        ),
    )
)

alpha = 0.3
fig.add_trace(
    go.Scatter(
        x=df_bif_alpha.query("alpha==@alpha")["T"],
        y=df_bif_alpha.query("alpha==@alpha")["D"],
        # showlegend=False,
        name="0.3",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["alpha_0.3"],
        ),
    )
)

alpha = 0.4
fig.add_trace(
    go.Scatter(
        x=df_bif_alpha.query("alpha==@alpha")["T"],
        y=df_bif_alpha.query("alpha==@alpha")["D"],
        # showlegend=False,
        name="0.4",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["alpha_0.4"],
        ),
    )
)


fig.update_yaxes(range=[75, 205])
fig.update_xaxes(range=[100, 300])


list_annotations = []
label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("a"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)
list_annotations.append(label_annotation)
fig["layout"].update(annotations=list_annotations)


# X axes properties
fig.update_xaxes(
    title=dict(
        text="T (ms)",
        standoff=axes_standoff,
    ),
    ticks="outside",
    tickwidth=tickwidth,
    # tickvals =np.arange(0,1.1,0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="D (ms)",
        standoff=axes_standoff,
    ),
    ticks="outside",
    # tickvals=np.arange(0,1.1,0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(
        x=0.8,
        y=0.02,
        title="    alpha",
        itemsizing="constant",
        title_font_family="Times New Roman",
    ),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

fig.write_html("figures/fig_bif_alpha.html")
fig.write_image("figures/fig_bif_alpha.png", scale=scale)


# -------
# Bifurcation fig for different scale
# --------

df_bif_scale = pd.read_csv("output/df_bif_scaleup.csv")

marker_size = 1.5

fig = go.Figure()

# # Scale up 0.7
# scale_up=0.7
# fig.add_trace(
#     go.Scatter(x=df_bif_scale.query('scale_up==@scale_up')['T'],
#                 y=df_bif_scale.query('scale_up==@scale_up')['D'],
#                 # showlegend=False,
#                 name='0.7',
#                 mode='markers',
#                 marker=dict(size=marker_size),
#                 )
#     )

# Scale up 0.8
scale_up = 0.8
fig.add_trace(
    go.Scatter(
        x=df_bif_scale.query("scale_up==@scale_up")["T"],
        y=df_bif_scale.query("scale_up==@scale_up")["D"],
        # showlegend=False,
        name="0.8",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["scale_up_0.8"],
        ),
    )
)

# Scale up 0.9
scale_up = 0.9
fig.add_trace(
    go.Scatter(
        x=df_bif_scale.query("scale_up==@scale_up")["T"],
        y=df_bif_scale.query("scale_up==@scale_up")["D"],
        # showlegend=False,
        name="0.9",
        mode="markers",
        marker=dict(
            size=marker_size,
            color=dic_colours["scale_up_0.9"],
        ),
    )
)

# Scale up 1.0
scale_up = 1.0
fig.add_trace(
    go.Scatter(
        x=df_bif_scale.query("scale_up==@scale_up")["T"],
        y=df_bif_scale.query("scale_up==@scale_up")["D"],
        # showlegend=False,
        name="1.0",
        mode="markers",
        marker=dict(size=marker_size, color=dic_colours["scale_up_1"]),
    )
)

# Scale up 1.1
scale_up = 1.1
fig.add_trace(
    go.Scatter(
        x=df_bif_scale.query("scale_up==@scale_up")["T"],
        y=df_bif_scale.query("scale_up==@scale_up")["D"],
        # showlegend=False,
        name="1.1",
        mode="markers",
        marker=dict(size=marker_size, color=dic_colours["scale_up_1.1"]),
    )
)

# Scale up 1.2
scale_up = 1.2
fig.add_trace(
    go.Scatter(
        x=df_bif_scale.query("scale_up==@scale_up")["T"],
        y=df_bif_scale.query("scale_up==@scale_up")["D"],
        # showlegend=False,
        name="1.2",
        mode="markers",
        marker=dict(size=marker_size, color=dic_colours["scale_up_1.2"]),
    )
)

fig.update_yaxes(range=[75, 220])
fig.update_xaxes(range=[100, 300])


# -------------
# General layout properties
# --------------

list_annotations = []
label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("b"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)
list_annotations.append(label_annotation)
fig["layout"].update(annotations=list_annotations)

# X axes properties
fig.update_xaxes(
    title=dict(
        text="T (ms)",
        standoff=axes_standoff,
    ),
    ticks="outside",
    tickwidth=tickwidth,
    # tickvals =np.arange(0,1.1,0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="D (ms)",
        standoff=axes_standoff,
    ),
    ticks="outside",
    # tickvals=np.arange(0,1.1,0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)

# Overall properties
fig.update_layout(
    legend=dict(
        x=0.05,
        y=1,
        title="    scale",
        itemsizing="constant",
        title_font_family="Times New Roman",
    ),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('figures/fig_bif_scale.html')
fig.write_image("figures/fig_bif_scale.png", scale=scale)


# ------------
# Make ROC curve for variance at different alpha
# -------------

df_roc = pd.read_csv("output/df_roc_alpha.csv")
df_roc = df_roc[df_roc["ews"] == "Variance"]

fig = go.Figure()

# ROC for alpha 0
alpha = 0
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0"],
        ),
    )
)

# ROC for alpha 0.1
alpha = 0.1
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.1"],
        ),
    )
)

# ROC for alpha 0.2
alpha = 0.2
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.2"],
        ),
    )
)

# ROC for alpha 0.3
alpha = 0.3
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.3"],
        ),
    )
)

# ROC for alpha 0.4
alpha = 0.4
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.4"],
        ),
    )
)

# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("c"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)


list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)


# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_alpha_var.png", scale=scale)


# ------------
# Make ROC curve for lag-1 AC at different alpha
# -------------

df_roc = pd.read_csv("output/df_roc_alpha.csv")
df_roc = df_roc[df_roc["ews"] == "Lag-1 AC"]

fig = go.Figure()

# ROC for alpha 0
alpha = 0
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0"],
        ),
    )
)

# ROC for alpha 0.1
alpha = 0.1
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.1"],
        ),
    )
)

# ROC for alpha 0.2
alpha = 0.2
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.2"],
        ),
    )
)

# ROC for alpha 0.3
alpha = 0.3
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.3"],
        ),
    )
)

# ROC for alpha 0.4
alpha = 0.4
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.4"],
        ),
    )
)

# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("e"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)


list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)


# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_alpha_ac.png", scale=scale)


# ------------
# Make ROC curve for DL at different alpha
# -------------

df_roc = pd.read_csv("output/df_roc_alpha.csv")
df_roc = df_roc[df_roc["ews"] == "DL bif"]

fig = go.Figure()

# ROC for alpha 0
alpha = 0
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0"],
        ),
    )
)

# ROC for alpha 0.1
alpha = 0.1
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.1"],
        ),
    )
)

# ROC for alpha 0.2
alpha = 0.2
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.2"],
        ),
    )
)

# ROC for alpha 0.3
alpha = 0.3
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.3"],
        ),
    )
)

# ROC for alpha 0.4
alpha = 0.4
df_trace = df_roc.query("alpha==@alpha")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["alpha_0.4"],
        ),
    )
)

# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("g"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)


list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)


# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_alpha_dl.png", scale=scale)


# ------------
# Make ROC curve for variance at different scale_up
# -------------

df_roc = pd.read_csv("output/df_roc_scale.csv")
df_roc = df_roc[df_roc["ews"] == "Variance"]

fig = go.Figure()


# ROC for scale up 0.8
scale_up = 0.8
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.8"],
        ),
    )
)

# ROC for scale up 0.9
scale_up = 0.9
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.9"],
        ),
    )
)

# ROC for scale up 1
scale_up = 1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1"],
        ),
    )
)

# ROC for scale up 1.1
scale_up = 1.1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.1"],
        ),
    )
)

# ROC for scale up 1.2
scale_up = 1.2
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.2"],
        ),
    )
)


# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("d"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)


list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)

# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_scale_var.png", scale=scale)


# ------------
# Make ROC curve for lag-1 ac at different scale_up
# -------------

df_roc = pd.read_csv("output/df_roc_scale.csv")
df_roc = df_roc[df_roc["ews"] == "Lag-1 AC"]

fig = go.Figure()


# ROC for scale up 0.8
scale_up = 0.8
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.8"],
        ),
    )
)

# ROC for scale up 0.9
scale_up = 0.9
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.9"],
        ),
    )
)

# ROC for scale up 1
scale_up = 1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1"],
        ),
    )
)

# ROC for scale up 1.1
scale_up = 1.1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.1"],
        ),
    )
)

# ROC for scale up 1.2
scale_up = 1.2
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.2"],
        ),
    )
)


# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("f"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)


list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)


# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_scale_ac.png", scale=scale)


# -------------
# Make ROC curve for DL at different scale_up
# -------------

df_roc = pd.read_csv("output/df_roc_scale.csv")
df_roc = df_roc[df_roc["ews"] == "DL bif"]

fig = go.Figure()

# ROC for scale up 0.8
scale_up = 0.8
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.8"],
        ),
    )
)

# ROC for scale up 0.9
scale_up = 0.9
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_0.9"],
        ),
    )
)

# ROC for scale up 1
scale_up = 1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1"],
        ),
    )
)

# ROC for scale up 1.1
scale_up = 1.1
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.1"],
        ),
    )
)

# ROC for scale up 1.2
scale_up = 1.2
df_trace = df_roc.query("scale==@scale_up")
auc_dl = df_trace.round(2)["auc"].iloc[0]
fig.add_trace(
    go.Scatter(
        x=df_trace["fpr"],
        y=df_trace["tpr"],
        showlegend=False,
        mode="lines",
        line=dict(
            width=linewidth,
            color=dic_colours["scale_up_1.2"],
        ),
    )
)

# Line y=x
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        showlegend=False,
        line=dict(
            color="black",
            dash="dot",
            width=linewidth,
        ),
    )
)


# --------------
# Add labels and titles
# ----------------------

list_annotations = []

label_annotation = dict(
    # x=sum(xrange)/2,
    x=0.02,
    y=1,
    text="<b>{}</b>".format("h"),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_letter_label,
    ),
)

annotation_auc = dict(
    # x=sum(xrange)/2,
    x=x_auc,
    y=y_auc,
    text="Mean AUC={:.2f}".format(df_roc["auc"].mean()),
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        color="black",
        size=font_size_auc_text,
    ),
)

list_annotations.append(label_annotation)
list_annotations.append(annotation_auc)

fig["layout"].update(annotations=list_annotations)


# -------------
# General layout properties
# --------------

# X axes properties
fig.update_xaxes(
    title=dict(
        text="False positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickwidth=tickwidth,
    tickvals=np.arange(0, 1.1, 0.2),
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)


# Y axes properties
fig.update_yaxes(
    title=dict(
        text="True positive",
        standoff=axes_standoff,
    ),
    range=[-0.04, 1.04],
    ticks="outside",
    tickvals=np.arange(0, 1.1, 0.2),
    tickwidth=tickwidth,
    showline=True,
    linewidth=linewidth_axes,
    linecolor="black",
    mirror=False,
)

# Overall properties
fig.update_layout(
    legend=dict(x=0.6, y=0),
    width=fig_width,
    height=fig_height,
    margin=dict(l=30, r=5, b=15, t=5),
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

# fig.write_html('temp.html')
fig.write_image("figures/fig_roc_scale_dl.png", scale=scale)


# ------------
# Inset box plot for DL weight
# -------------

import seaborn as sns


def make_inset_boxplot(df_dl_forced, target_bif, save_dir):
    """
    Make inset boxplot that shows the value of the
    DL weights where the predictions are made

    """

    sns.set(
        style="ticks",
        rc={
            "figure.figsize": (2.5 * 1.5, 1.5 * 1.5),
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#333F4B",
            "xtick.color": "#333F4B",
            "xtick.major.width": 0.5,
            "xtick.major.size": 3,
            "text.color": "#333F4B",
            "font.family": "Times New Roman",
            # 'font.size':20,
        },
        font_scale=1.5,
    )

    plt.figure()

    bif_types = [str(i) for i in np.arange(1, 6)]
    bif_labels = ["PD", "NS", "Fold", "TC", "PF"]
    map_bif = dict(zip(bif_types, bif_labels))
    df_plot = df_dl_forced[bif_types].melt(var_name="bif_type", value_name="DL prob")
    df_plot["bif_label"] = df_plot["bif_type"].map(map_bif)

    color_main = "#A9A9A9"
    color_target = "#FFA15A"
    col_palette = {bif: color_main for bif in bif_labels}
    col_palette[target_bif] = color_target

    b = sns.boxplot(
        df_plot,
        orient="h",
        x="DL prob",
        y="bif_label",
        width=0.8,
        palette=col_palette,
        linewidth=0.8,
        showfliers=False,
    )

    b.set(xlabel=None)
    b.set(ylabel=None)
    b.set_xticks([0, 0.5, 1])
    b.set_xticklabels(["0", "0.5", "1"])

    sns.despine(offset=3, trim=True)
    b.tick_params(left=False, bottom=True)

    fig = b.get_figure()
    # fig.tight_layout()

    fig.savefig(save_dir, dpi=330, bbox_inches="tight", pad_inches=0)


def combine_roc_inset(path_roc, path_inset, path_out):
    """
    Combine ROC plot and inset, and export to path_out
    """

    # Import image
    img_roc = Image.open(path_roc)
    img_inset = Image.open(path_inset)

    # Get height and width of frame (in pixels)
    height = img_roc.height
    width = img_roc.width

    # Create frame
    dst = Image.new("RGB", (width, height), (255, 255, 255))

    # Pasete in images
    dst.paste(img_roc, (0, 0))
    dst.paste(img_inset, (width - img_inset.width - 60, 900))

    dpi = 96 * 8  # (default dpi) * (scaling factor)
    dst.save(path_out, dpi=(dpi, dpi))

    return


# -------------
# Make inset for DL plot with varying alpha
# ------------

df_dl_forced = pd.read_csv("output/df_dl_forced_alpha.csv")
make_inset_boxplot(df_dl_forced, "PD", "figures/fig_inset_alpha.png")
combine_roc_inset(
    "figures/fig_roc_alpha_dl.png",
    "figures/fig_inset_alpha.png",
    "figures/fig_roc_alpha_dl_combine.png",
)


# -------------
# Make inset for DL plot with varying scale
# ------------

df_dl_forced = pd.read_csv("output/df_dl_forced_scale.csv")
make_inset_boxplot(df_dl_forced, "PD", "figures/fig_inset_scale.png")
combine_roc_inset(
    "figures/fig_roc_scale_dl.png",
    "figures/fig_inset_scale.png",
    "figures/fig_roc_scale_dl_combine.png",
)


# ------------
# Combine plots
# ------------

# -----------------
# Fig SXX of manuscript: 8-panel figure showing Fox model at diff params and ROC curves
# -----------------

list_filenames = [
    "fig_bif_alpha",
    "fig_bif_scale",
    "fig_roc_alpha_var",
    "fig_roc_scale_var",
    "fig_roc_alpha_ac",
    "fig_roc_scale_ac",
    "fig_roc_alpha_dl_combine",
    "fig_roc_scale_dl_combine",
]
list_filenames = ["figures/{}.png".format(s) for s in list_filenames]

list_img = []
for filename in list_filenames:
    img = Image.open(filename)
    list_img.append(img)

# Get heght and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width

# Create frame
dst = Image.new("RGB", (2 * ind_width, 4 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(4) * ind_height:
    for x in np.arange(2) * ind_width:
        dst.paste(list_img[i], (x, y))
        i += 1


dpi = 96 * 8  # (default dpi) * (scaling factor)
# dst.save('figures/fig_test_fox_params.png',
#           dpi=(dpi,dpi))
dst.save("../../results/figure_s7.png", dpi=(dpi, dpi))

# # Remove temporary images
# import os
# for filename in list_filenames+['temp_inset.png','temp_roc.png']:
#     try:
#         os.remove(filename)
#     except:
#         pass

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))
