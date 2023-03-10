#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17 Feb 2022

Generate training data for DL classifier.
Uses normal forms + randomly generated higher-order terms.

Key for trajectory type:
    0 : Null trajectory
    1 : Period-doubling trajectory
    2 : Neimark-Sacker trajectory
    3 : Fold trajectory
    4 : Transcritical trajectory
    5 : Pitchfork trajectory

@author: tbury
"""

import time
start = time.time()

import numpy as np
import pandas as pd

import train_funs as funs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nsims', type=int, 
                    help='Number of simulations for each model',
                    default=200)
parser.add_argument('--verbose', type=int, choices=[0,1], default=1)

args = parser.parse_args()
nsims = args.nsims
verbose = bool(args.verbose)

# Fix random number seed for reproducibility
np.random.seed(0)

tmax = 600
tburn = 100
ts_len = 500 #  length of time series to store for training
max_order = 10 # Max polynomial degree

# Noise amplitude distribution parameters (uniform dist.)
sigma_min = 0.005
sigma_max = 0.015

# Number of standard deviations from equilibrium that defines a transition
thresh_std = 10

# List to collect time series
list_df = []
tsid = 1 # time series id



print('Run forced PD simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcation parameter at random
    bl = np.random.uniform(-1.8,-0.2)
    # Eigenvalues lambda = -(1+mu)
    # Lower bound csp to lambda=0.8
    # Upper bound csp to lambda=-0.8
    bh = 0
    
    # Choose at random whether super or subcritical
    supercrit = bool(np.random.choice([0,1]))
    
    # Run simulation
    df_traj = funs.simulate_pd(bl=bl, bh=bh, tmax=tmax, tburn=tburn, 
                               sigma=sigma, max_order=max_order, 
                               dev_thresh=dev_thresh, supercrit=supercrit)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Forced PD trajectory diverged too early with sigma={}'.format(sigma))
        continue
    
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='pd_forced'
    df_traj['type'] = 1
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1




print('Run null PD simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcation parameter at random
    bl = np.random.uniform(-1.8,-0.2)
    bh = bl
    supercrit = bool(np.random.choice([0,1]))
    
    # Run simulation
    df_traj = funs.simulate_pd(bl=bl, bh=bh, tmax=tmax, tburn=tburn, 
                               sigma=sigma, max_order=max_order, 
                               dev_thresh=dev_thresh, supercrit=supercrit)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Null PD trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='pd_null'
    df_traj['type'] = 0
    df_traj['tsid']=tsid
    df_traj['sigma'] = sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1

  


print('Run forced NS simulations')
j=1
while j <= nsims:

    # Note: Amplitude of oscillations in NS are ~sqrt(-mu)

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalues lambda = (1+mu)*exp(+/-i*theta)
    # Lower bound csp to |lambda|=0.8
    # Upper bound csp to |lambda|=0.8 (opposite direction)
    bl = np.random.uniform(-1.8,-0.2)
    bh = 0
    supercrit = bool(np.random.choice([0,1]))
    
    # Set value for theta (frequency of oscillations)
    theta = np.random.uniform(0,np.pi)        
    # Run simulation
    df_traj = funs.simulate_ns(bl=bl, bh=bh, theta=theta, tmax=tmax, 
                               tburn=tburn, sigma=sigma, max_order=max_order, 
                               dev_thresh=dev_thresh, supercrit=supercrit)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Forced NS trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    
    
    df_traj['bif_type']='ns_forced'
    df_traj['type'] = 2
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1
   
    
print('Run null NS simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-1.8,-0.2)
    bh = bl
    supercrit = bool(np.random.choice([0,1]))

    # Set value for theta (frequency of oscillations)
    theta = np.random.uniform(0,np.pi)        
    
    # Run simulation
    df_traj = funs.simulate_ns(bl=bl, bh=bh, theta=theta, tmax=tmax, 
                               tburn=tburn, sigma=sigma, max_order=max_order, 
                               dev_thresh=dev_thresh, supercrit=supercrit)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Null NS trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    
    
    df_traj['bif_type']='ns_null'
    df_traj['type'] = 0
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1
    
    


print('Run forced fold simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-0.9,-0.1)
    # Eigenvalue lambda = 1-2*sqrt(-mu)
    # Lower bound csp to lambda=-0.8
    # Upper bound csp to lambda=0.8
    bh = 0
    
    # Run simulation
    df_traj = funs.simulate_fold(bl=bl, bh=bh, tmax=tmax, tburn=tburn, 
                                 sigma=sigma, max_order=max_order, 
                                 dev_thresh=dev_thresh)
    
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Forced fold trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='fold_forced'
    df_traj['type'] = 3
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1




print('Run null fold simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-0.9,-0.1)
    bh = bl
    
    # Run simulation
    df_traj = funs.simulate_fold(bl, bl, tmax, tburn, sigma, max_order, dev_thresh)
    
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Null fold trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='fold_null'
    df_traj['type'] = 0
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1


        
    
print('Run forced TC simulations')
j=1
while j <= nsims:
    
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalue lambda = 1+mu
    # Lower bound csp to lambda=-0.8
    # Upper bound csp to lambda=0.8
    bl = np.random.uniform(-1.8,-0.2)
    bh = 0
    
    # Run simulation
    df_traj = funs.simulate_tc(bl, bh, tmax, tburn, sigma, max_order, dev_thresh)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Forced TC trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='tc_forced'
    df_traj['type'] = 4
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1




print('Run null TC simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-1.8,-0.2)
    bh = bl
    
    # Run simulation
    df_traj = funs.simulate_tc(bl, bl, tmax, tburn, sigma, max_order, dev_thresh)
    
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Null TC trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='tc_null'
    df_traj['type'] = 0
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1



    
print('Run forced PF simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalue lambda = 1+mu
    # Lower bound csp to lambda=-0.8
    # Upper bound csp to lambda=0.8
    bl = np.random.uniform(-1.8,-0.2)
    bh = 0
    supercrit = bool(np.random.choice([0,1]))

    
    # Run simulation
    df_traj = funs.simulate_pf(bl, bh, tmax, tburn, sigma, max_order, dev_thresh,supercrit)
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Forced PF trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='pf_forced'
    df_traj['type'] = 5
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1



print('Run null PF simulations')
j=1
while j <= nsims:

    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)
    
    # Deviation that defines transition
    dev_thresh = sigma * thresh_std
    
    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-1.8,-0.2)
    bh = bl
    supercrit = bool(np.random.choice([0,1]))

    
    # Run simulation
    df_traj = funs.simulate_pf(bl, bl, tmax, tburn, sigma, max_order, dev_thresh, supercrit)
    
    
    # Drop Nans and keep only last ts_len points
    df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
    if len(df_traj)<ts_len:
        if verbose:
            print('Null PF trajectory diverged too early with sigma={}'.format(sigma))
        continue
    df_traj['time_reset'] = np.arange(0,ts_len)
    df_traj['bif_type']='pf_null'
    df_traj['type'] = 0
    df_traj['tsid']=tsid
    df_traj['sigma']=sigma
    df_traj['supercrit'] = supercrit
    list_df.append(df_traj)
    if verbose:
        print('Complete for tsid={}'.format(tsid))
    elif tsid%1000==0:
        print('Complete for tsid={}'.format(tsid))
    tsid+=1
    j+=1
    

#-------------
# Concatenate simulations and prepare for export
#-------------

df_full = pd.concat(list_df, ignore_index=True)

# Make classes of equal size : currently 5 times number of samples in null section
# Get null class
tsid_null = df_full[df_full['type']==0]['tsid'].unique()
# Downsample - Take random selection 
tsid_null_down = np.random.choice(tsid_null, 
                                  size=int(len(tsid_null)/5),
                                  replace=False)
df_null = df_full[df_full['tsid'].isin(tsid_null_down)]
df_not_null = df_full[df_full['type']!=0]
df_full_balanced = pd.concat((df_null, df_not_null))


# Set types to save disk space
df_out = pd.DataFrame()
df_out['tsid'] = df_full_balanced['tsid'].astype('int32')
df_out['time'] = df_full_balanced['time_reset'].astype('int32')
df_out['x'] = df_full_balanced['x'].astype('float32')
df_out['type'] = df_full_balanced['type'].astype('category')
df_out['bif_type'] = df_full_balanced['bif_type'].astype('category')


# Split into train/validation/test (include shuffle)
split_ratio = (0.95,0.025,0.025)
tsid_values = df_out['tsid'].unique()
max_index_train = int(split_ratio[0]*len(tsid_values))
max_index_val = int((split_ratio[0]+split_ratio[1])*len(tsid_values)) 

np.random.shuffle(tsid_values)
tsid_train = tsid_values[:max_index_train]
tsid_val = tsid_values[max_index_train:max_index_val]
tsid_test = tsid_values[max_index_val:]

df_out_train = df_out[df_out['tsid'].isin(tsid_train)]
df_out_val = df_out[df_out['tsid'].isin(tsid_val)]
df_out_test = df_out[df_out['tsid'].isin(tsid_test)]

df_out_train.to_parquet('output/df_train.parquet')
df_out_val.to_parquet('output/df_val.parquet')
df_out_test.to_parquet('output/df_test.parquet')

end = time.time()
print('Script took {:0.1f} seconds'.format(end-start))


