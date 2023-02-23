#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:51:19 2022

Generic functions stored here

@author: tbury
"""


# import python libraries
import numpy as np
import pandas as pd


def iterate_model(xn, yn, c, r, sigma_1, sigma_2):
    '''
    Iterate the model one time step using the Euler-Maruyama scheme
    
    Parameters:
        Mn : memory variable for beat n
        Dn : action potential duration for beat n
        T : period of pacing (bifurcation parameter)
    '''
    
    zeta_1 = np.random.normal()
    zeta_2 = np.random.normal()
    
    # Subsequent state variables
    xn_next = (r+1)*xn - r*xn**2 - c*xn*yn + sigma_1*zeta_1
    yn_next = c*xn*yn + sigma_2*zeta_2

    return xn_next, yn_next


def simulate_model(x0=0.5, y0=0.5, cvals=[0.5]*500, sigma_1=0.01, sigma_2=0.01):
    '''
    Simulate a single realisaiton of model
    
    Parameters:
        M0 : initial condition of memory variable
        D0 : intitial condition of apd
        Tstart : initial value of T (bifurcation param)
        Tend : final value of T (bifurcation param)
        tmax : length of time series
        sigma_1 : noise amplitude for M
        sigma_2 : noise amplitude for D

    Returns
    -------
    df_traj: pd.DataFrame
        Realisation fo model

    '''
    
    # Model parameters
    r = 0.5
    
    # Simulation parameters
    tburn = 100
    
    # Initialise arrays to store single time-series data
    t = np.arange(len(cvals))
    x_vals = np.zeros(len(cvals))
    y_vals = np.zeros(len(cvals))
    
    # Run burn-in period
    for i in range(int(tburn)):
        x0, y0 = iterate_model(x0,y0,cvals[0],r,sigma_1,sigma_2)
        
    # Initial condition post burn-in period
    x_vals[0]=x0
    y_vals[0]=y0
    
    # Run simulation
    for i, c in enumerate(cvals[:-1]):
        x_vals[i+1], y_vals[i+1] = iterate_model(
            x_vals[i],y_vals[i],c,r,sigma_1,sigma_2
            )
        
        # If simulation blows up, stop and insert Nan
        if np.linalg.norm([x_vals[i+1],y_vals[i+1]]) > 1e6:
            x_vals[i:] = np.nan
            y_vals[i:] = np.nan
            break
        
        # If variables go below zero, reset to zero (they are population numbers)
        if x_vals[i+1] < 0:
            x_vals[i+1] = 0
        if y_vals[i+1] < 0:
            y_vals[i+1] = 0    
    
    # Put data in df
    df_traj = pd.DataFrame(
        {'time':t,
         'x': x_vals,
         'y': y_vals})

    return df_traj
    


def sim_rate_forcing(sigma, rate_forcing=0.5/500):
    '''
    Run a simulation with the bifurcation parameter varying at some defined rate
    
    Parameters:
        sigma: noise ampltiude
        rate_forcing: (change in c)/(change in time)
    
    '''
    
    x0 = 1
    y0 = 0

    cstart = 0.5
    ccrit = 1
    cfinal = 1.25
    cvals = np.arange(cstart, cfinal, rate_forcing)
    
    # Take transition time as time at bifurcation
    transition = int((ccrit-cstart)/rate_forcing)

    df_forced = simulate_model(x0, y0, cvals, sigma, sigma)           
    series_forced = df_forced.set_index('time')['x']
    
    # Simulate a null trajectory with the same length as the pretransition section
    df_null = simulate_model(x0, y0, [cstart]*transition, sigma, sigma)   
    series_null = df_null.set_index('time')['x']
    
    return series_forced, transition, series_null



