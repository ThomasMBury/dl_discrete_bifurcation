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

def simulate_model(x0, fvals, sigma):
    '''
    Simulate a single realisaiton of model
    
    Parameters:
        x0 : initial condition of x
        bvals: 
        sigma : noise amplitude for x

    Returns
    -------
    df_traj: pd.DataFrame
        Realisation fo model

    '''
    
    # Model parameters
    r = 0.75 # growth rate
    k = 10 # carrying capacity
    h = 0.75 # half-saturation constant of harvesting function
    
        
    def iterate_model(x, f, sigma):
        '''
        Iterate the model one time step using the Euler-Maruyama scheme
        
        Parameters:
            x : State variable - normalised population density
            r : growth rate
            k : carrying capacity
            f : fishing rate
            h : sigmoid constant
            sigma : noise amplitude
        '''
        
        xi = np.random.normal()
        x_next  = x*np.exp(r*(1-x/k)) - f*x**2/(x**2+h**2) + sigma*xi
        
        return max(x_next, 0)
    

    
    # Simulation parameters
    tburn = 100 # burn-in period

    # Initialise arrays to store single time-series data
    t = np.arange(len(fvals))
    x_vals = np.zeros(len(fvals))

    # Run burn-in period
    for i in range(int(tburn)):
        x0 = iterate_model(x0, fvals[0], sigma)
        
    # Initial condition post burn-in period
    x_vals[0]=x0
    
    # Run simulation
    for i, f in enumerate(fvals[:-1]):
        x_vals[i+1] = iterate_model(x_vals[i], f, sigma)
            
    # Put data in df
    df_traj = pd.DataFrame(
        {'time':t, 'x': x_vals})

    return df_traj
    

def sim_rate_forcing(sigma, rate_forcing=2.36/500):
    '''
    Run a simulation with the bifurcation parameter varying at some defined rate
    
    Parameters:
        sigma: noise ampltiude
        rate_forcing: (change in T)/(change in time)
    
    Note change in T from start to bifurcation is 100.
    '''
    
    x0 = 10

    fstart = 0
    fcrit = 2.36
    ffinal = 1.5*fcrit
    fvals = np.arange(fstart, ffinal, rate_forcing)
    
    # Take transition time as the time when x crosses threhsold
    x_thresh = 4.5

    df_forced = simulate_model(x0, fvals, sigma)        
    series_forced = df_forced.set_index('time')['x']
    
    transition_thresh = int(df_forced[df_forced['x']<=x_thresh]['time'].iloc[0])-1
    transition_bif = int((fcrit-fstart)/rate_forcing)
    transition = min(transition_thresh, transition_bif)
    
    # Simulate a null trajectory with the same length as the pretransition section
    df_null = simulate_model(x0, [fstart]*transition, sigma)
    series_null = df_null.set_index('time')['x']
    
    return series_forced, transition, series_null


