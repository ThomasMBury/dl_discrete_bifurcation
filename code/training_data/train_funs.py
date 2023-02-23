#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:47:14 2022

Functions for generating training data

@author: tbury
"""


import numpy as np
import pandas as pd



def iterate_pd(x, mu, dict_coeffs, supercrit=True):
    '''
    Iterate model for period-doubling bifurcation
    Can be supercritical or subcritical
    
    x : state variable
    mu : bifurcation parameter
    dict_coeffs : dictionary
        Keys are integers representing order of higher-order term (4 upward)
        Values are weights/parameters for each higher-order term.
    supercrit: boolean
    '''
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x**order
        
    # Positive cubic coeff - supercritical bif
    # Negative cubic coeff - subcritical bif
    cubic_coeff = 1 if supercrit else -1
    x_next = -(1+mu)*x + cubic_coeff * x**3 + sum_hot
    
    return x_next
       

def iterate_ns(s, mu, theta, dict_coeffs_x, dict_coeffs_y, supercrit=True):
    '''
    Iterate model for Neimark-Sacker bifurcation.
    Can be super or subcritical
    
    s : state variable = (x,y)
    mu : bifurcation parameter
    theta: rotation number (theta=pi - period-doubling, theta=0 - fold)
    dict_coeffs : dictionary.
        Keys are integers representing order of higher-order term (4 upward)
        Values are lists contianing weights/parameters for each combination 
        of the given order. E.g. order=4 value=[0.234,0.134,0.266,0.341,0.211] for 
        each possibility of order 4: x^4, xy^3, x^2y^2, xy^3, y^4
    supercrit: boolean
        Supercritical or subcritical
        
    '''
    
    # Get sum of higher-order terms
    x,y = s
    sum_hot_x = 0
    sum_hot_y = 0
    for order in dict_coeffs_x.keys():
        for index in np.arange(0,order+1):    
            sum_hot_x += dict_coeffs_x[order][index] * x**(order-index) * y**index
            sum_hot_y += dict_coeffs_y[order][index] * x**(order-index) * y**index
    
    v_hot = np.array([sum_hot_x, sum_hot_y])
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Positive cubic coeff - subcritical bif
    # Negative cubic coeff - supercritical bif
    cubic_coeff = -1 if supercrit else 1
    
    # State vector update
    s_next = np.matmul((1+mu)*R,s) + cubic_coeff*np.matmul((x**2+y**2)*R,s) + v_hot
    
    return s_next


def iterate_fold(x, mu, dict_coeffs):
    '''
    Iterate model for fold bifurcation.
    
    x : state variable
    mu : bifurcation parameter
    dict_coeffs : dictionary. 
        Keys are integers representing order of higher-order term (3 upward)
        Values are weights/parameters for each higher-order term.
        
    Note that higher-order terms are centered on x=sqrt(mu) to retain this 
    as the equilibrium.
    '''
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * (x-np.sqrt(max(-mu,0)))**order
    
    x_next = x - mu - x**2 + sum_hot
    
    return x_next


def iterate_tc(x, mu, dict_coeffs):
    '''
    Iterate model for transcritical bifurcation.
    
    x : state variable
    mu : bifurcation parameter
    dict_coeffs : dictionary. 
        Keys are integers representing order of higher-order term (3 upward)
        Values are weights/parameters for each higher-order term.
        
    Note that higher-order terms are centered on x=sqrt(mu) to retain this 
    as the equilibrium.
    '''
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x**order
    
    x_next = x*(1+mu) - x**2 + sum_hot
    
    return x_next


def iterate_pf(x, mu, dict_coeffs, supercrit=True):
    '''
    Iterate model for pitchfork bifurcation.
    Can be super or subcritical
    
    x : state variable
    mu : bifurcation parameter
    dict_coeffs : dictionary. 
        Keys are integers representing order of higher-order term (3 upward)
        Values are weights/parameters for each higher-order term.
    supercrit: boolean
        Supercritical or subcritical
    '''
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x**order
    
    # Positive cubic coeff - subcritical bif
    # Negative cubic coeff - supercritical bif
    cubic_coeff = -1 if supercrit else 1
    
    x_next = x*(1+mu) + cubic_coeff * x**3 + sum_hot
    
    return x_next





def simulate_pd(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01, 
                max_order=10, dev_thresh=0.4, supercrit=True):
    '''
    Simulate a trajectory of the normal form for the supercritical PD
    bifurcation with the bifurcation parameter going from bl to bh
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: deviation from equilibrium that defines start of transition
        
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.
    '''

    # Initial condition
    x0 = 0
    # Time values 
    t = np.arange(0, tmax, 1)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t) 
    
    # Set random coefficients for higher-order terms
    dict_coeffs = {order: np.random.normal(0,1) for \
                   order in np.arange(4,max_order+1)}
    
    # # Set a proportion 'sparsity' to zero
    # sparsity = np.random.uniform(0,1)
    # list_order_tozero = np.random.choice(
    #     list(dict_coeffs.keys()),
    #     int((max_order-3)*sparsity), replace=False)
    # for order in list_order_tozero:
    #     dict_coeffs[order]=0
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size = len(t))
        
    # Run burn-in period on x0
    for i in range(int(tburn)):
        x0 = iterate_pd(x0,bl,dict_coeffs,supercrit) + dW_burn[i]
        # If blows up
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t))*np.nan
            return pd.DataFrame({'time':t, 'x':x})
            
            
    # Run simulation
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t)-1):
        x[i+1] = iterate_pd(x[i],b.iloc[i],dict_coeffs,supercrit) + dW[i]
        # Stop if transitioned
        if abs(x[i+1]) > dev_thresh:
            # Set rest of time series and previous point to NaN
            x[i:] = np.nan
            break
    
    return pd.DataFrame({'time':t, 'x':x})



def simulate_ns(bl=-1, bh=0, theta=np.pi/2, tmax=500, tburn=100, 
                sigma=0.01, max_order=10, dev_thresh=0.4, supercrit=True):
    '''
    Simulate a trajectory of the normal form for the NS bifurcation
    with the bifurcation parameter going from bl to bh
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        theta: frequency of oscillations
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: threshold deviation that defines start of transition
        
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.
    '''

    # Initial condition (set as equilibrium)
    s0 = np.array([0,0])
    # Time values 
    t = np.arange(0, tmax, 1)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t)
    
    # Set random coefficients for higher-order terms
    dict_coeffs_x = {}
    dict_coeffs_y = {}
    for order in np.arange(4,max_order+1):
        dict_coeffs_x[order] = np.random.normal(0,1,size=order+1)
        dict_coeffs_y[order] = np.random.normal(0,1,size=order+1)
    

    # # Set a proportion 'sparsity' to zero
    # sparsity = np.random.uniform(0,1)
    # list_order_tozero = np.random.choice(
    #     list(dict_coeffs.keys()),
    #     int((max_order-3)*sparsity), replace=False)
    # for order in list_order_tozero:
    #     dict_coeffs[order]=0
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = (int(tburn),2))
    dW = np.random.normal(loc=0, scale=sigma, size = (len(t),2))
        
    # Run burn-in period on s0
    for i in range(int(tburn)):
        s0 = iterate_ns(s0,bl,theta, dict_coeffs_x,dict_coeffs_y,supercrit) + dW_burn[i]
        # If blows up
        if np.linalg.norm(s0) > 1e6:
            print('Model diverged during burn in period')
            s = np.ones([len(t),2])*np.nan
            return pd.DataFrame({'time':t, 'x':s[:,0], 'y':s[:,1]})    
    
    
    # Run simulation
    s = np.zeros([len(t),2])
    s[0] = s0
    for i in range(len(t)-1):
        s[i+1] = iterate_ns(s[i],b.iloc[i], theta, dict_coeffs_x,dict_coeffs_y,supercrit) + dW[i]
        
        # Stop if transitioned
        if abs(s[i+1][0]) > dev_thresh:
            # Set rest of time series and previous point to NaN
            s[i:] = np.nan
            break

    return pd.DataFrame({'time':t, 'x':s[:,0], 'y':s[:,1]})
    



def simulate_fold(bl=-0.5, bh=0, tmax=500, tburn=100, sigma=0.01, 
                  max_order=10, dev_thresh=0.4, return_dev=True):
    '''
    Simulate a trajectory of the normal form for the fold
    bifurcation with the bifurcation parameter going from bl to bh
    
    Return deviation from analytical equilibrium (residuals)
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: threshold deviation that defines start of transition
        return_dev: boolean for whether to return deviations about equilibrium
            branch, or raw trajectory (only bifurcation where stable branch
            is not simply x=0)
            
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.
    '''

    # Initial condition (equilibrium)
    x0 = np.sqrt(-bl)
    # Time values 
    t = np.arange(0, tmax, 1)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t) 
    
    # Set random coefficients for higher-order terms
    # Note Fold only goes up to order 2 so include orders 3 and above.
    dict_coeffs = {order: np.random.normal(0,1) for \
                   order in np.arange(3,max_order+1)}
    
    # # Set a proportion 'sparsity' to zero
    # sparsity = np.random.uniform(0,1)
    # list_order_tozero = np.random.choice(
    #     list(dict_coeffs.keys()),
    #     int((max_order-3)*sparsity), replace=False)
    # for order in list_order_tozero:
    #     dict_coeffs[order]=0
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size = len(t))
        
    # Run burn-in period on x0
    for i in range(int(tburn)):
        x0 = iterate_fold(x0,bl,dict_coeffs) + dW_burn[i]
        # If blows up
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t))*np.nan
            return pd.DataFrame({'time':t, 'x':x})
    
    # Run simulation
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t)-1):
        x[i+1] = iterate_fold(x[i],b.iloc[i],dict_coeffs) + dW[i]
        # Stop if transitioned
        if abs(x[i+1]-np.sqrt(max(-b.loc[i+1],0))) > dev_thresh:
            # Set rest of time series and previous point to NaN
            x[i:] = np.nan
            break
        
    # Put in df
    df_traj = pd.DataFrame(
        {'time':t, 'x_raw':x, 'b':b})
    
    # Get residuals from equilibrium
    if return_dev:
        df_traj['x'] = df_traj['x_raw'] - np.sqrt((-b).apply(lambda x: max(x,0)))
    else:
        df_traj['x'] = df_traj['x_raw']
        
    return df_traj[['time','x']]




def simulate_tc(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4):
    '''
    Simulate a trajectory of the normal form for the transcritical
    bifurcation with the bifurcation parameter going from bl to bh
    
    Return deviation from analytical equilibrium (residuals)
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: threshold deviation that defines start of transition
        
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.        
    '''

    # Initial condition (equilibrium)
    x0 = 0
    # Time values 
    t = np.arange(0, tmax)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t) 
    
    # Set random coefficients for higher-order terms
    # Note TC only goes up to order 2 so include orders 3 and above.
    dict_coeffs = {order: np.random.normal(0,1) for \
                   order in np.arange(3,max_order+1)}
    
    # # Set a proportion 'sparsity' to zero
    # sparsity = np.random.uniform(0,1)
    # list_order_tozero = np.random.choice(
    #     list(dict_coeffs.keys()),
    #     int((max_order-3)*sparsity), replace=False)
    # for order in list_order_tozero:
    #     dict_coeffs[order]=0
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size = len(t))
        
    # Run burn-in period on x0
    for i in range(int(tburn)):
        x0 = iterate_tc(x0,bl,dict_coeffs) + dW_burn[i]
        # If blows up
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t))*np.nan
            return pd.DataFrame({'time':t, 'x':x})        
        
        
    # Run simulation
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t)-1):
        x[i+1] = iterate_tc(x[i],b.iloc[i],dict_coeffs) + dW[i]
        # Stop if transitioned
        if abs(x[i+1]) > dev_thresh:
            # Set rest of time series and previous point to NaN
            x[i:] = np.nan
            break
    
    return pd.DataFrame({'time':t, 'x':x})



def simulate_pf(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01, 
                max_order=10, dev_thresh=0.4, supercrit=True):
    '''
    Simulate a trajectory of the normal form for the pitchfork
    bifurcation with the bifurcation parameter going from bl to bh
    
    Return deviation from analytical equilibrium (residuals)
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: threshold deviation that defines start of transition
        
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.         
    '''

    # Initial condition (equilibrium)
    x0 = 0
    # Time values 
    t = np.arange(0, tmax, 1)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t) 
    
    # Set random coefficients for higher-order terms
    # Note PF goes up to order 3 so include orders 4 and above.
    dict_coeffs = {order: np.random.normal(0,1) for \
                   order in np.arange(4,max_order+1)}
    
    # # Set a proportion 'sparsity' to zero
    # sparsity = np.random.uniform(0,1)
    # list_order_tozero = np.random.choice(
    #     list(dict_coeffs.keys()),
    #     int((max_order-3)*sparsity), replace=False)
    # for order in list_order_tozero:
    #     dict_coeffs[order]=0
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size = len(t))
        
    # Run burn-in period on x0
    for i in range(int(tburn)):
        x0 = iterate_pf(x0,bl,dict_coeffs,supercrit) + dW_burn[i]
        # If blows up
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t))*np.nan
            return pd.DataFrame({'time':t, 'x':x})
        
    # Run simulation
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t)-1):
        x[i+1] = iterate_pf(x[i],b.iloc[i],dict_coeffs,supercrit) + dW[i]
        # Stop if transitioned
        if abs(x[i+1]) > dev_thresh:
            # Set rest of time series and previous point to NaN
            x[i:] = np.nan
            break
        
    
    return pd.DataFrame({'time':t, 'x':x})




