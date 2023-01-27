#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:39:30 2022

@author: rodolfofreitas
"""

import numpy as np
import h5py
import os
import cantera as ct
from sklearn.model_selection import train_test_split
import torch as th

# Compute PaSR*
def omegas_batch(rho, tau_star, Y_star, Y_0):
    omega = rho * (Y_star - Y_0) / tau_star
    return omega


def read_data(case_name, filter_name,taum_name,tauc_name, n_sp, data_dir):
    
    # Progress variable
    filename_C = data_dir+'/{}_filtered_C_O2_filter_{}.h5'.format(case_name,filter_name)
    with h5py.File(filename_C, 'r') as f:
        key = 'C_O2'
        C = np.array(f[key]).T
        
    # Progress variable variance
    filename_C = data_dir+'/{}_filtered_C_O2_var_filter_{}.h5'.format(case_name,filter_name)
    with h5py.File(filename_C, 'r') as f:
        key = 'C_O2_var'
        C_var = np.array(f[key]).T
    
    # Progress variable Chi
    filename_C = data_dir+'/{}_filtered_Chi_C_O2_filter_{}.h5'.format(case_name,filter_name)
    with h5py.File(filename_C, 'r') as f:
        key = 'Chi_C_O2'
        Chi_C = np.array(f[key]).T
    
    filename_taum = data_dir+'/{}_filtered_tau_m_{}_filter_{}.h5'.format(case_name,taum_name,filter_name)
    with h5py.File(filename_taum, 'r') as f:
        key = 'tau_m'
        tau_m = np.array(f[key]).T
        
        
    filename_tauc = data_dir+'/{}_filtered_tau_c_{}_filter_{}.h5'.format(case_name,tauc_name,filter_name)
    with h5py.File(filename_tauc, 'r') as f:
        key = 'tau_c'
        tau_c = np.array(f[key]).T
    
    tau_star = np.minimum(tau_c,tau_m)
    
    filename = data_dir+'/{}_filtered_STATES_filter_{}.h5'.format(case_name,filter_name)#RHO,T,Y
    
    
    # Thermodynamical states 
    with h5py.File(filename, 'r') as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        group_key = list(f.keys())
        data = {}
        for key in group_key:
            #print(key)
            data[key] = np.array(f[key]).T
            
    fdimz   = data['T'].shape[0]
    fdimx   = data['T'].shape[1]
    fdimy   = data['T'].shape[2]
    
    # Chemical source terms from DNS and from the based reactor model
    R_DNS       = np.zeros((fdimz,fdimx,fdimy, n_sp))
    Y_star      = np.zeros((fdimz,fdimx,fdimy, n_sp))
    Y0_         = np.zeros((fdimz,fdimx,fdimy, n_sp))
    R_batch     = np.zeros((fdimz,fdimx,fdimy, n_sp))
    DH_         = np.zeros((fdimz,fdimx,fdimy, n_sp))
    HRR_batch   = np.zeros((fdimz,fdimx,fdimy))
    tau_star    = np.minimum(tau_c,tau_m)
    Da          = tau_m / tau_c 
    
    for s in range(n_sp):
        Y0_[:,:,:,s]        = data['Y_{:02d}'.format(s+1)]
        if s == n_sp-1: 
            Y0_[:,:,:,s]    = 1. - np.sum(Y0_[:,:,:,:-1],axis=3)
        Y_star[:,:,:,s]     = np.transpose((h5py.File(data_dir+'/{}_filtered_Ystar_{}_{}_filter_{}.h5'.format(case_name,taum_name,tauc_name,filter_name), 'r'))['Y{:02d}'.format(s+1)][:])
        R_DNS[:,:,:,s]      = np.transpose((h5py.File(data_dir+'/{}_filtered_R_filter_{}.h5'.format(case_name,filter_name), 'r'))['R_{:02d}'.format(s+1)][:]) # Reaction Rates from the model
        R_batch[:,:,:,s]    = omegas_batch(data['RHO'], tau_star, Y_star[:,:,:,s], Y0_[:,:,:,s]) 
        DH_[:,:,:,s]        = np.transpose((h5py.File(data_dir+'/{}_filtered_DH_filter_{}.h5'.format(case_name,filter_name), 'r'))['DH_{:02d}'.format(s+1)][:])
        HRR_batch           = HRR_batch - R_batch[:,:,:,s] * DH_[:,:,:,s]
    
    # Compute the PaSR model
    gamma = 1 / (1 + Da)
    R_PaSR    = np.zeros((fdimz,fdimx,fdimy,n_sp))
    
    'Create a gas mixture'
    gas = ct.Solution('../../Mechanism/XML/chem.xml')
    
    # Species name
    list_sp = []
    for s in range(n_sp):
        list_sp.append(gas.species_name(s))
        R_PaSR[:,:,:,s]     = gamma * R_batch[:,:,:,s]
        
    # Heat Release Rate - HRR = - deltaH * R 
    HRR_DNS         = -np.transpose((h5py.File(data_dir+'/{}_filtered_HRR_filter_{}.h5'.format(case_name,filter_name), 'r'))['HRR'][:])
    HRR_PaSR        = gamma * HRR_batch
        
        
    return fdimz, fdimx, fdimy, Da, data['T'], C, C_var, Chi_C, R_DNS, R_batch, R_PaSR, DH_, HRR_batch, HRR_DNS, HRR_PaSR 

def load_data(Da, T, C, C_var, Chi_C, fdimz, fdimx, fdimy, R_DNS, HRR_DNS, R_batch, HRR_batch, args, kwargs):


    # reshape data  
    C_              = C.reshape(fdimz*fdimx*fdimy,1)
    C_var_          = C_var.reshape(fdimz*fdimx*fdimy,1)
    Chi_C_          = Chi_C.reshape(fdimz*fdimx*fdimy,1)
    Da_             = Da.reshape(fdimz*fdimx*fdimy,1)
    T_              = T.reshape(fdimz*fdimx*fdimy,1)
    # Scaling the data
    T_              = (T_ - T_.mean(0)) / T_.std(0)
    Da_             = (np.log10(Da_) - np.log10(Da_).mean(0)) / np.log10(Da_).std(0)
    C_var_          = (np.log10(C_var_) - np.log10(C_var_).mean(0)) / np.log10(C_var_).std(0)
    Chi_C_          = (np.log10(Chi_C_) - np.log10(Chi_C_).mean(0)) / np.log10(Chi_C_).std(0)
    
    X               = np.hstack([Da_, T_, C_, C_var_, Chi_C_])
    R_DNS_          = R_DNS.reshape(fdimz*fdimx*fdimy,args.n_sp)
    HRR_DNS_        = HRR_DNS.reshape(fdimz*fdimx*fdimy,1)
    R_batch_        = R_batch.reshape(fdimz*fdimx*fdimy,args.n_sp)
    HRR_batch_      = HRR_batch.reshape(fdimz*fdimx*fdimy,1)
    
    # Shuffle and Split data in train, test and cross-validation sets
    Xtrain, Xtest, YtrainDNS, YtestDNS, Ytrainbatch, Ytestbatch, RtrainDNS, RtestDNS, Rtrainbatch, Rtestbatch = train_test_split(X, 
                                                                                                                            HRR_DNS_, 
                                                                                                                            HRR_batch_,
                                                                                                                            R_DNS_,
                                                                                                                            R_batch_,
                                                                                                                            train_size=args.train_size,
                                                                                                                            random_state=42,
                                                                                                                            shuffle=True)
    
    Xtrain = th.FloatTensor(Xtrain) 
    YtrainDNS = th.FloatTensor(YtrainDNS) 
    Ytrainbatch = th.FloatTensor(Ytrainbatch) 
    RtrainDNS = th.FloatTensor(RtrainDNS)
    Rtrainbatch = th.FloatTensor(Rtrainbatch)
    Xtest = th.FloatTensor(Xtest) 
    YtestDNS = th.FloatTensor(YtestDNS)
    Ytestbatch = th.FloatTensor(Ytestbatch) 
    RtestDNS = th.FloatTensor(RtestDNS)
    Rtestbatch = th.FloatTensor(Rtestbatch)
    
    print("total input data shape: {}".format(X.shape))
    print("total output data shape: {}".format(HRR_DNS_.shape))

    return Xtrain, Xtest, YtrainDNS, YtestDNS, Ytrainbatch, Ytestbatch, RtrainDNS, RtestDNS, Rtrainbatch, Rtestbatch